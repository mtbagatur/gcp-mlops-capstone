terraform {
  required_version = ">= 1.6"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
}

provider "google" {
  project               = var.project_id
  region                = var.region
  billing_project       = var.project_id
  user_project_override = true
}

locals {
  services = [
    "run.googleapis.com",
    "cloudbuild.googleapis.com",
    "artifactregistry.googleapis.com",
    "bigquery.googleapis.com",
    "storage.googleapis.com",
    "aiplatform.googleapis.com",
    "billingbudgets.googleapis.com",
    "logging.googleapis.com",
    "monitoring.googleapis.com",
  ]
}

resource "google_project_service" "enabled" {
  for_each           = toset(local.services)
  service            = each.value
  disable_on_destroy = false
}

# Model artifacts and pipeline outputs. Versioned so a bad model can be rolled back.
resource "google_storage_bucket" "mlops" {
  name                        = "${var.project_id}-mlops"
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = true

  versioning {
    enabled = true
  }
}

# Must be US: the public census table lives in the US multi-region and
# BigQuery cannot query across locations.
resource "google_bigquery_dataset" "mlops" {
  dataset_id                 = "mlops"
  location                   = "US"
  delete_contents_on_destroy = true
  depends_on                 = [google_project_service.enabled]
}

resource "google_artifact_registry_repository" "mlops" {
  repository_id = "mlops"
  location      = var.region
  format        = "DOCKER"
  depends_on    = [google_project_service.enabled]

  cleanup_policies {
    id     = "keep-last-10"
    action = "KEEP"
    most_recent_versions {
      keep_count = 10
    }
  }
}

# Runtime identity for the API: reads model artifacts and calls Gemini, nothing else.
resource "google_service_account" "api" {
  account_id   = "income-api"
  display_name = "income-api Cloud Run runtime"
}

# Gemini calls for /predict-explained.
resource "google_project_iam_member" "api_calls_gemini" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.api.email}"
}

resource "google_storage_bucket_iam_member" "api_reads_models" {
  bucket = google_storage_bucket.mlops.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.api.email}"
}

# Budget alert (requirement G10). Emails go to the billing account's admins.
resource "google_billing_budget" "mlops" {
  billing_account = var.billing_account
  display_name    = "mlops-capstone"

  budget_filter {
    projects = ["projects/${data.google_project.this.number}"]
  }

  amount {
    # No currency_code: the budget uses the billing account's own currency.
    specified_amount {
      units = tostring(var.budget_amount)
    }
  }

  threshold_rules { threshold_percent = 0.5 }
  threshold_rules { threshold_percent = 0.9 }
  threshold_rules { threshold_percent = 1.0 }

  depends_on = [google_project_service.enabled]
}

data "google_project" "this" {}

# CI/CD identity used by the Cloud Build trigger: deploy rights only, no owner/editor.
resource "google_service_account" "build" {
  account_id   = "cloud-build-deployer"
  display_name = "Cloud Build deployer"
}

resource "google_project_iam_member" "build_roles" {
  for_each = toset([
    "roles/run.developer",
    "roles/artifactregistry.writer",
    "roles/logging.logWriter",
    "roles/storage.objectViewer", # read the uploaded build source in <project>_cloudbuild
  ])
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.build.email}"
}

# Lets the build deploy a Cloud Run service that runs as the API identity.
resource "google_service_account_iam_member" "build_acts_as_api" {
  service_account_id = google_service_account.api.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.build.email}"
}

# Identity the Vertex AI pipeline steps run as.
resource "google_service_account" "pipelines" {
  account_id   = "vertex-pipelines"
  display_name = "Vertex AI training pipeline"
}

resource "google_project_iam_member" "pipelines_roles" {
  for_each = toset([
    "roles/aiplatform.user",         # Model Registry uploads
    "roles/bigquery.jobUser",        # run the extract query
    "roles/artifactregistry.reader", # pull the trainer image
    "roles/run.developer",           # roll Cloud Run to the new model
    "roles/logging.logWriter",
  ])
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.pipelines.email}"
}

resource "google_bigquery_dataset_iam_member" "pipelines_reads_data" {
  dataset_id = google_bigquery_dataset.mlops.dataset_id
  role       = "roles/bigquery.dataViewer"
  member     = "serviceAccount:${google_service_account.pipelines.email}"
}

# Pipeline root, artifacts and promoted models.
resource "google_storage_bucket_iam_member" "pipelines_writes_bucket" {
  bucket = google_storage_bucket.mlops.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.pipelines.email}"
}

# Updating the Cloud Run service keeps it running as the API identity.
resource "google_service_account_iam_member" "pipelines_acts_as_api" {
  service_account_id = google_service_account.api.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.pipelines.email}"
}


# ---------------------------------------------------------------------------
# Stage 5: drift monitoring (G8)
# API prediction logs -> BigQuery (30-day retention) -> nightly PSI check in the
# pipeline -> WARNING log line -> email alert.

resource "google_bigquery_dataset" "logs" {
  dataset_id                  = "mlops_logs"
  location                    = "US" # same location as mlops, so drift queries can join
  default_table_expiration_ms = 30 * 24 * 60 * 60 * 1000
  delete_contents_on_destroy  = true
  depends_on                  = [google_project_service.enabled]
}

resource "google_logging_project_sink" "predictions" {
  name                   = "income-api-predictions"
  destination            = "bigquery.googleapis.com/projects/${var.project_id}/datasets/${google_bigquery_dataset.logs.dataset_id}"
  filter                 = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"income-api\" AND jsonPayload.event=\"prediction\""
  unique_writer_identity = true

  bigquery_options {
    use_partitioned_tables = true
  }
}

resource "google_bigquery_dataset_iam_member" "sink_writes_logs" {
  dataset_id = google_bigquery_dataset.logs.dataset_id
  role       = "roles/bigquery.dataEditor"
  member     = google_logging_project_sink.predictions.writer_identity
}

resource "google_bigquery_dataset_iam_member" "pipelines_reads_logs" {
  dataset_id = google_bigquery_dataset.logs.dataset_id
  role       = "roles/bigquery.dataViewer"
  member     = "serviceAccount:${google_service_account.pipelines.email}"
}

resource "google_monitoring_notification_channel" "email" {
  display_name = "MLOps alerts"
  type         = "email"
  labels = {
    email_address = var.alert_email
  }
  depends_on = [google_project_service.enabled]
}

resource "google_monitoring_alert_policy" "drift" {
  display_name = "Income model: input drift detected"
  combiner     = "OR"

  conditions {
    display_name = "Drift check logged a WARNING"
    condition_matched_log {
      filter = "logName=\"projects/${var.project_id}/logs/income-drift\" AND severity>=WARNING"
      label_extractors = {
        drifted = "EXTRACT(jsonPayload.drifted)"
      }
    }
  }

  alert_strategy {
    notification_rate_limit {
      period = "3600s"
    }
    auto_close = "86400s"
  }

  documentation {
    mime_type = "text/markdown"
    content   = <<-EOT
      Recent prediction requests differ from the training data (PSI > threshold).
      Check the drifted features in Logs Explorer (log `income-drift`), then decide whether
      the input pipeline changed (fix it) or the population changed (collect labels, retrain).
    EOT
  }

  notification_channels = [google_monitoring_notification_channel.email.id]
}
