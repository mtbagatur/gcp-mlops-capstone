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
