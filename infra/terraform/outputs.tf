output "bucket" {
  value = google_storage_bucket.mlops.url
}

output "image_repo" {
  value = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.mlops.repository_id}"
}

output "api_service_account" {
  value = google_service_account.api.email
}

output "build_service_account" {
  value = google_service_account.build.email
}

output "pipelines_service_account" {
  value = google_service_account.pipelines.email
}
