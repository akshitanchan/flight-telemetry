output "dataset_id" {
  description = "Fully-qualified BigQuery dataset ID (project:dataset)."
  value       = "${var.project}:${google_bigquery_dataset.flight_telemetry.dataset_id}"
}

output "dataset_self_link" {
  description = "Self-link URI of the BigQuery dataset."
  value       = google_bigquery_dataset.flight_telemetry.self_link
}

output "dbt_sa_email" {
  description = "Email of the service account dbt uses to authenticate against BigQuery."
  value       = local.dbt_sa_email
}
