variable "project" {
  description = "GCP project ID that hosts the BigQuery dataset."
  type        = string
}

variable "dataset" {
  description = "BigQuery dataset ID that dbt will target."
  type        = string
  default     = "flight_telemetry"
}

variable "location" {
  description = "BigQuery dataset location (multi-region or region)."
  type        = string
  default     = "US"
}

variable "dbt_sa_email" {
  description = <<-EOT
    Email of the existing GCP service account used by dbt.
    Leave empty to create a new service account named 'dbt-runner'.
    Example: dbt-runner@my-project.iam.gserviceaccount.com
  EOT
  type        = string
  default     = ""
}
