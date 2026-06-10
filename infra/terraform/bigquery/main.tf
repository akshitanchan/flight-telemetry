terraform {
  required_version = ">= 1.6.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

# ---------------------------------------------------------------------------
# Provider
# Auth is handled by GOOGLE_APPLICATION_CREDENTIALS in the owner's .env.
# Never hard-code credentials here.
# ---------------------------------------------------------------------------
provider "google" {
  project = var.project
}

# ---------------------------------------------------------------------------
# BigQuery dataset
# ---------------------------------------------------------------------------
resource "google_bigquery_dataset" "flight_telemetry" {
  dataset_id                 = var.dataset
  friendly_name              = "Flight Telemetry"
  description                = "Managed by Terraform. Target dataset for dbt models in the flight-telemetry project."
  location                   = var.location
  delete_contents_on_destroy = false

  labels = {
    managed_by  = "terraform"
    workstream  = "data-cloud"
    environment = "production"
  }
}

# ---------------------------------------------------------------------------
# dbt service account (created only when dbt_sa_email is not provided)
# ---------------------------------------------------------------------------
locals {
  create_sa    = var.dbt_sa_email == ""
  dbt_sa_email = local.create_sa ? google_service_account.dbt_runner[0].email : var.dbt_sa_email
}

resource "google_service_account" "dbt_runner" {
  count = local.create_sa ? 1 : 0

  account_id   = "dbt-runner"
  display_name = "dbt Runner"
  description  = "Service account used by dbt to read/write the flight_telemetry BigQuery dataset."
  project      = var.project
}

# ---------------------------------------------------------------------------
# IAM — BigQuery Data Editor on the dataset
# Grants: bigquery.tables.*, bigquery.datasets.get
# ---------------------------------------------------------------------------
resource "google_bigquery_dataset_iam_member" "dbt_data_editor" {
  project    = var.project
  dataset_id = google_bigquery_dataset.flight_telemetry.dataset_id
  role       = "roles/bigquery.dataEditor"
  member     = "serviceAccount:${local.dbt_sa_email}"
}

# ---------------------------------------------------------------------------
# IAM — BigQuery Job User on the project
# Grants: bigquery.jobs.create (required to execute queries/models)
# ---------------------------------------------------------------------------
resource "google_project_iam_member" "dbt_job_user" {
  project = var.project
  role    = "roles/bigquery.jobUser"
  member  = "serviceAccount:${local.dbt_sa_email}"
}
