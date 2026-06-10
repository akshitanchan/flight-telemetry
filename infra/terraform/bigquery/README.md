# Terraform — BigQuery dataset for flight-telemetry

This module provisions **exactly two things** inside GCP:

| Resource | What it creates |
|---|---|
| `google_bigquery_dataset.flight_telemetry` | The dataset dbt models write into |
| `google_service_account.dbt_runner` | SA used by dbt (only when `dbt_sa_email` is left empty) |

Plus two IAM bindings (no extra resources):

| Binding | Role | Scope |
|---|---|---|
| `google_bigquery_dataset_iam_member.dbt_data_editor` | `roles/bigquery.dataEditor` | dataset |
| `google_project_iam_member.dbt_job_user` | `roles/bigquery.jobUser` | project |

> **Databricks is NOT provisioned here.** Databricks Free Edition is workspace-managed by the owner directly.

---

## Prerequisites

1. `terraform` >= 1.6 installed on the machine running these commands.
2. A GCP project with the BigQuery API enabled.
3. `GOOGLE_APPLICATION_CREDENTIALS` set in the environment pointing to a service-account key
   that has at minimum `roles/bigquery.admin` and `roles/iam.serviceAccountAdmin` on the
   project (or narrower permissions if the SA already exists — see **Reusing an existing SA**).

The key file path is already wired in the owner's `.env`. Load it before running Terraform:

```bash
export $(grep -v '^#' .env | xargs)
```

---

## Variables

| Name | Default | Required | Description |
|---|---|---|---|
| `project` | — | yes | GCP project ID |
| `dataset` | `flight_telemetry` | no | BigQuery dataset ID |
| `location` | `US` | no | Dataset location |
| `dbt_sa_email` | `""` | no | Existing SA email; leave blank to create a new one |

### Passing variables

**Option A — CLI flags (no file on disk)**

```bash
terraform plan \
  -var="project=my-gcp-project-id"
```

**Option B — a local `.tfvars` file (git-ignored)**

Create `infra/terraform/bigquery/local.tfvars` (this file is gitignored and must stay off git):

```hcl
project      = "my-gcp-project-id"
dataset      = "flight_telemetry"
location     = "US"
# dbt_sa_email = "dbt-runner@my-gcp-project-id.iam.gserviceaccount.com"
```

Then reference it with `-var-file`:

```bash
terraform plan  -var-file="local.tfvars"
terraform apply -var-file="local.tfvars"
```

---

## Reusing an existing service account

If the dbt SA already exists, pass its email so Terraform skips creating a new one and only
attaches the IAM bindings:

```bash
terraform plan \
  -var="project=my-gcp-project-id" \
  -var="dbt_sa_email=dbt-runner@my-gcp-project-id.iam.gserviceaccount.com"
```

---

## Workflow

> **`apply` is owner-run only.** It touches real cloud infrastructure and incurs GCP costs.
> Never run `apply` in CI or as part of an automated pipeline without explicit owner approval.

```bash
# 1. Initialise — downloads the hashicorp/google provider (~5.x)
terraform init

# 2. Validate HCL syntax and provider schema (no network calls after init)
terraform validate

# 3. Format check — must pass before committing
terraform fmt -check -recursive

# 4. Preview the plan (read-only, safe to run in CI for drift detection)
terraform plan -var="project=<YOUR_PROJECT_ID>"

# 5. Apply — OWNER RUN ONLY, touches real GCP resources
terraform apply -var="project=<YOUR_PROJECT_ID>"
```

Expected plan output (fresh environment, no existing SA):

```
Plan: 4 to add, 0 to change, 0 to destroy.
  + google_bigquery_dataset.flight_telemetry
  + google_service_account.dbt_runner[0]
  + google_bigquery_dataset_iam_member.dbt_data_editor
  + google_project_iam_member.dbt_job_user
```

Expected plan output (existing SA supplied via `dbt_sa_email`):

```
Plan: 3 to add, 0 to change, 0 to destroy.
  + google_bigquery_dataset.flight_telemetry
  + google_bigquery_dataset_iam_member.dbt_data_editor
  + google_project_iam_member.dbt_job_user
```

---

## State management

Terraform state is **not** stored in this repository (covered by `.gitignore`).

For a shared team workflow, configure a GCS remote backend by adding this block to `main.tf`
before running `init`:

```hcl
terraform {
  backend "gcs" {
    bucket = "<YOUR_TF_STATE_BUCKET>"
    prefix = "flight-telemetry/bigquery"
  }
}
```

State locking is automatic with the GCS backend.

---

## Security notes

- Auth relies solely on `GOOGLE_APPLICATION_CREDENTIALS`; no credentials appear in HCL.
- `*.tfvars`, `*.tfstate*`, `.terraform/`, and `*-credentials.json` are all gitignored at both
  the module level (`infra/terraform/.gitignore`) and the repo root (`.gitignore`).
- `delete_contents_on_destroy = false` prevents accidental table loss if the dataset resource
  is removed from state.
- The SA receives the minimum roles needed for dbt: `dataEditor` (dataset-scoped) +
  `jobUser` (project-scoped). It does NOT receive `bigquery.admin` or project `editor`.
