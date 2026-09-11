# ===========================================================
# Flight Telemetry Intelligence Platform — Makefile
# ===========================================================
# One-command entrypoints per layer.
# Run `make help` to see available targets.

.DEFAULT_GOAL := help
SHELL := /bin/bash

# Use project venv if available, else system python3
PYTHON := $(shell if [ -x .venv/bin/python3 ]; then echo .venv/bin/python3; else echo python3; fi)

# ---- Tunables (override on CLI, e.g. `make ml-baseline-real LIMIT=1000 EPOCHS=15`) ----
LIMIT   ?= 500
EPOCHS  ?= 10

# ---- MLflow / registry tunables (override as needed) ------
# RUN_ID: MLflow run_id for validate-fullscale-run and ml-promote
RUN_ID            ?= ""
# MLFLOW_TRACKING_URI: passed as --tracking-uri; falls back to env var when blank
MLFLOW_TRACKING_URI ?= ""
# DATA_DIR: real PRC-2025 data root for score/ablation targets
DATA_DIR          ?= data/raw/prc_2025
# BENCH_ROWS: synthetic row count for bench-index-1m (default 1 M)
BENCH_ROWS        ?= 1000000
# BENCH_QUERIES: query workload size for bench-index-1m
BENCH_QUERIES     ?= 500

# ---- Directories (generated outputs — gitignored) ---------
DATA_RAW     := data/raw
DATA_INTERIM := data/interim
DATA_PROCESSED := data/processed
OUTPUTS      := outputs
ARTIFACTS    := artifacts

# ---- Help -------------------------------------------------
.PHONY: help
help: ## Show available targets
	@echo ""
	@echo "Flight Telemetry Intelligence Platform"
	@echo "======================================"
	@echo ""
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-30s\033[0m %s\n", $$1, $$2}'
	@echo ""

# ---- Setup ------------------------------------------------
.PHONY: setup
setup: ## Create local output directories (gitignored)
	@mkdir -p $(DATA_RAW) $(DATA_INTERIM) $(DATA_PROCESSED) $(OUTPUTS) $(ARTIFACTS)
	@echo "✓ Local output directories created."

.PHONY: check-env
check-env: ## Verify .env exists (does not validate contents)
	@if [ ! -f .env ]; then \
		echo "⚠ .env not found. Copy .env.example → .env and fill in values."; \
		exit 1; \
	fi
	@echo "✓ .env file found."

# ---- Contracts / Schemas ----------------------------------
.PHONY: test-contracts
test-contracts: ## Validate shared schemas and fixtures
	$(PYTHON) shared/contracts/validate_schemas.py --verbose

# ---- Data -------------------------------------------------
.PHONY: data-sample
data-sample: setup ## Download / generate a tiny development data subset
	$(PYTHON) data/scripts/bootstrap.py --mode sample

.PHONY: data-sample-verify
data-sample-verify: ## Verify sample data against manifest checksums
	$(PYTHON) data/scripts/bootstrap.py --verify

.PHONY: data-local-silver
data-local-silver: systems-replay-sample ## Transform local bronze landing to silver
	$(PYTHON) -m data.transforms.cli --input data/interim/landing.jsonl

.PHONY: data-local-gold
data-local-gold: data-local-silver ## Transform local silver to gold aggregates
	$(PYTHON) -m data.transforms.cli_gold --input data/processed/silver_flight_state.jsonl

.PHONY: data-export-gold-parquet
data-export-gold-parquet: data-local-gold ## Export typed local gold Parquet for BigQuery bq load
	$(PYTHON) data/scripts/export_gold_parquet.py

.PHONY: data-refresh-experiment-small
data-refresh-experiment-small: data-local-silver ## Run incremental-vs-recompute experiment (small)
	$(PYTHON) data/experiments/refresh_strategy.py --sizes 10000,100000,1000000 --inc-pct 0.1

.PHONY: test-data
test-data: ## Run data-layer tests
	$(PYTHON) -m pytest data/tests/ -v 2>/dev/null || $(PYTHON) -m unittest discover -s data/tests -v

.PHONY: test-cloud
test-cloud: ## Run cloud data-layer tests (242 tests, offline)
	$(PYTHON) -m pytest data/cloud/ -v

.PHONY: airflow-test
airflow-test: ## Run the medallion dag structure tests inside the astro container
	cd airflow && astro dev pytest

# ---- Systems ----------------------------------------------
.PHONY: systems-replay-sample
systems-replay-sample: data-sample ## Replay historical sample through ingestion
	$(PYTHON) -m systems.replay.cli --input data/raw/sample_state_vectors.jsonl

.PHONY: systems-benchmark-small
systems-benchmark-small: data-local-silver ## Run spatiotemporal index benchmark (small)
	$(PYTHON) -m systems.index.cli --input data/processed/silver_flight_state.jsonl --queries 200 --profile regional

.PHONY: bench-index-1m
bench-index-1m: ## Run 1M-row synthetic spatiotemporal index benchmark (BENCH_ROWS=1000000, BENCH_QUERIES=500)
	$(PYTHON) -m systems.index.cli --mode offline --synthetic-rows $(BENCH_ROWS) --queries $(BENCH_QUERIES) --profile regional

.PHONY: test-systems
test-systems: ## Run systems-layer tests
	$(PYTHON) -m pytest systems/tests/ -v 2>/dev/null || $(PYTHON) -m unittest discover -s systems/tests -v

# ---- ML ---------------------------------------------------
.PHONY: ml-extract-features
ml-extract-features: ## Precompute trajectory features from zipped parquets
	$(PYTHON) -m ml.extract_features --data-dir data/ml/prc_2025_mock --split train

.PHONY: ml-baseline-small
ml-baseline-small: ml-extract-features ## Reproduce ML baseline on mock data
	$(PYTHON) -m ml.train --data-dir data/ml/prc_2025_mock --epochs 5

.PHONY: ml-baseline-real
ml-baseline-real: ## Train the baseline on the REAL PRC 2025 data (bounded: LIMIT flights, EPOCHS epochs)
	$(PYTHON) -m ml.extract_features --data-dir data/raw/prc_2025 --split train --limit $(LIMIT)
	$(PYTHON) -m ml.train --data-dir data/raw/prc_2025 --epochs $(EPOCHS)

.PHONY: ml-cv
ml-cv: ## Chronological CV on the bounded real dataset (LIMIT flights, EPOCHS epochs)
	$(PYTHON) -m ml.extract_features --data-dir data/raw/prc_2025 --split train --limit $(LIMIT)
	$(PYTHON) -m ml.cv --data-dir data/raw/prc_2025 --epochs $(EPOCHS)

.PHONY: ml-serve-smoke
ml-serve-smoke: ## Start serving endpoint and run smoke test
	$(PYTHON) -m pytest ml/test_serve.py -v -s

.PHONY: test-ml
test-ml: ## Run full ML-layer test suite (226 tests, offline)
	$(PYTHON) -m pytest ml/ -v

.PHONY: ml-ablate
ml-ablate: ## Feature-group ablation + HistGBR challenger (offline mock; set DATA_DIR for real data)
	$(PYTHON) -m ml.ablation \
		$(if $(filter-out data/raw/prc_2025,$(DATA_DIR)),--data-dir $(DATA_DIR),--mock) \
		--epochs $(EPOCHS)

.PHONY: ml-drift
ml-drift: ## Run the feature-drift monitor offline demo (no args required)
	$(PYTHON) -m ml.drift

.PHONY: ml-score-rank
ml-score-rank: ## Score the rank-phase split against TRUE labels (DATA_DIR required)
	$(PYTHON) -m ml.score_rank --data-dir $(DATA_DIR)

.PHONY: ml-train-fullscale
ml-train-fullscale: ## Full-scale PRC-2025 training run (Databricks-ready; use --override via EXTRA_ARGS)
	$(PYTHON) -m ml.train_fullscale \
		$(if $(MLFLOW_TRACKING_URI),--override mlflow.tracking_uri=$(MLFLOW_TRACKING_URI),) \
		$(if $(EXTRA_ARGS),$(EXTRA_ARGS),)

.PHONY: validate-fullscale-run
validate-fullscale-run: ## Validate a full-scale MLflow run against the ml-05 contract (RUN_ID required)
	$(PYTHON) -m ml.validate_fullscale_run \
		--run-id $(RUN_ID) \
		$(if $(MLFLOW_TRACKING_URI),--tracking-uri $(MLFLOW_TRACKING_URI),)

.PHONY: ml-promote
ml-promote: ## Register and promote a challenger run to production (RUN_ID required)
	$(PYTHON) -m ml.registry promote \
		--run-id $(RUN_ID) \
		$(if $(MLFLOW_TRACKING_URI),--tracking-uri $(MLFLOW_TRACKING_URI),)

.PHONY: ml-tag-stale
ml-tag-stale: ## Tag pre-leakage stale runs in the FuelBurn_Baseline experiment
	$(PYTHON) -m ml.registry tag-stale \
		$(if $(MLFLOW_TRACKING_URI),--tracking-uri $(MLFLOW_TRACKING_URI),)

# ---- AI ---------------------------------------------------
.PHONY: ai-eval-fixtures
ai-eval-fixtures: ## Validate AI golden question set and fixtures (offline, no LLM)
	$(PYTHON) -m ai.eval.run_fixtures

.PHONY: test-ai
test-ai: ## Run AI-layer tests
	$(PYTHON) -m pytest ai/tests/ -v 2>/dev/null || $(PYTHON) -m unittest discover -s ai/tests -v

.PHONY: ai-eval-small
ai-eval-small: ## Run AI answer-path eval on the golden set (deterministic, offline)
	$(PYTHON) -m ai.eval.run_answers

.PHONY: ai-compare-small
ai-compare-small: ## Compare AI answer strategies on golden set (deterministic + Ollama if available)
	$(PYTHON) -m ai.eval.compare

# ---- Dashboard --------------------------------------------
.PHONY: test-dashboard
test-dashboard: ## Run dashboard test suite (80 tests, offline)
	$(PYTHON) -m pytest dashboard/ -v

# ---- Cross-cutting ----------------------------------------
.PHONY: smoke
smoke: ## Run cross-layer smoke test (all layers, bounded data, offline)
	$(PYTHON) scripts/smoke.py

.PHONY: clean
clean: ## Remove generated local outputs (not source)
	rm -rf $(DATA_RAW) $(DATA_INTERIM) $(DATA_PROCESSED) $(OUTPUTS) $(ARTIFACTS)
	@echo "✓ Generated output directories removed."
