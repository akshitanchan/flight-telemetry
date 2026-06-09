# ===========================================================
# Flight Telemetry Intelligence Platform — Makefile
# ===========================================================
# One-command entrypoints per layer.
# Run `make help` to see available targets.

.DEFAULT_GOAL := help
SHELL := /bin/bash

# Use project venv if available, else system python3
PYTHON := $(shell if [ -x .venv/bin/python3 ]; then echo .venv/bin/python3; else echo python3; fi)

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
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
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

.PHONY: data-refresh-experiment-small
data-refresh-experiment-small: data-local-silver ## Run incremental-vs-recompute experiment (small)
	$(PYTHON) data/experiments/refresh_strategy.py --sizes 10000,100000,1000000 --inc-pct 0.1

.PHONY: test-data
test-data: ## Run data-layer tests
	$(PYTHON) -m pytest data/tests/ -v 2>/dev/null || $(PYTHON) -m unittest discover -s data/tests -v

# ---- Systems ----------------------------------------------
.PHONY: systems-replay-sample
systems-replay-sample: data-sample ## Replay historical sample through ingestion
	$(PYTHON) -m systems.replay.cli --input data/raw/sample_state_vectors.jsonl

.PHONY: systems-benchmark-small
systems-benchmark-small: data-local-silver ## Run spatiotemporal index benchmark (small)
	$(PYTHON) -m systems.index.cli --input data/processed/silver_flight_state.jsonl --queries 200 --profile regional

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

.PHONY: ml-serve-smoke
ml-serve-smoke: ## Start serving endpoint and run smoke test
	$(PYTHON) -m pytest ml/test_serve.py -v -s

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
ai-compare-small: ## Compare AI answer strategies on golden set
	@echo "→ ai-compare-small: not yet implemented (target for W3.3)"

# ---- Cross-cutting ----------------------------------------
.PHONY: smoke
smoke: ## Run cross-layer smoke test (all layers, bounded data)
	@echo "→ smoke: not yet implemented (target for W3.4)"

.PHONY: docs-check
docs-check: ## Lint / check documentation
	@echo "→ docs-check: not yet implemented (target for W3.5)"

.PHONY: clean
clean: ## Remove generated local outputs (not source)
	rm -rf $(DATA_RAW) $(DATA_INTERIM) $(DATA_PROCESSED) $(OUTPUTS) $(ARTIFACTS)
	@echo "✓ Generated output directories removed."
