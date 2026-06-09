# ML Layer — PRC-2025 fuel-burn baseline

**What this proves:** a research-grade prediction task wrapped in production scaffolding —
reproducible feature extraction, config-driven training with experiment tracking and best-model
logging, and a serving endpoint with latency logging.

## Metrics (measured)

| Item | Result |
|---|---|
| Baseline val RMSE | **369 kg** *(on mock PRC-shaped data — sanity only, not the published benchmark)* |
| Training | reproducible: 152 mock intervals, 5 epochs, MLflow-tracked (best, not final, model logged) |
| Serving smoke | **2/2** (health + fake-mode prediction bounds) |

> The model targets the **Eurocontrol PRC-2025 fuel-burn challenge** (published RMSE leaderboard).
> The real ~3.1 GB dataset is present in `data/raw/prc_2025/` (gitignored); the pipeline is not yet
> pointed at it, so the published-baseline comparison is **pending** (ADR-0005).

## Run

```bash
make ml-baseline-small   # extract features (mock) + train; logs to MLflow
make ml-serve-smoke      # FastAPI serving smoke test
```

## Key files

- `mock_data.py` — generator for PRC-shaped parquet/zip fixtures (offline fallback).
- `extract_features.py` — trajectory features from zipped parquets → flat parquet.
- `model.py` — baseline PyTorch MLP. `train.py` — training + MLflow (best-model logging).
- `serve.py` — FastAPI endpoint (Pydantic schemas, latency middleware, fake-mode toggle).

## Notes

Serving still uses a deprecated FastAPI startup hook and a hardcoded fake-model mode rather than
MLflow model loading — both tracked for W3.6 cleanup. Trainer INFO logs are currently suppressed
when MLflow configures the root logger first (metrics still land in MLflow).
