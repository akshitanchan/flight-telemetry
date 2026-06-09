# ML Layer — PRC-2025 fuel-burn baseline

**What this proves:** a research-grade prediction task wrapped in production scaffolding —
reproducible feature extraction, config-driven training with experiment tracking and best-model
logging, and a serving endpoint with latency logging.

## Metrics (measured)

| Item | Result |
|---|---|
| Baseline val RMSE (real data) | **~395 kg** *(real PRC-2025 fuel labels; bounded 500-flight subset, 5,508 intervals)* |
| Training | reproducible + seeded; MLflow-tracked (best, not final, model logged) |
| Serving smoke | **2/2** (health + fake-mode prediction bounds) |

> The model targets the **Eurocontrol PRC-2025 fuel-burn challenge** (published RMSE leaderboard).
> It now trains on the **real dataset** in `data/raw/prc_2025/` (gitignored) via `make ml-baseline-real`
> (bounded by `LIMIT`/`EPOCHS`). Full-scale training over all 11,037 flights and the official leaderboard
> scoring remain **pending** (ADR-0005); `make ml-baseline-small` still runs the offline mock path.

## Run

```bash
make ml-baseline-real    # extract features from real PRC data (bounded) + train; LIMIT=500 EPOCHS=10
make ml-baseline-small   # offline mock path (no real data needed)
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
