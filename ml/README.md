# ML Layer — PRC-2025 Fuel-Burn Estimation

**What this layer does:** end-to-end supervised learning for per-interval fuel-burn
estimation on the Eurocontrol PRC Data Challenge 2025 — from raw trajectory extraction
through experiment tracking, drift monitoring, alias-based model registry, and a
production FastAPI serving endpoint.

---

## Honest baseline (the only hard ML result)

> **CV RMSE 442.65 ± 36.36 kg** — chronological grouped cross-validation on real
> PRC-2025 data (`ml/cv.py`, `FuelBurn_Baseline` / `ChronologicalCV` run).

This number replaced an earlier ~395 kg figure that was produced by a
**train/val leakage bug**: the original pipeline used an interval-level
`random_split`, so trajectory intervals from the same flight appeared in both
train and validation sets. The fix replaced that split with a flight-level
`GroupShuffleSplit` and then with the current chronological expanding-window
CV (`ml/cv.py`), which groups all intervals of a flight into one side of every
fold. The post-fix honest RMSE is higher because the split no longer leaks.

The ~395 kg figure is **stale and must not be cited as a current result**.
Stale lineage is identified only through explicit tags such as
`data.leakage_fix=pre`; a low RMSE is never treated as proof of leakage.

---

## Metrics summary

| Metric | Value | Source |
|---|---|---|
| CV RMSE (mean ± std) | **442.65 ± 36.36 kg** | `ml/cv.py` on real PRC-2025 data |
| Single-split val RMSE | 444.78 kg | `ml/train.py` run (500-flight bounded set) |
| Serving latency p99 | **< 50 ms** (observed ~1.1 ms single-request CPU) | `ml/serve.py` smoke |
| Serving smoke | 2/2 (health + fake-mode prediction bounds) | `make ml-serve-smoke` |
| Full-data MLP CV RMSE | **355.04 ± 89.12 kg** (4 folds, 11,037 flights) | `outputs/ablation-full/challenger_results.json` |
| Full-data HistGBR CV RMSE | **142.20 ± 22.68 kg** (4 folds, 11,037 flights) | `outputs/ablation-full/challenger_results.json` |
| Full-scale RMSE (11,037 flights) | **pending owner Databricks run** | `ml/train_fullscale.py` |
| Rank-phase RMSE — HistGBR (held-out, 24,289 intervals / 1,888 flights) | **248.62 kg** | `ml/score_rank.py`; source: `data/raw/prc_2025/rank_score_result.json` |
| Rank-phase RMSE — MLP (held-out, same set) | **411.79 kg** | `ml/score_rank.py` |

---

## Feature schema (`ml/features.py`)

Single source of truth for the 38-feature input schema shared by all consumers
(extraction, training, CV, ablation, serving, drift monitoring).

| Group | Features | Count |
|---|---|---|
| Trajectory numeric | `duration_s`, `alt_change`, `avg_speed`, `max_vrate`, `avg_altitude`, `max_altitude`, `alt_std`, `avg_track_change`, `avg_mach`, `avg_tas`, `avg_cas` | 11 |
| Aircraft-type one-hot | `ac_A20N` … `ac___unknown__` (26 PRC-2025 types + 1 unknown bucket, alphabetical) | 27 |
| **Total** | `FEATURE_COLUMNS` / `INPUT_DIM` | **38** |

NaN-fill policy: `avg_mach`, `avg_tas`, and `avg_cas` fill to `0.0` only when
unavailable. In the completed extraction they are nonzero in 84,325, 44,511,
and 21,296 intervals respectively. All numerics fill to `0.0` for isolated
dropouts, so the MLP is NaN-safe. `aircraft_type` is one-hot encoded.

---

## CV harness (`ml/cv.py`)

Chronological expanding-window cross-validation, grouped by `flight_id`:

- Folds split by `flight_date` from `flightlist_train.parquet`.
- Each fold: train = all dates up to fold k, val = date k+1.
- A runtime assertion verifies zero `flight_id` overlap between train and val in
  every fold (leakage guard).
- Reports per-fold RMSE, aggregate mean ± std, and per-aircraft-type / per-duration-
  bucket slice RMSE to MLflow (`FuelBurn_Baseline` / `ChronologicalCV` run).
- Authoritative result: **CV RMSE 442.65 ± 36.36 kg**.

```bash
python -m ml.cv \
    --data-dir data/raw/prc_2025 \
    --epochs 10 \
    --n-folds 5      # optional: cap to last N folds (largest training sets)
```

---

## Ablation + challenger (`ml/ablation.py`)

Six feature-group ablation experiments and a `HistGradientBoostingRegressor`
challenger, both scored through the same chronological CV folds.

**Feature groups:**

| ID | Group | Columns |
|---|---|---|
| G1 | duration | `duration_s` |
| G2 | altitude | `alt_change`, `avg_altitude`, `max_altitude`, `alt_std` |
| G3 | speed / vertical rate | `avg_speed`, `max_vrate` |
| G4 | track / turning | `avg_track_change` |
| G5 | Mach / TAS / CAS | `avg_mach`, `avg_tas`, `avg_cas` (0-filled only when unavailable) |
| G6 | aircraft type | all 27 `ac_*` one-hot columns |

MLP ablation zeros out the group's columns (input dimension stays at 38).
HistGBR ablation drops the columns entirely.

The authoritative local run used 131,530 intervals from all 11,037 training
flights and the last four expanding-window chronological folds. HistGBR scored
**142.20 ± 22.68 kg**, versus **355.04 ± 89.12 kg** for the MLP. Removing
duration hurt HistGBR most (**+382.33 kg**), followed by aircraft type
(**+87.85 kg**) and altitude (**+26.64 kg**).

```bash
# Offline illustration (no real data needed):
python -m ml.ablation --mock --epochs 5 --output-dir outputs/ablation

# Authoritative run (full-data challenger + HistGBR feature ablation):
python -m ml.ablation \
  --data-dir data/raw/prc_2025 \
  --epochs 10 \
  --n-folds 4 \
  --model histgbr \
  --output-dir outputs/ablation-full
```

---

## Full-scale path — owner-cloud (`ml/train_fullscale.py`)

Databricks-ready wrapper around `ml/train.py` + `ml/extract_features.py` for
full 11,037-flight training. Config at `ml/configs/fullscale.yaml`; Databricks
job spec at `ml/configs/databricks_job.json`.

Key settings (`ml/configs/fullscale.yaml` defaults):

| Key | Default | Notes |
|---|---|---|
| `training.epochs` | 50 | Override with `--override training.epochs=N` |
| `training.batch_size` | 16 | |
| `training.lr` | 0.001 | Adam |
| `training.seed` | 42 | |
| `mlflow.experiment_name` | `FuelBurn_Baseline` | |
| `mlflow.tracking_uri` | `sqlite:///mlflow.db` | Set `databricks` for Databricks |
| `data.extract_limit` | null (all flights) | Set to integer for bounded runs |

Feature extraction is **resumable**: checkpoint shards under
`features_checkpoint_<split>/` let interrupted jobs continue from where they
stopped; existing `features_train.parquet` skips extraction entirely.

**Infrastructure status:** offline smoke passes (`python -m ml.train_fullscale --smoke`).
Full-scale RMSE is **pending the owner's Databricks run**.
See runbook: [`docs/runbooks/ml-fullscale-databricks.md`](../docs/runbooks/ml-fullscale-databricks.md)

```bash
# Offline smoke (no real data, no Databricks):
python -m ml.train_fullscale --smoke

# Full scale on Databricks (owner):
python -m ml.train_fullscale \
    --config ml/configs/fullscale.yaml \
    --override data.data_dir=/Volumes/<catalog>/<schema>/<vol>/prc_2025 \
    --override mlflow.tracking_uri=databricks \
    --override training.epochs=50
```

---

## Post-run validation + registry handoff (`ml/validate_fullscale_run.py`)

After the Databricks run completes, this script validates the MLflow run against
the expected-output contract and then calls `ml.registry.compare_and_promote()`:

| Check | Threshold | Hard FAIL? |
|---|---|---|
| `best_val_rmse` metric present | must exist | Yes |
| `best_val_rmse` below regression threshold | < 500 kg | Yes |
| `best_val_rmse` at or below CV re-baseline | <= 442.65 kg | WARN (or FAIL with `--strict`) |
| `model/` artifact present | must exist | Yes |
| `train_rmse` per-epoch history present | must exist | Yes |
| `train_rmse` improves first-to-last | last <= first | Yes |

On PASS the script promotes the run to the registry. On FAIL the registry is
not touched. Invoked as part of the Databricks runbook (Step 11).

---

## Rank-phase scoring (`ml/score_rank.py`)

Loads the production model from `models:/FuelBurn@production`, predicts on the
PRC-2025 rank phase (24,289 intervals, 1,888 flights from `fuel_rank.parquet`),
computes per-interval RMSE-kg against TRUE labels, and reports the held-out score.

The JOAS paper's 201 kg winning score is for the separate final phase and is not
a same-split rank baseline. Use `--baseline-rmse` only for a reference evaluated
against the exact same rank labels.

See runbook: [`docs/runbooks/ml-headtohead.md`](../docs/runbooks/ml-headtohead.md)

```bash
python -m ml.score_rank \
    --data-dir data/raw/prc_2025 \
    --model-uri models:/FuelBurn@production
```

---

## Drift monitor (`ml/drift.py`)

Evidently `DataDriftPreset` over 12 columns: 11 numeric trajectory features +
one decoded categorical column `aircraft_type_cat` (collapsed from the 27 one-hot
columns to avoid testing 27 near-zero binary columns individually).

- Numeric: Kolmogorov-Smirnov test (p-value threshold 0.05).
- Categorical: chi-square test (p-value threshold 0.05).
- Retrain threshold: 20% of monitored columns drifted (`DEFAULT_RETRAIN_THRESHOLD = 0.2`).
- `trigger_retrain()` accepts an injectable hook (used in tests); falls back to
  `ml.cv.run_cv` for offline demo.

**Demonstrated offline:** stable batch → no retrain; heavily drifted batch (shifted
duration, speed, altitude, aircraft-type distribution) → retrain trigger fires.
Both scenarios are asserted at the end of `python -m ml.drift`.

```bash
python -m ml.drift    # offline demo with synthetic batches
```

---

## Model registry (`ml/registry.py`)

MLflow 3.x **alias-based** registry under registered model name **`FuelBurn`**.

| Alias | Meaning |
|---|---|
| `production` | primary production alias; resolves via `models:/FuelBurn@production` |
| `champion` | same version as `production`; set together |
| `challenger` | ephemeral alias during evaluation; removed after promotion decision |

Champion/challenger promotion compares `cv_mean_rmse_kg` (preferred) or
`best_val_rmse` (fallback). Lower is better. Pre-leakage stale runs are rejected
only when explicit provenance tags identify them; metric values are not lineage.

**Production URI (canonical):** `models:/FuelBurn@production`

Do not use the deprecated stage form `models:/FuelBurn/Production` — MLflow 3.x
stages carry a `FutureWarning` and the stage form will not resolve against an
alias-only registry.

```bash
# Promote a run (champion/challenger comparison):
python -m ml.registry promote \
    --run-id <run-id> \
    --tracking-uri sqlite:///mlflow.db

# Tag pre-leakage stale runs (dry-run first):
python -m ml.registry tag-stale --dry-run --tracking-uri sqlite:///mlflow.db
python -m ml.registry tag-stale --tracking-uri sqlite:///mlflow.db
```

---

## Serving endpoint (`ml/serve.py`)

FastAPI endpoint on the 38-feature schema. Clients send 11 numeric fields + an
`aircraft_type` string; the server assembles the full 38-dim tensor server-side
using `encode_aircraft_type()` from `ml/features.py`.

- **Latency SLO:** p99 < 50 ms (observed ~1.1 ms single-request CPU).
- **C2 logging:** every prediction is written to the `ml_predictions` Postgres
  table (availability-gated: silently skipped if DB unreachable).
- **Prometheus metric:** `ml_predict_latency_seconds`.
- **Model loading priority:**
  1. `ML_MODEL_PATH` → `torch.load` (offline-safe, no tracking server)
  2. `ML_MODEL_URI` → `mlflow.pytorch.load_model` (set to `models:/FuelBurn@production`)
  3. Neither set → falls back to fake-heuristic mode

Environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `ML_FAKE_MODE` | `true` | Set to `false` to load a real model |
| `ML_MODEL_PATH` | — | Local `.pth` checkpoint path (takes priority) |
| `ML_MODEL_URI` | — | MLflow model URI; set to `models:/FuelBurn@production` |
| `ML_RUN_ID` | `serve` | Written to `ml_predictions.run_id` |

---

## Make targets

```bash
make ml-baseline-real    # extract features (bounded: LIMIT=500) + train; EPOCHS=10
make ml-baseline-small   # offline mock path (no real data needed)
make ml-serve-smoke      # start FastAPI serving + run smoke test (2/2 expected)
```

---

## Key files

| File | Purpose |
|---|---|
| `ml/features.py` | Single source of truth: `FEATURE_COLUMNS`, `INPUT_DIM=38`, `encode_aircraft_type()` |
| `ml/model.py` | `FuelBurnMLP` PyTorch model |
| `ml/train.py` | Training loop + MLflow tracking (best-model logging) |
| `ml/cv.py` | Chronological CV harness; produced CV RMSE 442.65 ± 36.36 kg |
| `ml/ablation.py` | 6-group feature ablation + HistGBR challenger |
| `ml/train_fullscale.py` | Databricks-ready full-scale wrapper (11,037 flights) |
| `ml/configs/fullscale.yaml` | Full-scale training config |
| `ml/configs/databricks_job.json` | Git-backed Databricks serverless job spec |
| `ml/validate_fullscale_run.py` | Post-run contract validation + registry handoff |
| `ml/score_rank.py` | Standalone rank-phase scoring |
| `ml/drift.py` | Evidently drift monitor + retrain trigger |
| `ml/registry.py` | MLflow 3.x alias-based promotion (`models:/FuelBurn@production`) |
| `ml/serve.py` | FastAPI serving endpoint (38-feature schema, C2 logging) |
| `ml/dataset.py` | `FuelBurnDataset` (`torch.utils.data.Dataset`) |
| `ml/extract_features.py` | Resumable per-flight feature extraction with checkpointing |
| `ml/mock_data.py` | PRC-shaped mock parquet/zip fixture for offline tests |

---

## Reproducibility

| Item | Value |
|---|---|
| Random seed | 42 (Python, NumPy, PyTorch — set in `cv.py`, `train.py`, `train_fullscale.py`) |
| MLflow experiment | `FuelBurn_Baseline` |
| MLflow tracking (local) | `sqlite:///mlflow.db` |
| Framework | PyTorch (MLP), scikit-learn (HistGBR challenger), Evidently (drift) |
| PRC-2025 dataset | `data/raw/prc_2025/` (gitignored; ~3.1 GB) |

---

## Rank-phase results (close-out, 2026-06-11)

The held-out rank score has been measured. Train on 131,530 training intervals;
predict on **24,289 rank intervals / 1,888 flights** (`fuel_rank.parquet`, TRUE
labels):

| Model | Rank-phase RMSE (kg) |
|---|---|
| **HistGBR** | **248.62** |
| MLP | 411.79 |

HistGBR wins by approximately 163 kg. The held-out rank RMSE (248.62 kg) is
higher than the in-sample CV RMSE (142.20 kg), which is expected — the rank
phase is a true held-out generalization test on separate flights. Source
artifact: `data/raw/prc_2025/rank_score_result.json`.

The JOAS-2026 published baseline for the rank phase is not available in this
repository. The 248.62 kg figure is reported standalone; a like-for-like
comparison awaits the owner supplying that figure.

## Pending (owner-cloud)

The following result still requires the owner to run the full-scale pipeline on
Databricks:

1. **Full-scale RMSE** (`ml/train_fullscale.py` on 11,037 flights, Databricks)

The standalone rank-phase RMSE has been measured (see above). The JOAS-2026
rank-phase baseline comparison remains open pending that published figure.
