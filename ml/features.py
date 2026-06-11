"""
ml/features.py — Single source of truth for the fuel-burn feature schema.
=========================================================================

Every consumer (dataset.py, cv.py, model.py, serve.py) must import
FEATURE_COLUMNS and INPUT_DIM from here.  Editing this file is the ONE
place to extend/change the feature set for the MLP path.

Feature list (ordered)
----------------------
Trajectory-derived (11 numeric):
  duration_s       — interval length in seconds  (from fuel table start/end)
  alt_change       — altitude[-1] - altitude[0] for the interval (ft)
  avg_speed        — mean groundspeed over interval (kts); fill 0 if all NaN
  max_vrate        — max |vertical_rate| over interval (ft/min); fill 0 if NaN
  avg_altitude     — mean barometric altitude over interval (ft); fill 0 if NaN
  max_altitude     — maximum altitude over interval (ft); fill 0 if NaN
  alt_std          — std dev of altitude over interval (ft); fill 0 if < 2 pts
  avg_track_change — mean absolute per-step change in track angle (deg/step);
                     captures turning intensity; fill 0 if < 2 pts
  avg_mach         — mean Mach number over interval; fill 0.0 when all NaN
  avg_tas          — mean True Air Speed over interval (kts); fill 0.0 (NaN)
  avg_cas          — mean Calibrated Air Speed over interval (kts); fill 0.0 (NaN)

Aircraft-type one-hot (27 columns):
  ac_{type}        — one column per entry in AIRCRAFT_TYPES (alphabetical order)
                     The last entry "__unknown__" fires for any type not in the
                     fixed list, ensuring robustness to new types at inference.

NaN-fill strategy
-----------------
The downstream model is an MLP that cannot accept NaN values.
  - mach / TAS / CAS  : filled with 0.0 — these are missing for the majority of
    ADS-B-only tracks; 0.0 is distinguishable from genuine zero (no aircraft
    cruises at Mach 0) and is far simpler than imputing from sparse data.
  - groundspeed / vertical_rate / track : filled with 0.0 — isolated NaN
    dropouts in otherwise dense tracks; median imputation would require a
    per-split statistic and risk leakage; 0.0 is a safe neutral value.
  - alt_std / avg_track_change : set to 0.0 when the interval has fewer than
    2 trajectory points (degenerate intervals).

Aircraft-type encoding
----------------------
One-hot encoding was chosen over ordinal encoding because:
  a) The MLP has no built-in notion of aircraft-type similarity (unlike a tree).
  b) It makes the contribution of each type transparent and auditable.
  c) Embedding layers would require a lookup table at inference time, adding
     coupling complexity; one-hot keeps the pipeline purely numeric.

The 26 types observed in PRC-2025 training data (flightlist_train.parquet) plus
a synthetic "__unknown__" bucket are encoded alphabetically, yielding 27 columns.
INPUT_DIM accounts for the full encoded width: 11 numeric + 27 one-hot = 38.

ml-10 import contract
---------------------
  from ml.features import FEATURE_COLUMNS, INPUT_DIM, AIRCRAFT_TYPES, encode_aircraft_type

  FEATURE_COLUMNS : list[str]   — ordered column names (length == INPUT_DIM)
  INPUT_DIM       : int         — width of the feature tensor (38)
  AIRCRAFT_TYPES  : list[str]   — the 27 types used for one-hot encoding
  encode_aircraft_type(ac_type) — returns a list[float] of length 27
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Aircraft-type vocabulary (26 types from PRC-2025 + 1 unknown bucket)
# Sorted alphabetically so the column order is deterministic across Python
# versions and is easy to audit.
# ---------------------------------------------------------------------------

AIRCRAFT_TYPES: list[str] = sorted([
    "A20N", "A21N", "A306", "A318", "A319", "A320", "A321",
    "A332", "A333", "A359", "A388",
    "B38M", "B39M", "B737", "B738", "B739", "B744", "B748",
    "B752", "B763", "B772", "B77L", "B77W", "B788", "B789",
    "MD11",
    "__unknown__",  # catch-all for new/unseen types
])

# ---------------------------------------------------------------------------
# Numeric trajectory features (in fixed order)
# ---------------------------------------------------------------------------

_NUMERIC_FEATURES: list[str] = [
    "duration_s",
    "alt_change",
    "avg_speed",
    "max_vrate",
    "avg_altitude",
    "max_altitude",
    "alt_std",
    "avg_track_change",
    "avg_mach",
    "avg_tas",
    "avg_cas",
]

# One-hot column names derived from AIRCRAFT_TYPES
_AC_TYPE_COLUMNS: list[str] = [f"ac_{t}" for t in AIRCRAFT_TYPES]

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

FEATURE_COLUMNS: list[str] = _NUMERIC_FEATURES + _AC_TYPE_COLUMNS
INPUT_DIM: int = len(FEATURE_COLUMNS)  # 11 numeric + 27 one-hot = 38


# ---------------------------------------------------------------------------
# Helper: encode aircraft type → one-hot float list (length 27)
# ---------------------------------------------------------------------------

_AC_TYPE_INDEX: dict[str, int] = {t: i for i, t in enumerate(AIRCRAFT_TYPES)}


def encode_aircraft_type(ac_type: str | None) -> list[float]:
    """Return a one-hot encoded list[float] of length len(AIRCRAFT_TYPES).

    Unknown or NaN types map to the '__unknown__' bucket (last column).
    """
    vec = [0.0] * len(AIRCRAFT_TYPES)
    idx = _AC_TYPE_INDEX.get(ac_type or "__unknown__", _AC_TYPE_INDEX["__unknown__"])
    vec[idx] = 1.0
    return vec
