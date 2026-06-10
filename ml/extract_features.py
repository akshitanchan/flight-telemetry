"""
ml/extract_features.py — Resumable, rich feature extraction for PRC-2025.
==========================================================================

Features extracted (see ml/features.py for the authoritative ordered list):
  Trajectory-derived (11):
    duration_s       — interval duration in seconds
    alt_change       — altitude[-1] - altitude[0] (ft)
    avg_speed        — mean groundspeed (kts); 0.0 when all NaN
    max_vrate        — max |vertical_rate| (ft/min); 0.0 when all NaN
    avg_altitude     — mean barometric altitude (ft); 0.0 when all NaN
    max_altitude     — max barometric altitude (ft); 0.0 when all NaN
    alt_std          — std dev of altitude (ft); 0.0 when < 2 pts
    avg_track_change — mean absolute per-step track change (deg/step); 0.0
                       when < 2 pts or all NaN
    avg_mach         — mean Mach; 0.0 when all NaN (real data mostly NaN)
    avg_tas          — mean True Air Speed (kts); 0.0 when all NaN
    avg_cas          — mean Calibrated Air Speed (kts); 0.0 when all NaN

  Aircraft-type one-hot (27):
    ac_{TYPE}        — one-hot vector from ml.features.AIRCRAFT_TYPES
                       Unknown types map to ac___unknown__

  Metadata (not in FEATURE_COLUMNS, but retained in the parquet for CV):
    aircraft_type    — raw string from flightlist (for slice analysis in cv.py)
    flight_id, idx, fuel_kg

NaN-fill strategy
-----------------
All NaN fills are applied at extraction time so the parquet on disk is already
clean.  The MLP (and FuelBurnDataset) receive zero NaN values.
  - mach / TAS / CAS  : 0.0  (missing for most ADS-B-only tracks)
  - groundspeed / vertical_rate / track : 0.0  (isolated dropouts)
  - degenerate intervals (< 2 pts) : all stats set to 0.0

Resumability / checkpointing
-----------------------------
When --checkpoint-dir is set (default: <data_dir>/features_checkpoint/),
each processed flight is written as a separate parquet shard named
<flight_id>.parquet inside that directory.  On startup, the set of already-
processed flight_ids is read from shard filenames; those flights are skipped.

Final assembly: all shards are concatenated and written to features_{split}.parquet.
Re-running after interruption is therefore idempotent: existing shards are not
re-written, and the final file is re-assembled from whatever shards exist.

To force a full re-extract, delete the checkpoint directory before running.
"""

import argparse
import zipfile
import io
import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm

from ml.features import AIRCRAFT_TYPES, encode_aircraft_type

logging.basicConfig(level=logging.INFO, format="%(asctime)s INFO  [%(name)s] %(message)s")
logger = logging.getLogger("ml.extract")


# ---------------------------------------------------------------------------
# Per-interval feature computation
# ---------------------------------------------------------------------------

def _safe_mean(series: pd.Series, fill: float = 0.0) -> float:
    v = series.dropna()
    return float(v.mean()) if len(v) > 0 else fill


def _safe_max(series: pd.Series, fill: float = 0.0) -> float:
    v = series.dropna()
    return float(v.max()) if len(v) > 0 else fill


def _interval_features(df_interval: pd.DataFrame, duration_s: float) -> dict:
    """Compute the numeric trajectory features for one fuel interval."""
    if len(df_interval) < 2:
        return {
            "duration_s": duration_s,
            "alt_change": 0.0,
            "avg_speed": 0.0,
            "max_vrate": 0.0,
            "avg_altitude": 0.0,
            "max_altitude": 0.0,
            "alt_std": 0.0,
            "avg_track_change": 0.0,
            "avg_mach": 0.0,
            "avg_tas": 0.0,
            "avg_cas": 0.0,
        }

    alt = df_interval["altitude"]
    alt_valid = alt.dropna()

    alt_change = 0.0
    if len(alt_valid) >= 2:
        alt_change = float(alt.dropna().iloc[-1] - alt.dropna().iloc[0])

    avg_altitude = _safe_mean(alt)
    max_altitude = _safe_max(alt)
    alt_std = float(alt_valid.std()) if len(alt_valid) >= 2 else 0.0
    if np.isnan(alt_std):
        alt_std = 0.0

    avg_speed = _safe_mean(df_interval["groundspeed"])
    max_vrate = _safe_max(df_interval["vertical_rate"].abs())

    # Track change: mean absolute first difference of track angle.
    # Track wraps at 360 → use circular difference.
    track = df_interval["track"].dropna()
    if len(track) >= 2:
        diff = track.diff().dropna()
        # Wrap into [-180, 180] to handle the 359→1 boundary correctly.
        diff = ((diff + 180) % 360) - 180
        avg_track_change = float(diff.abs().mean())
        if np.isnan(avg_track_change):
            avg_track_change = 0.0
    else:
        avg_track_change = 0.0

    avg_mach = _safe_mean(df_interval["mach"])
    avg_tas = _safe_mean(df_interval["TAS"])
    avg_cas = _safe_mean(df_interval["CAS"])

    return {
        "duration_s": duration_s,
        "alt_change": alt_change,
        "avg_speed": avg_speed,
        "max_vrate": max_vrate,
        "avg_altitude": avg_altitude,
        "max_altitude": max_altitude,
        "alt_std": alt_std,
        "avg_track_change": avg_track_change,
        "avg_mach": avg_mach,
        "avg_tas": avg_tas,
        "avg_cas": avg_cas,
    }


def _zero_features(duration_s: float = 0.0) -> dict:
    """Return zero-valued features for a flight missing from the zip."""
    return {
        "duration_s": duration_s,
        "alt_change": 0.0,
        "avg_speed": 0.0,
        "max_vrate": 0.0,
        "avg_altitude": 0.0,
        "max_altitude": 0.0,
        "alt_std": 0.0,
        "avg_track_change": 0.0,
        "avg_mach": 0.0,
        "avg_tas": 0.0,
        "avg_cas": 0.0,
    }


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def extract_features(
    data_dir: str,
    split: str = "train",
    limit: int | None = None,
    checkpoint_dir: str | None = None,
):
    """
    Extract per-interval features into features_{split}.parquet.

    Parameters
    ----------
    data_dir       : directory containing fuel_{split}.parquet,
                     flights_{split}.zip, and flightlist_{split}.parquet.
    split          : dataset split name ("train", "val", etc.)
    limit          : if set, process at most this many flights total
                     (including any already checkpointed).
    checkpoint_dir : directory for per-flight checkpoint shards.  Defaults to
                     <data_dir>/features_checkpoint_{split}/.  Set to "" or
                     "none" to disable checkpointing (single-pass mode).
    """
    data_path = Path(data_dir)
    fuel_path = data_path / f"fuel_{split}.parquet"
    zip_path = data_path / f"flights_{split}.zip"
    flightlist_path = data_path / f"flightlist_{split}.parquet"

    if not fuel_path.exists() or not zip_path.exists():
        raise FileNotFoundError(f"Missing required datasets in {data_path}")

    # --- Resolve checkpoint directory ---
    use_checkpointing = checkpoint_dir not in (None, "", "none", "None")
    if checkpoint_dir is None:
        ckpt_path = data_path / f"features_checkpoint_{split}"
        use_checkpointing = True
    elif checkpoint_dir in ("", "none", "None"):
        ckpt_path = None
        use_checkpointing = False
    else:
        ckpt_path = Path(checkpoint_dir)
        use_checkpointing = True

    if use_checkpointing:
        ckpt_path.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]
        # Read already-completed flight_ids from existing shard filenames.
        done_ids: set[str] = {p.stem for p in ckpt_path.glob("*.parquet")}
        logger.info(
            f"Checkpointing enabled at {ckpt_path}. "
            f"{len(done_ids)} flight(s) already processed — will be skipped."
        )
    else:
        done_ids = set()
        logger.info("Checkpointing disabled (single-pass mode).")

    # --- Load fuel intervals ---
    logger.info(f"Loading fuel intervals from {fuel_path}")
    df_fuel = pd.read_parquet(fuel_path)
    flight_intervals = df_fuel.groupby("flight_id")

    # --- Load aircraft types from flightlist (optional; graceful if missing) ---
    ac_type_map: dict[str, str] = {}
    if flightlist_path.exists():
        df_fl = pd.read_parquet(flightlist_path, columns=["flight_id", "aircraft_type"])
        ac_type_map = dict(zip(df_fl["flight_id"], df_fl["aircraft_type"]))
        logger.info(f"Loaded aircraft types for {len(ac_type_map)} flights.")
    else:
        logger.warning(
            f"flightlist_{split}.parquet not found at {flightlist_path}; "
            "aircraft_type will be unknown for all flights."
        )

    # --- Determine processing scope ---
    # With checkpointing, flights already done are skipped but still count
    # toward the limit so --limit N always gives N flights total in the output.
    all_flight_ids = [fid for fid, _ in flight_intervals]
    if limit is not None:
        # Honour the limit across the total output (done + to-do combined).
        already_in_limit = sum(1 for fid in all_flight_ids if str(fid) in done_ids)
        remaining_budget = max(0, limit - already_in_limit)
        to_process = [
            fid for fid in all_flight_ids
            if str(fid) not in done_ids
        ][:remaining_budget]
    else:
        to_process = [fid for fid in all_flight_ids if str(fid) not in done_ids]

    logger.info(
        f"Flights to process this run: {len(to_process)} "
        f"(already done: {len(done_ids)})"
    )

    # --- Main extraction loop ---
    in_memory_rows: list[dict] = []  # used when checkpointing is disabled

    logger.info(f"Extracting features from {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zip_map = {Path(n).stem: n for n in zf.namelist() if n.endswith(".parquet")}

        for flight_id in tqdm(to_process, desc="Processing flights"):
            group = flight_intervals.get_group(flight_id)
            ac_type = ac_type_map.get(str(flight_id), None)
            ac_onehot = encode_aircraft_type(ac_type)
            ac_onehot_dict = {f"ac_{t}": v for t, v in zip(AIRCRAFT_TYPES, ac_onehot)}

            entry = zip_map.get(str(flight_id))
            if entry is None:
                # Flight missing from zip — emit zero-features, keep the labels.
                flight_rows = []
                for _, row in group.iterrows():
                    duration_s = (
                        pd.to_datetime(row["end"]) - pd.to_datetime(row["start"])
                    ).total_seconds()
                    feat = _zero_features(duration_s)
                    feat.update(ac_onehot_dict)
                    feat.update({
                        "idx": row["idx"],
                        "flight_id": flight_id,
                        "aircraft_type": ac_type,
                        "fuel_kg": row["fuel_kg"],
                    })
                    flight_rows.append(feat)
            else:
                with zf.open(entry) as f:
                    df_traj = pd.read_parquet(io.BytesIO(f.read()))
                df_traj["timestamp"] = pd.to_datetime(df_traj["timestamp"])

                flight_rows = []
                for _, row in group.iterrows():
                    start_ts = pd.to_datetime(row["start"])
                    end_ts = pd.to_datetime(row["end"])
                    duration_s = (end_ts - start_ts).total_seconds()

                    mask = (
                        (df_traj["timestamp"] >= start_ts)
                        & (df_traj["timestamp"] <= end_ts)
                    )
                    df_interval = df_traj[mask]

                    feat = _interval_features(df_interval, duration_s)
                    feat.update(ac_onehot_dict)
                    feat.update({
                        "idx": row["idx"],
                        "flight_id": flight_id,
                        "aircraft_type": ac_type,
                        "fuel_kg": row["fuel_kg"],
                    })
                    flight_rows.append(feat)

            # --- Write checkpoint shard or accumulate in-memory ---
            if use_checkpointing:
                shard_df = pd.DataFrame(flight_rows)
                shard_path = ckpt_path / f"{flight_id}.parquet"  # type: ignore[operator]
                shard_df.to_parquet(shard_path)
            else:
                in_memory_rows.extend(flight_rows)

    # --- Assemble final parquet ---
    out_path = data_path / f"features_{split}.parquet"

    if use_checkpointing:
        # Gather ALL shards (previously done + just processed) up to the limit.
        all_shards = sorted(ckpt_path.glob("*.parquet"))  # type: ignore[union-attr]
        if limit is not None:
            # Keep only shards whose flight_id was in the overall allowed set.
            allowed_ids = set(str(fid) for fid in all_flight_ids[:limit])
            all_shards = [s for s in all_shards if s.stem in allowed_ids]
        if not all_shards:
            logger.warning("No checkpoint shards found — output parquet will be empty.")
            df_features = pd.DataFrame()
        else:
            df_features = pd.concat(
                [pd.read_parquet(s) for s in tqdm(all_shards, desc="Assembling shards")],
                ignore_index=True,
            )
    else:
        df_features = pd.DataFrame(in_memory_rows)

    df_features.to_parquet(out_path)
    logger.info(f"Successfully extracted features for {len(df_features)} intervals.")
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract and cache per-interval trajectory features."
    )
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N flights total (including already-checkpointed ones).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help=(
            "Directory for per-flight checkpoint shards. "
            "Defaults to <data-dir>/features_checkpoint_<split>/. "
            "Pass '' or 'none' to disable checkpointing."
        ),
    )
    args = parser.parse_args()

    if args.data_dir == "data/ml/prc_2025_mock" and not Path(args.data_dir).exists():
        from ml.mock_data import generate_mock_eurocontrol_data
        logger.info(f"Generating mock dataset in {args.data_dir}...")
        generate_mock_eurocontrol_data(args.data_dir, num_flights=20)

    extract_features(
        args.data_dir,
        args.split,
        limit=args.limit,
        checkpoint_dir=args.checkpoint_dir,
    )
