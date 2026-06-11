# Refresh Experiment Results — ds-05 (W4.3)

**Instructions for the owner**: run `03_refresh_experiment.py` in the serverless
medallion job, then copy the Section D output into the table below.

The DuckDB proxy (local smoke) numbers are in the row labelled `DuckDB proxy`
at the bottom of each table.  They are a sanity reference, not the headline.

---

## Compute spec (fill in)

| Field | Value |
|---|---|
| Run date (UTC) | YYYY-MM-DD HH:MM |
| Compute type | Databricks serverless |
| Environment version | `2` |
| Runtime / Spark version shown by run | ??? |
| Serverless performance mode | ??? |
| silver_table | `<catalog>.<schema>.silver_flight_state` |
| gold_congestion_table | `<catalog>.<schema>.gold_airport_congestion` |
| inc_pct | 0.10 |

---

## Timing results (fill in from Section D output)

| target_size | base_rows | inc_rows | full_recompute_s | incremental_merge_s | speedup_factor |
|---|---|---|---|---|---|
| 10 000 | 9 000 | 1 000 | ???.??? | ???.??? | ???.??x |
| 50 000 | 45 000 | 5 000 | ???.??? | ???.??? | ???.??x |
| 200 000 | 180 000 | 20 000 | ???.??? | ???.??? | ???.??x |
| *(optional)* 500 000 | 450 000 | 50 000 | ???.??? | ???.??? | ???.??x |

---

## DuckDB proxy results (local smoke — reference only, NOT the headline)

Run with: `python data/cloud/databricks/experiments/harness.py --sizes 10000,50000,200000`

| target_size | base_rows | inc_rows | full_recompute_s | incremental_merge_s | speedup_factor |
|---|---|---|---|---|---|
| *(paste harness output here)* | | | | | |

---

## Observations (fill in)

*Describe where MERGE wins, where full recompute is cheaper, and why
(e.g. small-incremental MERGE amortises write overhead at low row counts
but breaks even at scale due to Delta transaction overhead vs. columnar scan).*

---

## Idempotency check

Re-ran Section C with identical widget values on the same cluster:

- Second run MERGE new rows inserted: ???
- Second run MERGE rows updated: ???
- Result: PASS / FAIL (expected: 0 new, 0 updated)

---

*Completed by*: &lt;owner name&gt;  
*Date*: YYYY-MM-DD  
*Notebook path*: `data/cloud/databricks/notebooks/03_refresh_experiment.py`
