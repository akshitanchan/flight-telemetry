# refresh benchmark

I timed a full GROUP BY recompute of the routing-stats aggregate over 1,000,000 rows of inflated silver data against an incremental MERGE of a 10 percent batch into a pre-aggregated base, wall clock around each statement, with the same SQL on both engines apart from dialect. Both rows below come from one run each on the date shown.

| engine | rows | full recompute (s) | incremental merge (s) | ratio | hardware or warehouse | date |
|---|---:|---:|---:|---:|---|---|
| duckdb (local harness) | 1000000 | 0.046 | 0.0153 | 3.0x | Apple M2, 8 cores, 16 GB, macOS | 2026-09-11 |
| snowflake | 1000000 | 1.0923 | 1.1196 | 0.98x | X-Small warehouse, AWS_EU_WEST_2 | 2026-09-11 |

These commands produced the results:

`python data/experiments/refresh_strategy.py --sizes 1000000 --inc-pct 0.1`
`python data/cloud/snowflake/refresh_benchmark.py --rows 1000000 --inc-pct 0.1 --warehouse-size XSMALL`

On Snowflake at this size the incremental MERGE is no faster than the recompute, because a one-million-row GROUP BY already finishes in about a second on an X-Small warehouse and the MERGE pays the same fixed per-statement overhead plus the join. At 10 thousand and 100 thousand rows the same script measured 0.36x and 0.79x, so the ratio rises with size but had not crossed 1.0x by one million rows on this warehouse. The local DuckDB harness shows 3.0x because there is no per-statement overhead to amortise. The earlier README figure of roughly 2.5x was the DuckDB harness on an undated run and is replaced by the dated row above.

## dbt

`dbt build --target bigquery` and `dbt build --target snowflake` each finished with `Done. PASS=63 WARN=0 ERROR=0 SKIP=0 TOTAL=63` on 2026-09-11 with 4 models, 54 data tests, and 5 unit tests, run twice on each warehouse so the second run exercised the incremental MERGE path. The Snowflake run used the same X-Small warehouse and the BigQuery run the free tier with billing enabled for DML.
