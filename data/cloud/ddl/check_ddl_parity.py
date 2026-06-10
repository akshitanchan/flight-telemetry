#!/usr/bin/env python3
"""
Offline DDL parity checker for the flight-telemetry cloud layer.

Parses each Delta and BigQuery SQL DDL file (column name, type, nullability,
PARTITION BY clause, CLUSTER BY clause) and asserts a 1:1 match against the
corresponding JSON Schema contract.

Covers all 10 files: 5 Delta + 5 BigQuery.

Exit codes:
    0  all checks passed
    1  one or more parity violations found

Usage:
    python data/cloud/ddl/check_ddl_parity.py
    python data/cloud/ddl/check_ddl_parity.py --verbose

No third-party dependencies — stdlib only.
"""

import json
import re
import sys
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
DDL_DIR = THIS_FILE.parent
DELTA_DIR = DDL_DIR / "delta"
BQ_DIR = DDL_DIR / "bigquery"
CONTRACTS_DIR = THIS_FILE.parents[3] / "shared" / "contracts"


# ---------------------------------------------------------------------------
# Type maps (schema JSON type -> expected DDL type token, upper-cased)
# ---------------------------------------------------------------------------
# "format: date-time" fields are ALWAYS mapped to TIMESTAMP regardless of the
# base JSON type (they are always "string" in the contracts).
DATETIME_FIELDS = {
    "window_start", "window_end", "event_ts", "first_seen_ts", "last_seen_ts",
}

DELTA_TYPE_MAP = {
    "string":   "STRING",
    "number":   "DOUBLE",
    "integer":  "BIGINT",
    "boolean":  "BOOLEAN",
}

BQ_TYPE_MAP = {
    "string":   "STRING",
    "number":   "FLOAT64",
    "integer":  "INT64",
    "boolean":  "BOOL",
}

# ---------------------------------------------------------------------------
# C3-locked partition/cluster expectations
# ---------------------------------------------------------------------------
# Format: (partition_col_or_expr, cluster_col_or_None)
# None means no partition or cluster expected.
#
# Delta:     partition is on a derived DATE column (window_date / event_date).
#            We check that PARTITIONED BY is present and contains a date column
#            name. ZORDER is advisory (applied via OPTIMIZE); we do NOT assert
#            it in the DDL text.
# BigQuery:  partition is the literal expression DATE(window_start) etc.
#            Cluster is the column name.

EXPECTATIONS = {
    # table_name -> {
    #   "delta":    (partition_present: bool, cluster_col: str|None),
    #   "bigquery": (partition_expr: str|None, cluster_col: str|None),
    # }
    "silver_flight_state": {
        "delta":    (True, None),
        "bigquery": ("DATE(event_ts)", None),
    },
    "gold_airport_congestion": {
        "delta":    (True, None),          # ZORDER advisory; not in DDL body
        "bigquery": ("DATE(window_start)", "airport_icao"),
    },
    "gold_sector_load": {
        "delta":    (True, None),
        "bigquery": ("DATE(window_start)", "h3_r4"),
    },
    "gold_emergency_events": {
        "delta":    (False, None),
        "bigquery": (None, None),
    },
    "gold_routing_stats": {
        "delta":    (False, None),
        "bigquery": (None, None),
    },
}


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def load_schema(table_name: str) -> dict:
    path = CONTRACTS_DIR / f"{table_name}.schema.json"
    with open(path) as fh:
        return json.load(fh)


def schema_columns(schema: dict, dialect: str) -> dict[str, dict]:
    """
    Return {col_name: {"type": EXPECTED_DDL_TYPE, "nullable": bool}}
    for every property in the schema.

    dialect: "delta" | "bigquery"
    """
    type_map = DELTA_TYPE_MAP if dialect == "delta" else BQ_TYPE_MAP
    required = set(schema.get("required", []))
    cols = {}
    for col, prop in schema["properties"].items():
        raw_type = prop.get("type")
        fmt = prop.get("format", "")

        # Resolve base type from union or scalar
        if isinstance(raw_type, list):
            # e.g. ["number", "null"] or ["string", "null"]
            base_types = [t for t in raw_type if t != "null"]
            base_type = base_types[0] if base_types else "string"
            nullable = True  # union with null => nullable
        else:
            base_type = raw_type
            nullable = col not in required

        # date-time format always → TIMESTAMP
        if fmt == "date-time" or col in DATETIME_FIELDS:
            ddl_type = "TIMESTAMP"
        else:
            ddl_type = type_map.get(base_type, "STRING")

        cols[col] = {"type": ddl_type, "nullable": nullable}
    return cols


# ---------------------------------------------------------------------------
# DDL parser
# ---------------------------------------------------------------------------
# Lightweight hand-rolled parser — no grammar, just regex over normalised text.
# Handles the specific DDL style produced by this project.

def _strip_comments(sql: str) -> str:
    """Remove -- line comments and /* */ block comments."""
    # Block comments first
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    # Line comments
    sql = re.sub(r"--[^\n]*", " ", sql)
    return sql


def _normalise(sql: str) -> str:
    """Collapse whitespace and upper-case for easier regex matching."""
    return re.sub(r"\s+", " ", _strip_comments(sql)).strip().upper()


def parse_ddl(sql_path: Path) -> dict:
    """
    Parse a DDL file and return:
    {
        "columns":   {col_name_lower: {"type": STR, "nullable": bool}},
        "partition": str | None,   # normalised partition expression (upper)
        "cluster":   str | None,   # cluster column (lower, first token only)
    }
    """
    raw = sql_path.read_text()
    norm = _normalise(raw)

    # ---- Extract the column block between the outermost parentheses ----------
    # Find the first '(' after CREATE TABLE ... and its matching ')'
    create_match = re.search(r"CREATE\s+(OR\s+REPLACE\s+)?TABLE\b", norm)
    if not create_match:
        raise ValueError(f"No CREATE TABLE found in {sql_path}")

    after_create = norm[create_match.end():]
    paren_start = after_create.index("(")
    # Walk to find the matching closing paren (accounting for nesting)
    depth = 0
    col_block_end = None
    for i, ch in enumerate(after_create[paren_start:], start=paren_start):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                col_block_end = i
                break
    if col_block_end is None:
        raise ValueError(f"Unbalanced parentheses in {sql_path}")

    col_block = after_create[paren_start + 1 : col_block_end]
    after_cols = after_create[col_block_end + 1 :]  # text after closing paren

    # ---- Parse individual column definitions --------------------------------
    # Split on commas, but commas inside nested parens (TBLPROPERTIES-like) are
    # already excluded since we've extracted only the column block.
    columns = {}
    for token in col_block.split(","):
        token = token.strip()
        if not token:
            continue
        # Skip constraint lines (PRIMARY KEY, FOREIGN KEY, etc.)
        if re.match(r"(PRIMARY|FOREIGN|UNIQUE|CHECK|CONSTRAINT)\b", token):
            continue
        parts = token.split()
        if len(parts) < 2:
            continue
        col_name = parts[0].strip("`\"[]").lower()
        col_type = parts[1]
        # Handle two-word types like NOT NULL modifier: parts[2] might be NOT
        not_null = "NOT NULL" in token
        nullable = not not_null
        columns[col_name] = {"type": col_type, "nullable": nullable}

    # ---- Extract PARTITION BY -----------------------------------------------
    partition_match = re.search(
        r"PARTITION\s+BY\s+((?:DATE\s*\([^)]+\)|[A-Z0-9_]+))",
        after_cols,
    )
    # Also check PARTITIONED BY (Delta style)
    partitioned_match = re.search(
        r"PARTITIONED\s+BY\s*\(\s*([A-Z0-9_]+)\s*\)",
        after_cols,
    )

    partition_expr = None
    if partition_match:
        partition_expr = re.sub(r"\s+", "", partition_match.group(1))  # e.g. DATE(WINDOW_START)
    elif partitioned_match:
        partition_expr = partitioned_match.group(1)  # e.g. WINDOW_DATE

    # ---- Extract CLUSTER BY -------------------------------------------------
    cluster_match = re.search(r"CLUSTER\s+BY\s+([A-Z0-9_]+)", after_cols)
    cluster_col = cluster_match.group(1).lower() if cluster_match else None

    return {
        "columns":   columns,
        "partition": partition_expr,
        "cluster":   cluster_col,
    }


# ---------------------------------------------------------------------------
# Parity assertion
# ---------------------------------------------------------------------------

def check_parity(table_name: str, dialect: str, verbose: bool) -> list[str]:
    """
    Compare the DDL file against the schema contract.
    Returns a list of error strings (empty = all good).
    """
    ddl_dir = DELTA_DIR if dialect == "delta" else BQ_DIR
    sql_path = ddl_dir / f"{table_name}.sql"

    if not sql_path.exists():
        return [f"DDL file not found: {sql_path}"]

    schema = load_schema(table_name)
    expected_cols = schema_columns(schema, dialect)
    exp = EXPECTATIONS[table_name][dialect]
    expect_partition, expect_cluster = exp

    try:
        parsed = parse_ddl(sql_path)
    except Exception as exc:
        return [f"Failed to parse {sql_path}: {exc}"]

    errors = []
    actual_cols = parsed["columns"]

    # -- Column set: no extra, no missing -------------------------------------
    expected_names = set(expected_cols.keys())
    actual_names = set(actual_cols.keys())

    # Delta DDL may include the derived partition helper column (event_date /
    # window_date); we exclude it from the schema-column comparison since it
    # is not in the schema but is required for Delta partitioning.
    DELTA_HELPER_COLS = {"event_date", "window_date"}
    if dialect == "delta":
        actual_names_schema = actual_names - DELTA_HELPER_COLS
    else:
        actual_names_schema = actual_names

    missing = expected_names - actual_names_schema
    extra   = actual_names_schema - expected_names
    for col in sorted(missing):
        errors.append(f"[{dialect}] {table_name}: missing column '{col}'")
    for col in sorted(extra):
        errors.append(f"[{dialect}] {table_name}: extra column '{col}' not in schema")

    # -- Per-column type and nullability --------------------------------------
    for col in sorted(expected_names & actual_names_schema):
        exp_col = expected_cols[col]
        act_col = actual_cols[col]

        if act_col["type"] != exp_col["type"]:
            errors.append(
                f"[{dialect}] {table_name}.{col}: type mismatch — "
                f"expected {exp_col['type']}, got {act_col['type']}"
            )
        if act_col["nullable"] != exp_col["nullable"]:
            exp_null = "nullable" if exp_col["nullable"] else "NOT NULL"
            act_null = "nullable" if act_col["nullable"] else "NOT NULL"
            errors.append(
                f"[{dialect}] {table_name}.{col}: nullability mismatch — "
                f"expected {exp_null}, got {act_null}"
            )

    # -- Partition ------------------------------------------------------------
    if dialect == "delta":
        has_partition = parsed["partition"] is not None
        if expect_partition and not has_partition:
            errors.append(
                f"[delta] {table_name}: expected PARTITIONED BY clause, none found"
            )
        elif not expect_partition and has_partition:
            errors.append(
                f"[delta] {table_name}: unexpected PARTITIONED BY clause present"
            )
    else:
        # BigQuery: check exact partition expression (normalised, no spaces)
        actual_partition = parsed["partition"]  # already normalised upper
        if expect_partition is not None:
            norm_expect = re.sub(r"\s+", "", expect_partition.upper())
            if actual_partition != norm_expect:
                errors.append(
                    f"[bigquery] {table_name}: PARTITION BY mismatch — "
                    f"expected '{norm_expect}', got '{actual_partition}'"
                )
        else:
            if actual_partition is not None:
                errors.append(
                    f"[bigquery] {table_name}: unexpected PARTITION BY '{actual_partition}'"
                )

    # -- Cluster --------------------------------------------------------------
    actual_cluster = parsed["cluster"]
    if expect_cluster is not None:
        norm_expect_c = expect_cluster.lower()
        if actual_cluster != norm_expect_c:
            errors.append(
                f"[{dialect}] {table_name}: CLUSTER BY mismatch — "
                f"expected '{norm_expect_c}', got '{actual_cluster}'"
            )
    else:
        if actual_cluster is not None:
            errors.append(
                f"[{dialect}] {table_name}: unexpected CLUSTER BY '{actual_cluster}'"
            )

    if verbose and not errors:
        print(f"  OK  [{dialect:8s}] {table_name}")

    return errors


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

TABLES = [
    "silver_flight_state",
    "gold_airport_congestion",
    "gold_sector_load",
    "gold_emergency_events",
    "gold_routing_stats",
]

DIALECTS = ["delta", "bigquery"]


def main() -> int:
    parser = argparse.ArgumentParser(description="DDL parity checker")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print OK lines as well as failures")
    args = parser.parse_args()

    all_errors: list[str] = []

    print("=" * 64)
    print("DDL Parity Checker — flight-telemetry cloud layer")
    print("=" * 64)

    for table in TABLES:
        for dialect in DIALECTS:
            errs = check_parity(table, dialect, verbose=args.verbose)
            if errs:
                for e in errs:
                    print(f"  FAIL  {e}")
            all_errors.extend(errs)

    print("=" * 64)
    if all_errors:
        print(f"RESULT: {len(all_errors)} violation(s) found — DDL is out of parity with schemas.")
        return 1
    else:
        print(f"RESULT: all 10 DDL files pass parity checks (5 Delta + 5 BigQuery).")
        return 0


if __name__ == "__main__":
    sys.exit(main())
