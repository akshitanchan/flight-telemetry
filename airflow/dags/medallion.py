"""medallion dag: bronze landing -> silver -> gold -> snowflake landing -> dbt build/test."""

from datetime import datetime

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

REPO = "/usr/local/airflow/repo"

# bronze -> silver, ending with the contract check on the silver output.
# there is no bronze contract in shared/contracts/ yet, so this is the first check.
INGEST_TASKS = [
    ("data_sample", "python data/scripts/bootstrap.py --mode sample"),
    ("ingest", "python -m systems.replay.cli --input data/raw/sample_state_vectors.jsonl"),
    ("bronze_to_silver", "python -m data.transforms.cli --input data/interim/landing.jsonl"),
    (
        "check_silver_contract",
        "python shared/contracts/validate_schemas.py "
        "--data data/processed/silver_flight_state.jsonl --schema silver_flight_state",
    ),
]

# task id doubles as the gold table's file base name and contract name.
GOLD_TABLES = [
    ("gold_airport_congestion", "congestion"),
    ("gold_sector_load", "sector"),
    ("gold_emergency_events", "emergency"),
    ("gold_routing_stats", "routing"),
]

CHECK_GOLD_CONTRACTS = " && ".join(
    f"python shared/contracts/validate_schemas.py "
    f"--data data/processed/{name}.jsonl --schema {name}"
    for name, _ in GOLD_TABLES
)

WAREHOUSE_TASKS = [
    ("check_gold_contracts", CHECK_GOLD_CONTRACTS),
    ("export_gold_parquet", "python data/scripts/export_gold_parquet.py"),
    ("load_snowflake_landing", "python data/cloud/snowflake/load_landing.py"),
    ("dbt_run", "cd data/cloud/dbt/flight_telemetry && dbt run --target snowflake"),
    ("dbt_test", "cd data/cloud/dbt/flight_telemetry && dbt test --target snowflake"),
]

with DAG(
    dag_id="medallion",
    schedule=None,
    catchup=False,
    start_date=datetime(2024, 1, 1),
    default_args={"retries": 0},
) as dag:
    ingest_ops = {
        task_id: BashOperator(task_id=task_id, bash_command=command, cwd=REPO)
        for task_id, command in INGEST_TASKS
    }
    gold_ops = {
        task_id: BashOperator(
            task_id=task_id,
            bash_command=(
                "python -m data.transforms.cli_gold "
                f"--input data/processed/silver_flight_state.jsonl --table {table}"
            ),
            cwd=REPO,
        )
        for task_id, table in GOLD_TABLES
    }
    warehouse_ops = {
        task_id: BashOperator(task_id=task_id, bash_command=command, cwd=REPO)
        for task_id, command in WAREHOUSE_TASKS
    }

    ingest_ops["data_sample"] >> ingest_ops["ingest"] >> ingest_ops["bronze_to_silver"]
    ingest_ops["bronze_to_silver"] >> ingest_ops["check_silver_contract"]

    for gold_task_id in gold_ops:
        ingest_ops["check_silver_contract"] >> gold_ops[gold_task_id]
        gold_ops[gold_task_id] >> warehouse_ops["check_gold_contracts"]

    warehouse_ops["check_gold_contracts"] >> warehouse_ops["export_gold_parquet"]
    warehouse_ops["export_gold_parquet"] >> warehouse_ops["load_snowflake_landing"]
    warehouse_ops["load_snowflake_landing"] >> warehouse_ops["dbt_run"] >> warehouse_ops["dbt_test"]
