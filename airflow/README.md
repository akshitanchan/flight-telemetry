I built this Astro project to orchestrate the flight-telemetry lakehouse pipeline in Airflow. The DAG that lives here walks the medallion hops from bronze to gold, checking each hop's output against the repo's data contracts before it moves on, then runs dbt build and dbt test against Snowflake to refresh the marts.

I source airflow/.env in my shell first (`set -a; source .env; set +a`), because docker compose substitutes the key file path from the shell environment rather than from the file, then run `astro dev start` from this directory to bring up the local scheduler, triggerer, dag processor, and API server. The API server answers on port 8090 and Postgres on 5440, so they do not collide with the kind cluster and the local Postgres I run alongside.

The repo root is mounted read-write into the scheduler, triggerer, and dag processor containers at /usr/local/airflow/repo, so the DAG imports the repo's own Python modules directly and runs dbt against the profile checked into data/cloud/dbt/flight_telemetry.

Before starting, I copy the root .env.example to airflow/.env and fill in SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PRIVATE_KEY_PATH, SNOWFLAKE_ROLE, SNOWFLAKE_WAREHOUSE, SNOWFLAKE_DATABASE, SNOWFLAKE_SCHEMA, BIGQUERY_PROJECT, and GOOGLE_APPLICATION_CREDENTIALS; the key file itself is mounted into the container.

Data the DAG writes lands back in the repo, under data/interim, data/processed, and outputs at the repo root, plus dbt's own target and logs directories under data/cloud/dbt/flight_telemetry, since all of those are the same mount the containers see.

Airflow's own scheduler and task logs stay inside Docker, not in this folder.
