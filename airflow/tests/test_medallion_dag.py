#!/usr/bin/env python3
"""structural tests for the medallion dag: task count, edges, operator shape, no leaked creds."""

import re
import unittest

from airflow.models import DagBag
from airflow.providers.standard.operators.bash import BashOperator

DAG_FOLDER = "/usr/local/airflow/dags"
REPO_CWD = "/usr/local/airflow/repo"

GOLD_TASK_IDS = {
    "gold_airport_congestion",
    "gold_sector_load",
    "gold_emergency_events",
    "gold_routing_stats",
}

CREDENTIAL_PATTERN = re.compile(r"password|secret|token", re.IGNORECASE)


class MedallionDagTestCase(unittest.TestCase):
    """loads the dag bag once and checks the medallion dag's shape."""

    @classmethod
    def setUpClass(cls):
        cls.dag_bag = DagBag(dag_folder=DAG_FOLDER, include_examples=False)
        cls.dag = cls.dag_bag.dags.get("medallion")

    def test_dag_imports_without_errors(self):
        """the dag file parses with zero import errors and registers as medallion."""
        self.assertEqual(self.dag_bag.import_errors, {})
        self.assertIn("medallion", self.dag_bag.dags)

    def test_dag_has_thirteen_tasks(self):
        """the dag has exactly the 13 tasks wired in medallion.py."""
        self.assertEqual(len(self.dag.tasks), 13)

    def test_check_silver_contract_gates_the_gold_tasks(self):
        """check_silver_contract sits between bronze_to_silver and every gold task."""
        check = self.dag.get_task("check_silver_contract")
        self.assertIn("bronze_to_silver", check.upstream_task_ids)
        self.assertEqual(check.downstream_task_ids, GOLD_TASK_IDS)

    def test_gold_tasks_only_feed_check_gold_contracts(self):
        """all four gold tasks converge on check_gold_contracts and nowhere else."""
        for task_id in GOLD_TASK_IDS:
            gold_task = self.dag.get_task(task_id)
            self.assertEqual(gold_task.downstream_task_ids, {"check_gold_contracts"})

        check_gold = self.dag.get_task("check_gold_contracts")
        self.assertEqual(check_gold.upstream_task_ids, GOLD_TASK_IDS)

    def test_dbt_test_is_the_single_leaf(self):
        """dbt_test is the only task with no downstream tasks."""
        leaves = [task.task_id for task in self.dag.tasks if not task.downstream_task_ids]
        self.assertEqual(leaves, ["dbt_test"])

    def test_snowflake_load_sits_between_export_and_dbt_run(self):
        """load_snowflake_landing runs after the parquet export and before dbt run."""
        load_task = self.dag.get_task("load_snowflake_landing")
        self.assertEqual(load_task.upstream_task_ids, {"export_gold_parquet"})
        self.assertEqual(load_task.downstream_task_ids, {"dbt_run"})

    def test_every_task_is_a_bash_operator_rooted_in_the_repo(self):
        """every task shells out via BashOperator with the repo mount as its cwd."""
        for task in self.dag.tasks:
            self.assertIsInstance(task, BashOperator)
            self.assertEqual(task.cwd, REPO_CWD)

    def test_no_bash_command_contains_a_credential_looking_string(self):
        """none of the bash commands embed a password/secret/token literal."""
        for task in self.dag.tasks:
            match = CREDENTIAL_PATTERN.search(task.bash_command)
            self.assertIsNone(
                match, f"{task.task_id} bash_command looks like it embeds a credential"
            )


if __name__ == "__main__":
    unittest.main()
