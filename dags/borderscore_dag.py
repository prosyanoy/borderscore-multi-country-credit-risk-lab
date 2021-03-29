"""Airflow DAG wrapping the BorderScore pipeline (same task table as the plain DAG runner).

Run locally with ``airflow standalone`` and AIRFLOW__CORE__DAGS_FOLDER pointing at this folder
(see docker-compose.yml, profile "airflow"). Requires ``pip install -e .`` in the Airflow env.
"""
from __future__ import annotations

import os
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

from borderscore.config import Settings
from borderscore.tasks import PIPELINE, RunContext


def _run(task_name: str, **airflow_ctx) -> None:
    spec = next(t for t in PIPELINE if t.name == task_name)
    spec.fn(RunContext(Settings.from_env(), run_id=airflow_ctx["run_id"]))


with DAG(
    dag_id="borderscore_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule=os.getenv("BORDERSCORE_SCHEDULE") or None,
    catchup=False,
    tags=["borderscore"],
) as dag:
    operators = {t.name: PythonOperator(task_id=t.name, python_callable=_run, op_kwargs={"task_name": t.name})
                 for t in PIPELINE}
    for t in PIPELINE:
        for dep in t.deps:
            operators[dep] >> operators[t.name]
