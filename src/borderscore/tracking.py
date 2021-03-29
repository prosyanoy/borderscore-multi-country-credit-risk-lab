"""Thin MLflow helpers (local SQLite backend by default, file artifacts under BORDERSCORE_HOME)."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import mlflow

from .config import Settings


def setup(settings: Settings) -> None:
    settings.home.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(settings.mlflow_uri)
    if mlflow.get_experiment_by_name(settings.experiment) is None:
        mlflow.create_experiment(settings.experiment, artifact_location=(settings.home / "mlartifacts").as_uri())
    mlflow.set_experiment(settings.experiment)


@contextmanager
def run(settings: Settings, name: str, stage: str, pipeline_run: str = "", params: dict | None = None):
    setup(settings)
    with mlflow.start_run(run_name=name) as r:
        mlflow.set_tags({"stage": stage, "pipeline_run": pipeline_run})
        if params:
            mlflow.log_params(params)
        yield r


def flat_metrics(prefix: str, by_country: dict[str, dict[str, float]]) -> dict[str, float]:
    """{'BR': {'auc': .7}} -> {'<prefix>_auc_BR': .7}; the macro average is named '<prefix>_macro_auc'."""
    out = {}
    for country, ms in by_country.items():
        for k, v in ms.items():
            out[f"{prefix}_macro_{k}" if country == "macro" else f"{prefix}_{k}_{country}"] = float(v)
    return out


def log_artifact(path: Path, artifact_dir: str | None = None) -> None:
    mlflow.log_artifact(str(path), artifact_dir)
