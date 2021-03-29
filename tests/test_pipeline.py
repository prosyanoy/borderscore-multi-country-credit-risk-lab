import json
from dataclasses import replace

import pytest

from borderscore.config import Settings
from borderscore.coldstart import METHODS, plot, run_experiment, summarise
from borderscore.tasks import RunContext, run_pipeline


@pytest.fixture(scope="module")
def settings(tmp_path_factory):
    home = tmp_path_factory.mktemp("bs")
    return replace(Settings(), home=home, mlflow_uri=f"sqlite:///{home / 'mlflow.db'}",
                   n_source=1200, n_new=300, n_holdout=1200)


def test_pipeline_end_to_end_and_gate_blocks_identical_rerun(settings):
    res = run_pipeline(RunContext(settings, "run1", seed=1))
    assert all(r["status"] == "success" for r in res.values()), res
    gate1 = json.loads((settings.runs_dir / "run1" / "gate.json").read_text())
    assert gate1["promoted"] is True
    assert (settings.registry_dir / "champion" / "model.joblib").exists()
    assert (settings.runs_dir / "run1" / "monitoring" / "report.md").exists()

    res = run_pipeline(RunContext(settings, "run2", seed=1))  # same data, same models -> no strict win
    assert all(r["status"] == "success" for r in res.values()), res
    gate2 = json.loads((settings.runs_dir / "run2" / "gate.json").read_text())
    assert gate2["promoted"] is False
    history = (settings.registry_dir / "history.jsonl").read_text().strip().splitlines()
    assert len(history) == 2

    import mlflow
    mlflow.set_tracking_uri(settings.mlflow_uri)
    runs = mlflow.search_runs(experiment_names=[settings.experiment])
    assert {"train", "evaluate", "gate", "monitor", "data"} <= set(runs["tags.stage"])
    ev = runs[runs["tags.stage"] == "evaluate"].iloc[0]
    for m in ("auc", "ks", "logloss", "ece"):
        assert f"metrics.holdout_{m}_ID" in runs.columns and ev[f"metrics.holdout_{m}_BR"] > 0


def test_coldstart_experiment(tmp_path):
    results = run_experiment(sizes=[60, 300], seeds=[0], n_source=800, n_test=1500)
    assert set(results.method) == set(METHODS) | {"zero_shot"}
    assert results.auc.between(0.4, 1.0).all()
    assert (summarise(results).method == "zero_shot").any()
    assert plot(results, tmp_path / "p.png").stat().st_size > 1000
