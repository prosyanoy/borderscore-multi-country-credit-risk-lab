"""Pipeline tasks: data generation -> training -> evaluation -> promotion gate -> monitoring.

Each task takes a RunContext and communicates with others only through files in the run
directory, so the same functions run under the plain DAG runner or as Airflow tasks.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import mlflow
import pandas as pd

from . import tracking
from .config import Settings
from .data import generate_splits
from .metrics import evaluate_by_country
from .models import fit_model, predict_df
from .monitoring import DEFAULT_SCENARIO, build_report
from .registry import Registry, decide
from .dag import TaskSpec, run_dag

CANDIDATES = ("lightgbm", "xgboost", "mlp")


@dataclass
class RunContext:
    settings: Settings
    run_id: str = ""
    seed: int | None = None
    scenario: dict | None = None  # monitoring drift scenario; None = default

    def __post_init__(self):
        self.run_id = re.sub(r"[^A-Za-z0-9_.-]", "_", self.run_id or datetime.now().strftime("%Y%m%d-%H%M%S"))
        if self.seed is None:
            self.seed = self.settings.seed

    @property
    def dir(self) -> Path:
        d = self.settings.runs_dir / self.run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def read(self, split: str) -> pd.DataFrame:
        return pd.read_csv(self.dir / "data" / f"{split}.csv.gz")


def generate_data(ctx: RunContext) -> None:
    splits = generate_splits(ctx.settings.train_sizes(), ctx.seed, ctx.settings.n_holdout)
    (ctx.dir / "data").mkdir(exist_ok=True)
    for name, df in splits.items():
        df.to_csv(ctx.dir / "data" / f"{name}.csv.gz", index=False)
    with tracking.run(ctx.settings, "generate_data", "data", ctx.run_id,
                      {"seed": ctx.seed, **{f"n_{k}": len(v) for k, v in splits.items()}}):
        for name, df in splits.items():
            mlflow.log_metrics({f"default_rate_{name}_{c}": float(r) for c, r in df.groupby("country").default.mean().items()})


def _train(name: str):
    def task(ctx: RunContext) -> None:
        train, valid = ctx.read("train"), ctx.read("valid")
        model = fit_model(name, train, valid, seed=ctx.seed)
        path = ctx.dir / "models" / f"{name}.joblib"
        path.parent.mkdir(exist_ok=True)
        joblib.dump(model, path)
        with tracking.run(ctx.settings, f"train-{name}", "train", ctx.run_id,
                          {"model": name, "n_train": len(train), **model.get_params()}):
            mlflow.log_metrics(tracking.flat_metrics("train", evaluate_by_country(train, predict_df(model, train))))
            tracking.log_artifact(path, "model")
    task.__name__ = f"train_{name}"
    return task


def evaluate_models(ctx: RunContext) -> None:
    """Score every candidate on valid + holdout (per country), pick the challenger by validation AUC."""
    valid, holdout = ctx.read("valid"), ctx.read("holdout")
    report = {}
    for name in CANDIDATES:
        model = joblib.load(ctx.dir / "models" / f"{name}.joblib")
        report[name] = {"valid": evaluate_by_country(valid, predict_df(model, valid)),
                        "holdout": evaluate_by_country(holdout, predict_df(model, holdout))}
        with tracking.run(ctx.settings, f"evaluate-{name}", "evaluate", ctx.run_id, {"model": name}):
            mlflow.log_metrics(tracking.flat_metrics("valid", report[name]["valid"]))
            mlflow.log_metrics(tracking.flat_metrics("holdout", report[name]["holdout"]))
    challenger = max(CANDIDATES, key=lambda n: report[n]["valid"]["macro"]["auc"])
    (ctx.dir / "evaluation.json").write_text(json.dumps({"challenger": challenger, "models": report}, indent=2))


def promotion_gate(ctx: RunContext) -> None:
    """Re-score the champion on the *current* holdout and promote the challenger only if it wins on AUC and ECE."""
    s = ctx.settings
    ev = json.loads((ctx.dir / "evaluation.json").read_text())
    name = ev["challenger"]
    chall = ev["models"][name]["holdout"]["macro"]
    registry = Registry(s.registry_dir)
    champion_model = registry.load_champion()
    champ = None
    if champion_model is not None:
        champ = evaluate_by_country(ctx.read("holdout"), predict_df(champion_model, ctx.read("holdout")))["macro"]
    decision = decide(chall, champ, s.gate_min_auc_gain, s.gate_min_ece_gain)
    record = {"run_id": ctx.run_id, "challenger_model": name, "promoted": decision.promote, **decision.to_dict()}
    (ctx.dir / "gate.json").write_text(json.dumps(record, indent=2))
    if decision.promote:
        registry.promote(ctx.dir / "models" / f"{name}.joblib",
                         {"run_id": ctx.run_id, "model": name, "holdout_macro": chall})
    registry.log_decision(record)
    with tracking.run(ctx.settings, "promotion_gate", "gate", ctx.run_id, {"challenger": name}):
        mlflow.set_tag("promoted", str(decision.promote).lower())
        mlflow.log_metrics({f"challenger_{k}": v for k, v in chall.items()})
        if champ:
            mlflow.log_metrics({f"champion_{k}": v for k, v in champ.items()})


def monitoring_report(ctx: RunContext) -> None:
    model = Registry(ctx.settings.registry_dir).load_champion()
    if model is None:
        raise RuntimeError("no champion in the registry; run the pipeline first")
    out = ctx.dir / "monitoring"
    report = build_report(model, ctx.read("holdout"), ctx.scenario or DEFAULT_SCENARIO, out)
    with tracking.run(ctx.settings, "monitoring", "monitor", ctx.run_id):
        for c, v in report["max_psi"].items():
            mlflow.log_metric(f"max_psi_{c}", v)
        for r in report["table"][report["table"].kind == "score"].itertuples():
            mlflow.log_metric(f"score_psi_{r.country}", r.psi)
        for f in ("report.md", "report.json", "psi_heatmap.png"):
            tracking.log_artifact(out / f, "monitoring")


PIPELINE = [
    TaskSpec("generate_data", generate_data),
    TaskSpec("train_lightgbm", _train("lightgbm"), ("generate_data",)),
    TaskSpec("train_xgboost", _train("xgboost"), ("generate_data",)),
    TaskSpec("train_mlp", _train("mlp"), ("generate_data",)),
    TaskSpec("evaluate", evaluate_models, ("train_lightgbm", "train_xgboost", "train_mlp")),
    TaskSpec("promotion_gate", promotion_gate, ("evaluate",)),
    TaskSpec("monitoring_report", monitoring_report, ("promotion_gate",)),
]
TASKS = {t.name: t for t in PIPELINE}


def run_pipeline(ctx: RunContext) -> dict[str, dict]:
    results = run_dag(PIPELINE, ctx)
    (ctx.dir / "dag_result.json").write_text(json.dumps(results, indent=2))
    return results
