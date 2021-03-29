"""Command line: ``python -m borderscore {pipeline,coldstart,monitor}``."""
from __future__ import annotations

import argparse
import warnings
import json
import sys
from dataclasses import replace

import mlflow

from . import coldstart, tracking
from .config import Settings
from .data import NEW_COUNTRY
from .tasks import RunContext, TASKS, run_pipeline


def _pipeline(args, settings) -> int:
    ctx = RunContext(settings, args.run_id or "", args.seed)
    results = run_pipeline(ctx)
    for name, r in results.items():
        print(f"{name:20s} {r['status']:16s} {r['seconds']:.1f}s")
        if r.get("error"):
            print(r["error"], file=sys.stderr)
    if all(r["status"] == "success" for r in results.values()):
        gate = json.loads((ctx.dir / "gate.json").read_text())
        print(f"\nChallenger: {gate['challenger_model']}  promoted: {gate['promoted']}")
        for line in gate["reasons"]:
            print("  -", line)
        print(f"Artifacts: {ctx.dir}")
        return 0
    return 1


def _monitor(args, settings) -> int:
    from .monitoring import DEFAULT_SCENARIO
    from .data import Shift

    ctx = RunContext(settings, args.run_id, args.seed, {c: Shift() for c in DEFAULT_SCENARIO} if args.no_drift else None)
    TASKS["monitoring_report"].fn(ctx)
    print(f"Report: {ctx.dir / 'monitoring' / 'report.md'}")
    return 0


def _coldstart(args, settings) -> int:
    sizes = [int(s) for s in args.sizes.split(",")]
    seeds = list(range(settings.seed, settings.seed + args.seeds))
    results = coldstart.run_experiment(sizes, seeds, n_source=settings.n_source)
    out = settings.home / "coldstart"
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / "results.csv", index=False)
    summ = coldstart.summarise(results)
    summ.to_csv(out / "summary.csv", index=False)
    png = coldstart.plot(results, out / "coldstart.png")
    with tracking.run(settings, "coldstart", "coldstart", params={"sizes": args.sizes, "seeds": args.seeds, "target": NEW_COUNTRY}):
        for r in summ.itertuples():
            for m in ("auc", "ks", "logloss", "ece"):
                mlflow.log_metric(f"{r.method}_{m}", getattr(r, f"{m}_mean"), step=int(r.n_local))
        for f in (out / "results.csv", out / "summary.csv", png):
            tracking.log_artifact(f)
    show = summ[summ.method.isin(["zero_shot"] + coldstart.METHODS)][["method", "n_local", "auc_mean", "ece_mean"]]
    print(show.pivot(index="n_local", columns="method", values="auc_mean").round(4).to_string())
    print(f"\nPlot: {png}")
    return 0


def main(argv=None) -> int:
    warnings.filterwarnings("ignore", message=".*eval_set.*")
    p = argparse.ArgumentParser(prog="borderscore")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("pipeline", help="generate data, train, evaluate, gate, monitor")
    a.add_argument("--run-id")
    a.add_argument("--seed", type=int)
    c = sub.add_parser("coldstart", help="new-market transfer experiment")
    c.add_argument("--sizes", default="50,100,200,500,1000,2000")
    c.add_argument("--seeds", type=int, default=3)
    m = sub.add_parser("monitor", help="monitoring report on an existing run's data using the champion")
    m.add_argument("--run-id", required=True)
    m.add_argument("--seed", type=int)
    m.add_argument("--no-drift", action="store_true", help="control scenario without drift")
    args = p.parse_args(argv)
    settings = Settings.from_env()
    return {"pipeline": _pipeline, "coldstart": _coldstart, "monitor": _monitor}[args.cmd](args, settings)


if __name__ == "__main__":
    raise SystemExit(main())
