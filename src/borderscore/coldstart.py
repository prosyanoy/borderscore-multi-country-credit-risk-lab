"""Cold-start experiment: how to model a new country with little local data.

Strategies compared as a function of the number of local labelled samples:
  * zero_shot        - source-country model applied as-is (reference line)
  * scratch_lgbm     - LightGBM trained on local samples only
  * pooled_lgbm      - LightGBM on source + local data with a country feature
  * warmstart_lgbm   - source LightGBM, boosting continued on local samples
  * scratch_mlp      - MLP trained on local samples only
  * finetune_mlp     - source-pretrained MLP fine-tuned on local samples
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import HOLDOUT_SEED, TARGET, NEW_COUNTRY, SOURCE_COUNTRIES, generate_country, generate_portfolio
from .metrics import evaluate
from .models import GBMClassifier, MLPClassifier, design_matrix, labels

METHODS = ["scratch_lgbm", "pooled_lgbm", "warmstart_lgbm", "scratch_mlp", "finetune_mlp"]
COLORS = {"scratch_lgbm": "#0072B2", "pooled_lgbm": "#E69F00", "warmstart_lgbm": "#009E73",
          "scratch_mlp": "#CC79A7", "finetune_mlp": "#D55E00", "zero_shot": "#555555"}


def _row(method, n, seed, y, p):
    return {"method": method, "n_local": n, "seed": seed, **evaluate(y, p)}


def run_seed(seed: int, sizes: list[int], n_source: int, n_test: int, target: str = NEW_COUNTRY) -> list[dict]:
    src_sizes = {c: n_source for c in SOURCE_COUNTRIES}
    src = generate_portfolio(src_sizes, seed)
    src_val = generate_portfolio({c: max(300, n_source // 3) for c in SOURCE_COUNTRIES}, seed + 10_000)
    pool = generate_country(target, max(sizes), seed + 20_000)
    test = generate_country(target, n_test, HOLDOUT_SEED)
    yt, Xt_plain, Xt_c = labels(test), design_matrix(test, False), design_matrix(test, True)

    # Source models are trained once per seed (no country indicator: target is unseen).
    Xs, ys = design_matrix(src, False), labels(src)
    Xv, yv = design_matrix(src_val, False), labels(src_val)
    src_gbm = GBMClassifier("lightgbm", seed=seed).fit(Xs, ys, Xv, yv)
    src_mlp = MLPClassifier(seed=seed).fit(Xs, ys, Xv, yv)
    Xs_c, Xv_c = design_matrix(src, True), design_matrix(src_val, True)

    rows = [_row("zero_shot", 0, seed, yt, src_gbm.predict_proba(Xt_plain))]
    for n in sizes:
        loc = pool.iloc[:n]  # nested subsets -> smoother curves
        if loc[TARGET].nunique() < 2:
            continue
        Xl, yl = design_matrix(loc, False), labels(loc)
        Xl_c = design_matrix(loc, True)

        scratch = GBMClassifier("lightgbm", seed=seed, n_estimators=150, num_leaves=7).fit(Xl, yl)
        rows.append(_row("scratch_lgbm", n, seed, yt, scratch.predict_proba(Xt_plain)))

        pooled = GBMClassifier("lightgbm", seed=seed, n_estimators=300).fit(
            pd.concat([Xs_c, Xl_c], ignore_index=True), np.concatenate([ys, yl]))
        rows.append(_row("pooled_lgbm", n, seed, yt, pooled.predict_proba(Xt_c)))

        warm = src_gbm.warm_start(Xl, yl, n_estimators=60, learning_rate=0.03)
        rows.append(_row("warmstart_lgbm", n, seed, yt, warm.predict_proba(Xt_plain)))

        mlp = MLPClassifier(hidden=(32, 16), seed=seed, epochs=80).fit(Xl, yl)
        rows.append(_row("scratch_mlp", n, seed, yt, mlp.predict_proba(Xt_plain)))

        ft = src_mlp.fine_tune(Xl, yl, epochs=30, lr=5e-4)
        rows.append(_row("finetune_mlp", n, seed, yt, ft.predict_proba(Xt_plain)))
    return rows


def run_experiment(sizes: list[int], seeds: list[int], n_source: int = 6000, n_test: int = 5000) -> pd.DataFrame:
    rows: list[dict] = []
    for s in seeds:
        rows += run_seed(s, sizes, n_source, n_test)
    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    g = results.groupby(["method", "n_local"])[["auc", "ks", "logloss", "ece"]].agg(["mean", "std"])
    g.columns = [f"{m}_{s}" for m, s in g.columns]
    return g.reset_index()


def plot(results: pd.DataFrame, path: Path) -> Path:
    summ = summarise(results)
    zero = summ[summ.method == "zero_shot"].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, metric, label in zip(axes, ["auc", "ece"], ["AUC (higher is better)", "ECE (lower is better)"]):
        for m in METHODS:
            d = summ[summ.method == m].sort_values("n_local")
            if d.empty:
                continue
            ax.plot(d.n_local, d[f"{metric}_mean"], marker="o", ms=4, lw=1.8, color=COLORS[m], label=m)
            ax.fill_between(d.n_local, d[f"{metric}_mean"] - d[f"{metric}_std"].fillna(0),
                            d[f"{metric}_mean"] + d[f"{metric}_std"].fillna(0), color=COLORS[m], alpha=0.12)
        ax.axhline(zero[f"{metric}_mean"], color=COLORS["zero_shot"], ls="--", lw=1.2, label="zero_shot (source model)")
        ax.set_xscale("log")
        ax.set_xlabel("local training samples in the new market")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"Cold start in {NEW_COUNTRY}: transfer strategies vs. local sample size (mean ± sd over seeds)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
