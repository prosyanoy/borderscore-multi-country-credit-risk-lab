"""Drift monitoring: PSI per feature and per score, per country, under a simulated drift scenario."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import ALL_COUNTRIES, FEATURES, TARGET, Shift, generate_portfolio
from .metrics import evaluate
from .models import predict_df

PSI_WARN, PSI_ALERT = 0.10, 0.25

# Simulated production month. BR: pure label shift (features and scores look stable but
# defaults rise); MX: strong covariate shift; PL: control, no drift; ID: moderate covariate shift.
DEFAULT_SCENARIO = {
    "BR": Shift(covariate=0.0, label=0.6),
    "MX": Shift(covariate=2.0, label=0.0),
    "PL": Shift(),
    "ID": Shift(covariate=0.8, label=0.0),
}


def psi(reference, current, bins: int = 10, eps: float = 1e-4) -> float:
    """Population Stability Index with reference-quantile bins (open-ended outer bins)."""
    ref, cur = np.asarray(reference, float), np.asarray(current, float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)[1:-1]))
    r = np.bincount(np.searchsorted(edges, ref, side="right"), minlength=len(edges) + 1) / len(ref)
    c = np.bincount(np.searchsorted(edges, cur, side="right"), minlength=len(edges) + 1) / len(cur)
    r, c = np.clip(r, eps, None), np.clip(c, eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def status(value: float) -> str:
    return "alert" if value >= PSI_ALERT else "warn" if value >= PSI_WARN else "ok"


def drift_table(ref: pd.DataFrame, cur: pd.DataFrame, ref_scores: np.ndarray, cur_scores: np.ndarray) -> pd.DataFrame:
    rows = []
    for c in ALL_COUNTRIES:
        mr, mc = (ref["country"] == c).to_numpy(), (cur["country"] == c).to_numpy()
        if not mr.any() or not mc.any():
            continue
        for f in FEATURES:
            rows.append({"country": c, "item": f, "kind": "feature",
                         "psi": psi(ref.loc[mr, f], cur.loc[mc, f])})
        rows.append({"country": c, "item": "score", "kind": "score",
                     "psi": psi(ref_scores[mr], cur_scores[mc])})
    df = pd.DataFrame(rows)
    df["status"] = df.psi.map(status)
    return df


def country_summary(ref, cur, ref_scores, cur_scores) -> pd.DataFrame:
    rows = []
    for c in ALL_COUNTRIES:
        mr, mc = (ref["country"] == c).to_numpy(), (cur["country"] == c).to_numpy()
        m = evaluate(cur.loc[mc, TARGET], cur_scores[mc])
        rows.append({"country": c, "ref_default_rate": ref.loc[mr, TARGET].mean(),
                     "cur_default_rate": cur.loc[mc, TARGET].mean(),
                     "mean_score_ref": ref_scores[mr].mean(), "mean_score_cur": cur_scores[mc].mean(),
                     "auc_cur": m["auc"], "ece_cur": m["ece"]})
    return pd.DataFrame(rows)


def _heatmap(table: pd.DataFrame, path: Path) -> None:
    pivot = table.pivot(index="country", columns="item", values="psi").reindex(index=ALL_COUNTRIES, columns=FEATURES + ["score"])
    fig, ax = plt.subplots(figsize=(10, 3.4))
    im = ax.imshow(pivot.values, cmap="YlOrRd", vmin=0, vmax=0.5, aspect="auto")
    ax.set_xticks(range(pivot.shape[1]), pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(pivot.shape[0]), pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if v > 0.3 else "black")
    fig.colorbar(im, ax=ax, label="PSI (warn 0.10, alert 0.25)")
    ax.set_title("Population stability by country (reference = holdout, current = simulated production batch)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _md_table(df: pd.DataFrame, fmt: str = "{:.3f}") -> str:
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    body = "".join("| " + " | ".join(fmt.format(v) if isinstance(v, float) else str(v) for v in r) + " |\n"
                   for r in df.itertuples(index=False))
    return head + body


def build_report(model, reference: pd.DataFrame, scenario: dict[str, Shift], out_dir: Path,
                 n_current: int | dict[str, int] = 3000, seed: int = 2024) -> dict:
    """Score a drifted batch with ``model`` and write report.md / report.json / psi_heatmap.png."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = n_current if isinstance(n_current, dict) else {c: n_current for c in ALL_COUNTRIES}
    current = generate_portfolio(sizes, seed, scenario)
    ref_scores, cur_scores = predict_df(model, reference), predict_df(model, current)
    table = drift_table(reference, current, ref_scores, cur_scores)
    summary = country_summary(reference, current, ref_scores, cur_scores)
    _heatmap(table, out_dir / "psi_heatmap.png")

    flagged = table[table.status != "ok"]
    alerts = {c: sorted(flagged[flagged.country == c].item.tolist()) for c in ALL_COUNTRIES}
    report = {
        "scenario": {c: vars(s) for c, s in scenario.items()},
        "psi": table.to_dict("records"),
        "summary": summary.to_dict("records"),
        "flagged": {c: v for c, v in alerts.items() if v},
        "max_psi": {c: float(table[table.country == c].psi.max()) for c in ALL_COUNTRIES},
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    lines = ["# BorderScore monitoring report\n",
             "Reference: fixed holdout. Current: simulated production batch. Thresholds: PSI >= 0.10 warn, >= 0.25 alert.\n",
             "## Scenario\n", _md_table(pd.DataFrame([{"country": c, "covariate_shift": s.covariate, "label_shift": s.label}
                                                       for c, s in scenario.items()])),
             "\n## Country summary (labels assumed matured)\n", _md_table(summary),
             "\n## Score drift\n", _md_table(table[table.kind == "score"][["country", "psi", "status"]]),
             "\n## Feature drift (flagged only)\n"]
    feat = table[(table.kind == "feature") & (table.status != "ok")].sort_values(["country", "psi"], ascending=[True, False])
    lines.append(_md_table(feat[["country", "item", "psi", "status"]]) if len(feat) else "No feature drift flagged.\n")
    lines.append("\n![PSI heatmap](psi_heatmap.png)\n")
    (out_dir / "report.md").write_text("\n".join(lines))
    report["table"], report["summary_df"] = table, summary
    return report
