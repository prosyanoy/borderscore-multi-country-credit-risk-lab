import numpy as np

from borderscore.data import Shift, generate_portfolio
from borderscore.models import fit_model
from borderscore.monitoring import DEFAULT_SCENARIO, build_report, psi, status


def test_psi_identical_is_zero_and_shift_is_large():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 20_000)
    assert psi(a, rng.normal(0, 1, 20_000)) < 0.02
    assert psi(a, rng.normal(1.0, 1, 20_000)) > 0.25
    assert psi(a, a) == 0.0


def test_psi_handles_discrete_and_out_of_range():
    ref = np.array([0, 0, 0, 1, 1, 2] * 100, float)
    assert psi(ref, ref + 50) > 0.25
    assert status(0.05) == "ok" and status(0.15) == "warn" and status(0.3) == "alert"


def test_report_flags_drifted_countries_only(tmp_path):
    sizes = {"BR": 1500, "MX": 1500, "PL": 1500, "ID": 600}
    train, valid, ref = generate_portfolio(sizes, 0), generate_portfolio(sizes, 1), generate_portfolio(sizes, 2)
    model = fit_model("lightgbm", train, valid)
    report = build_report(model, ref, DEFAULT_SCENARIO, tmp_path, n_current=2500)
    flagged = report["flagged"]
    assert "MX" in flagged and "score" in flagged["MX"]
    assert "PL" not in flagged
    # pure label shift is invisible to PSI but visible in default rate vs mean score
    br = report["summary_df"].set_index("country").loc["BR"]
    assert report["max_psi"]["BR"] < 0.1 and br.cur_default_rate > br.mean_score_cur * 1.3
    for f in ("report.md", "report.json", "psi_heatmap.png"):
        assert (tmp_path / f).exists()


def test_no_drift_scenario_is_quiet(tmp_path):
    sizes = {"BR": 1500, "MX": 1500, "PL": 1500, "ID": 600}
    train, valid, ref = generate_portfolio(sizes, 0), generate_portfolio(sizes, 1), generate_portfolio(sizes, 2)
    model = fit_model("lightgbm", train, valid)
    report = build_report(model, ref, {c: Shift() for c in sizes}, tmp_path, n_current=2500)
    assert report["flagged"] == {}
