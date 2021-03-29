import pytest

from borderscore.dag import TaskSpec, run_dag, topological_order
from borderscore.registry import Registry, decide

CH = {"auc": 0.82, "ece": 0.010}
CP = {"auc": 0.80, "ece": 0.020}


def test_gate_promotes_when_better_on_both():
    assert decide(CH, CP).promote


def test_gate_blocks_if_either_metric_fails():
    assert not decide({"auc": 0.82, "ece": 0.03}, CP).promote
    assert not decide({"auc": 0.79, "ece": 0.01}, CP).promote


def test_gate_requires_strict_improvement_and_margins():
    assert not decide(CP, CP).promote
    assert not decide(CH, CP, min_auc_gain=0.05).promote
    assert not decide(CH, CP, min_ece_gain=0.02).promote


def test_gate_bootstraps_without_champion():
    d = decide(CH, None)
    assert d.promote and "bootstrap" in d.reasons[0]


def test_registry_roundtrip(tmp_path):
    reg = Registry(tmp_path / "reg")
    assert reg.meta() is None and reg.load_champion() is None
    import joblib
    f = tmp_path / "m.joblib"
    joblib.dump({"x": 1}, f)
    reg.promote(f, {"model": "lightgbm"})
    assert reg.meta()["model"] == "lightgbm" and reg.load_champion() == {"x": 1}


def test_topological_order_and_cycle_detection():
    t = [TaskSpec("c", print, ("b",)), TaskSpec("b", print, ("a",)), TaskSpec("a", print)]
    assert [x.name for x in topological_order(t)] == ["a", "b", "c"]
    with pytest.raises(ValueError, match="cycle"):
        topological_order([TaskSpec("a", print, ("b",)), TaskSpec("b", print, ("a",))])
    with pytest.raises(ValueError, match="unknown"):
        topological_order([TaskSpec("a", print, ("zzz",))])


def test_failure_propagates_downstream_only():
    calls = []

    def boom(ctx):
        raise RuntimeError("x")

    tasks = [TaskSpec("a", lambda c: calls.append("a")), TaskSpec("b", boom, ("a",)),
             TaskSpec("c", lambda c: calls.append("c"), ("b",)), TaskSpec("d", lambda c: calls.append("d"), ("a",))]
    res = run_dag(tasks, None)
    assert [res[k]["status"] for k in "abcd"] == ["success", "failed", "upstream_failed", "success"]
    assert calls == ["a", "d"]
