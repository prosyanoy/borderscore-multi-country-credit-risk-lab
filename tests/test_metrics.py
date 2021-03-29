import numpy as np
import pandas as pd
import pytest

from borderscore.metrics import evaluate, evaluate_by_country, expected_calibration_error, ks_statistic


def test_perfect_separation():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.1, 0.2, 0.8, 0.9, 0.9])
    assert evaluate(y, p)["auc"] == 1.0
    assert ks_statistic(y, p) == 1.0


def test_ece_zero_when_calibrated_and_large_when_not():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 50_000)
    y = (rng.random(50_000) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.01
    assert expected_calibration_error(y, np.clip(p * 0.3, 0, 1)) > 0.2


def test_single_class_raises():
    with pytest.raises(ValueError):
        evaluate([0, 0, 0], [0.1, 0.2, 0.3])


def test_by_country_has_macro():
    df = pd.DataFrame({"country": ["A"] * 6 + ["B"] * 6, "default": [0, 1] * 6})
    p = np.array([0.2, 0.8] * 3 + [0.8, 0.2] * 3)
    out = evaluate_by_country(df, p)
    assert out["A"]["auc"] == 1.0 and out["B"]["auc"] == 0.0
    assert out["macro"]["auc"] == pytest.approx(0.5)
