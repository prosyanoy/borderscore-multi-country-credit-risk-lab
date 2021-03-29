import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from borderscore.data import generate_portfolio
from borderscore.models import GBMClassifier, MLPClassifier, design_matrix, fit_model, labels, predict_df

SIZES = {"BR": 1500, "MX": 1500, "PL": 1500, "ID": 300}


@pytest.fixture(scope="module")
def frames():
    return generate_portfolio(SIZES, 0), generate_portfolio({c: 500 for c in SIZES}, 1), generate_portfolio({c: 1500 for c in SIZES}, 2)


@pytest.mark.parametrize("name", ["lightgbm", "xgboost", "mlp"])
def test_models_learn_signal(frames, name):
    train, valid, test = frames
    model = fit_model(name, train, valid, seed=0)
    p = predict_df(model, test)
    assert p.shape == (len(test),) and ((p >= 0) & (p <= 1)).all()
    assert roc_auc_score(labels(test), p) > 0.7


def test_design_matrix_country_columns(frames):
    X = design_matrix(frames[0])
    assert {"country_BR", "country_ID"} <= set(X.columns)
    assert X[[c for c in X.columns if c.startswith("country_")]].sum(axis=1).eq(1).all()
    assert not any(c.startswith("country_") for c in design_matrix(frames[0], False).columns)


@pytest.mark.parametrize("kind", ["lightgbm", "xgboost"])
def test_warm_start_adds_trees_and_keeps_original(frames, kind):
    train, valid, _ = frames
    base = GBMClassifier(kind, n_estimators=40, early_stopping=5).fit(design_matrix(train, False), labels(train))
    before = base.n_trees
    local = train[train.country == "ID"]
    warm = base.warm_start(design_matrix(local, False), labels(local), n_estimators=10)
    assert warm.n_trees == before + 10
    assert base.n_trees == before


def test_mlp_fine_tune_changes_copy_only(frames):
    train, _, test = frames
    X, y = design_matrix(train, False), labels(train)
    base = MLPClassifier(epochs=5, seed=0).fit(X, y)
    p0 = base.predict_proba(design_matrix(test, False))
    local = train[train.country == "ID"]
    ft = base.fine_tune(design_matrix(local, False), labels(local), epochs=5)
    assert np.allclose(p0, base.predict_proba(design_matrix(test, False)))
    assert not np.allclose(p0, ft.predict_proba(design_matrix(test, False)))
    assert np.allclose(base.mean_, ft.mean_)
