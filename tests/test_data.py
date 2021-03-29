import numpy as np

from borderscore.data import ALL_COUNTRIES, COUNTRIES, FEATURES, Shift, generate_country, generate_portfolio


def test_default_rates_match_targets():
    for code in ALL_COUNTRIES:
        df = generate_country(code, 40_000, seed=1)
        assert abs(df.default.mean() - COUNTRIES[code].default_rate) < 0.012, code


def test_deterministic_and_seed_sensitive():
    a, b = generate_country("BR", 500, 3), generate_country("BR", 500, 3)
    assert a.equals(b)
    assert not a.equals(generate_country("BR", 500, 4))


def test_countries_differ():
    means = {c: generate_country(c, 5000, 0).log_income.mean() for c in ALL_COUNTRIES}
    assert max(means.values()) - min(means.values()) > 0.8


def test_covariate_shift_moves_features_not_labels_directly():
    base = generate_country("MX", 20_000, 5)
    shifted = generate_country("MX", 20_000, 5, Shift(covariate=2.0))
    assert shifted.log_income.mean() < base.log_income.mean() - 0.5
    assert shifted.dti.mean() > base.dti.mean() + 0.08


def test_label_shift_raises_default_rate_only():
    base = generate_country("PL", 30_000, 5)
    shifted = generate_country("PL", 30_000, 5, Shift(label=1.0))
    assert shifted.default.mean() > base.default.mean() * 1.8
    for f in FEATURES:
        assert np.allclose(base[f], shifted[f])


def test_portfolio_sizes():
    df = generate_portfolio({"BR": 100, "ID": 30}, 0)
    assert df.country.value_counts().to_dict() == {"BR": 100, "ID": 30}
