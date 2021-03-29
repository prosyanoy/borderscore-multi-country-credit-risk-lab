"""Synthetic multi-country loan data with controllable covariate and label shift.

Each country has its own feature distributions, its own (slightly different) true
risk function, and a base intercept calibrated so that the unshifted portfolio hits
the country's target default rate. Shifts are applied *after* calibration, so a label
shift really changes the default rate and a covariate shift moves the features.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

FEATURES = [
    "age", "log_income", "loan_to_income", "dti", "credit_history_years",
    "utilization", "delinq_2y", "employment_years",
]
TARGET = "default"

# Fixed reference scaling so that country differences are visible in the risk function.
_REF = {  # feature: (mean, sd)
    "age": (40.0, 11.0), "log_income": (9.8, 0.8), "loan_to_income": (0.6, 0.5),
    "dti": (0.33, 0.12), "credit_history_years": (8.0, 5.0), "utilization": (0.4, 0.22),
    "delinq_2y": (0.5, 0.9), "employment_years": (6.0, 5.0),
}
_BASE_W = {
    "age": -0.20, "log_income": -0.50, "loan_to_income": 0.40, "dti": 0.50,
    "credit_history_years": -0.30, "utilization": 0.55, "delinq_2y": 0.60,
    "employment_years": -0.25,
}


@dataclass(frozen=True)
class CountrySpec:
    code: str
    name: str
    role: str                 # "source" (lots of data) or "new" (cold-start market)
    default_rate: float       # target rate for the unshifted population
    age_mu: float
    income_mu: float
    income_sd: float
    lti_mu: float
    dti_mu: float
    hist_mu: float
    util_mu: float
    delinq_lam: float
    emp_mu: float
    weight_scale: tuple[tuple[str, float], ...] = ()


COUNTRIES: dict[str, CountrySpec] = {
    "BR": CountrySpec("BR", "Brava", "source", 0.08, 38, 9.4, 0.75, 0.50, 0.34, 7.0, 0.45, 0.60, 5.5,
                      (("utilization", 1.2),)),
    "MX": CountrySpec("MX", "Mexara", "source", 0.11, 36, 9.1, 0.80, 0.55, 0.38, 6.0, 0.50, 0.70, 5.0,
                      (("dti", 1.15), ("log_income", 0.8))),
    "PL": CountrySpec("PL", "Polaris", "source", 0.04, 42, 10.1, 0.65, 0.65, 0.30, 11.0, 0.30, 0.30, 8.0,
                      (("delinq_2y", 1.2),)),
    "ID": CountrySpec("ID", "Indara (new market)", "new", 0.14, 33, 8.8, 0.85, 0.70, 0.36, 4.0, 0.50, 0.60, 4.0,
                      (("log_income", 1.3), ("credit_history_years", 0.8), ("utilization", 0.9))),
}
ALL_COUNTRIES = list(COUNTRIES)
SOURCE_COUNTRIES = [c for c, s in COUNTRIES.items() if s.role == "source"]
NEW_COUNTRY = next(c for c, s in COUNTRIES.items() if s.role == "new")
HOLDOUT_SEED = 777_001  # fixed so every challenger/champion is compared on identical data


@dataclass(frozen=True)
class Shift:
    """covariate: strength of feature drift (0 = none, 1 ~ a clear shift).
    label: additive change of the default log-odds (concept/prior shift)."""
    covariate: float = 0.0
    label: float = 0.0


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _sample_features(spec: CountrySpec, n: int, rng: np.random.Generator, s: float) -> pd.DataFrame:
    age = np.clip(rng.normal(spec.age_mu, 11, n), 18, 75)
    log_income = rng.normal(spec.income_mu - 0.35 * s, spec.income_sd, n) + 0.01 * (age - 40)
    lti = np.clip(np.exp(rng.normal(np.log(spec.lti_mu) + 0.15 * s, 0.5, n)), 0.02, 5.0)
    dti = np.clip(rng.normal(spec.dti_mu + 0.06 * s, 0.10, n) + 0.03 * (lti - spec.lti_mu), 0.01, 0.95)
    hist = np.minimum(rng.gamma(3.0, spec.hist_mu / 3.0, n), age - 18)
    mean_util = min(0.9, spec.util_mu + 0.08 * s)
    util = np.clip(rng.beta(2.0, 2.0 * (1 - mean_util) / mean_util, n), 0, 1)
    lam = spec.delinq_lam * (1 + 0.4 * s) * np.where(util > 0.7, 1.5, 1.0)
    delinq = rng.poisson(lam).astype(float)
    emp = np.minimum(rng.gamma(2.0, spec.emp_mu / 2.0, n), age - 18)
    return pd.DataFrame({
        "age": age, "log_income": log_income, "loan_to_income": lti, "dti": dti,
        "credit_history_years": hist, "utilization": util, "delinq_2y": delinq,
        "employment_years": emp,
    })


def _raw_logit(spec: CountrySpec, df: pd.DataFrame) -> np.ndarray:
    scale = dict(spec.weight_scale)
    z = {f: (df[f].to_numpy() - _REF[f][0]) / _REF[f][1] for f in FEATURES}
    logit = sum(_BASE_W[f] * scale.get(f, 1.0) * z[f] for f in FEATURES)
    # non-linear terms so tree/NN models have something to learn beyond a logistic fit
    logit = logit + 0.30 * z["utilization"] * z["dti"] + 0.15 * (z["age"] ** 2 - 1) + 0.25 * (df["delinq_2y"].to_numpy() >= 3)
    return logit


@lru_cache(maxsize=None)
def _intercept(code: str) -> float:
    """Bisection for the intercept that yields the target default rate (unshifted)."""
    spec = COUNTRIES[code]
    rng = np.random.default_rng(12345)
    raw = _raw_logit(spec, _sample_features(spec, 40_000, rng, 0.0))
    lo, hi = -15.0, 5.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if _sigmoid(raw + mid).mean() > spec.default_rate:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def generate_country(code: str, n: int, seed: int, shift: Shift = Shift()) -> pd.DataFrame:
    spec = COUNTRIES[code]
    idx = ALL_COUNTRIES.index(code)
    rng = np.random.default_rng([seed, idx])
    df = _sample_features(spec, n, rng, shift.covariate)
    p = _sigmoid(_raw_logit(spec, df) + _intercept(code) + shift.label)
    df[TARGET] = (rng.random(n) < p).astype(int)
    df.insert(0, "country", code)
    return df


def generate_portfolio(sizes: dict[str, int], seed: int, shifts: dict[str, Shift] | None = None) -> pd.DataFrame:
    shifts = shifts or {}
    parts = [generate_country(c, n, seed, shifts.get(c, Shift())) for c, n in sizes.items() if n > 0]
    return pd.concat(parts, ignore_index=True)


def generate_splits(train_sizes: dict[str, int], seed: int, n_holdout: int) -> dict[str, pd.DataFrame]:
    """train / valid (for early stopping and model selection) / holdout (fixed seed)."""
    valid_sizes = {c: max(200, n // 3) for c, n in train_sizes.items()}
    return {
        "train": generate_portfolio(train_sizes, seed),
        "valid": generate_portfolio(valid_sizes, seed + 10_000),
        "holdout": generate_portfolio({c: n_holdout for c in train_sizes}, HOLDOUT_SEED),
    }
