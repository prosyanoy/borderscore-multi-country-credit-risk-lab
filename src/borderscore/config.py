"""Runtime settings, read from environment variables with safe local defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    home: Path = Path("artifacts")
    mlflow_uri: str = ""
    experiment: str = "borderscore"
    seed: int = 42
    n_source: int = 6000      # training rows per established country
    n_new: int = 400          # training rows for the new market
    n_holdout: int = 4000     # fixed holdout rows per country (gate + monitoring reference)
    gate_min_auc_gain: float = 0.0
    gate_min_ece_gain: float = 0.0

    @classmethod
    def from_env(cls) -> "Settings":
        home = Path(_env("BORDERSCORE_HOME", "artifacts")).resolve()
        return cls(
            home=home,
            mlflow_uri=_env("MLFLOW_TRACKING_URI", f"sqlite:///{home / 'mlflow.db'}"),
            experiment=_env("MLFLOW_EXPERIMENT_NAME", "borderscore"),
            seed=int(_env("BORDERSCORE_SEED", "42")),
            n_source=int(_env("BORDERSCORE_N_SOURCE", "6000")),
            n_new=int(_env("BORDERSCORE_N_NEW", "400")),
            n_holdout=int(_env("BORDERSCORE_N_HOLDOUT", "4000")),
            gate_min_auc_gain=float(_env("BORDERSCORE_GATE_MIN_AUC_GAIN", "0.0")),
            gate_min_ece_gain=float(_env("BORDERSCORE_GATE_MIN_ECE_GAIN", "0.0")),
        )

    @property
    def registry_dir(self) -> Path:
        return self.home / "registry"

    @property
    def runs_dir(self) -> Path:
        return self.home / "runs"

    def train_sizes(self) -> dict[str, int]:
        from .data import COUNTRIES

        return {c: (self.n_new if spec.role == "new" else self.n_source) for c, spec in COUNTRIES.items()}
