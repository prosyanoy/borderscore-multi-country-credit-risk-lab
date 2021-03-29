"""Model wrappers: LightGBM / XGBoost (with warm start) and a PyTorch tabular MLP (with fine-tuning)."""
from __future__ import annotations

import copy

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from torch import nn

from .data import ALL_COUNTRIES, FEATURES, TARGET


def design_matrix(df: pd.DataFrame, with_country: bool = True) -> pd.DataFrame:
    """Feature matrix; optionally appends one-hot country indicators (fixed column set)."""
    X = df[FEATURES].astype(float).reset_index(drop=True)
    if with_country:
        for c in ALL_COUNTRIES:
            X[f"country_{c}"] = (df["country"].to_numpy() == c).astype(float)
    return X


def labels(df: pd.DataFrame) -> np.ndarray:
    return df[TARGET].to_numpy().astype(int)


class GBMClassifier:
    """Gradient boosted trees with a uniform interface for LightGBM and XGBoost."""

    def __init__(self, kind: str = "lightgbm", seed: int = 0, n_estimators: int = 400,
                 learning_rate: float = 0.05, num_leaves: int = 15, early_stopping: int = 30):
        if kind not in ("lightgbm", "xgboost"):
            raise ValueError(f"unknown GBM kind: {kind}")
        self.kind, self.seed = kind, seed
        self.n_estimators, self.learning_rate = n_estimators, learning_rate
        self.num_leaves, self.early_stopping = num_leaves, early_stopping
        self.model_ = None

    def get_params(self) -> dict:
        return {"kind": self.kind, "n_estimators": self.n_estimators, "learning_rate": self.learning_rate,
                "num_leaves": self.num_leaves, "early_stopping": self.early_stopping, "seed": self.seed}

    def _make(self, n_rows: int, n_estimators: int, learning_rate: float, early_stopping: bool):
        if self.kind == "lightgbm":
            return lgb.LGBMClassifier(
                n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=self.num_leaves,
                min_child_samples=min(20, max(3, n_rows // 10)), subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, reg_lambda=1.0, random_state=self.seed, n_jobs=1, verbose=-1)
        return xgb.XGBClassifier(
            n_estimators=n_estimators, learning_rate=learning_rate,
            max_depth=max(2, int(np.log2(self.num_leaves))), subsample=0.8, colsample_bytree=0.8,
            min_child_weight=1, reg_lambda=1.0, tree_method="hist", n_jobs=1, random_state=self.seed,
            eval_metric="logloss", early_stopping_rounds=self.early_stopping if early_stopping else None)

    def _fit(self, model, X, y, X_val, y_val, sample_weight, init):
        use_val = X_val is not None and y_val is not None
        if self.kind == "lightgbm":
            kw = {"init_model": init} if init is not None else {}
            if use_val:
                kw.update(eval_set=[(X_val, y_val)], eval_metric="binary_logloss",
                          callbacks=[lgb.early_stopping(self.early_stopping, verbose=False)])
            model.fit(X, y, sample_weight=sample_weight, **kw)
        else:
            kw = {"xgb_model": init} if init is not None else {}
            kw["eval_set"] = [(X_val, y_val)] if use_val else None
            model.fit(X, y, sample_weight=sample_weight, verbose=False, **kw)
        return model

    def fit(self, X, y, X_val=None, y_val=None, sample_weight=None) -> "GBMClassifier":
        use_val = X_val is not None
        model = self._make(len(X), self.n_estimators, self.learning_rate, use_val)
        self.model_ = self._fit(model, X, y, X_val, y_val, sample_weight, None)
        return self

    def warm_start(self, X, y, n_estimators: int = 60, learning_rate: float = 0.03,
                   sample_weight=None) -> "GBMClassifier":
        """Return a new model that continues boosting this one on (local) data."""
        new = copy.copy(self)
        init = self.model_.booster_ if self.kind == "lightgbm" else self.model_.get_booster()
        model = self._make(len(X), n_estimators, learning_rate, False)
        new.model_ = self._fit(model, X, y, None, None, sample_weight, init)
        return new

    @property
    def n_trees(self) -> int:
        if self.kind == "lightgbm":
            return int(self.model_.booster_.num_trees())
        return int(len(self.model_.get_booster().get_dump()))

    def predict_proba(self, X) -> np.ndarray:
        return self.model_.predict_proba(X)[:, 1]


class MLPClassifier:
    """Small tabular MLP (standardised inputs, Adam, early stopping, optional fine-tuning)."""

    def __init__(self, hidden=(64, 32), dropout: float = 0.1, lr: float = 2e-3, weight_decay: float = 1e-4,
                 epochs: int = 60, batch_size: int = 256, patience: int = 8, min_steps: int = 200, seed: int = 0):
        self.hidden, self.dropout, self.lr, self.weight_decay = tuple(hidden), dropout, lr, weight_decay
        self.epochs, self.batch_size, self.patience, self.min_steps, self.seed = epochs, batch_size, patience, min_steps, seed
        self.net_: nn.Module | None = None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def get_params(self) -> dict:
        return {"hidden": list(self.hidden), "dropout": self.dropout, "lr": self.lr,
                "weight_decay": self.weight_decay, "epochs": self.epochs, "batch_size": self.batch_size,
                "patience": self.patience, "seed": self.seed}

    def _tensor(self, X) -> torch.Tensor:
        return torch.tensor((np.asarray(X, dtype=np.float32) - self.mean_) / self.std_, dtype=torch.float32)

    def _build(self, d_in: int) -> nn.Module:
        layers, d = [], d_in
        for h in self.hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(self.dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        return nn.Sequential(*layers)

    def _train(self, X, y, X_val, y_val, sample_weight, epochs, lr):
        torch.manual_seed(self.seed)
        gen = torch.Generator().manual_seed(self.seed)
        Xt, yt = self._tensor(X), torch.tensor(np.asarray(y), dtype=torch.float32)
        wt = torch.ones(len(yt)) if sample_weight is None else torch.tensor(np.asarray(sample_weight), dtype=torch.float32)
        n = len(yt)
        bs = min(self.batch_size, n)
        steps_per_epoch = int(np.ceil(n / bs))
        epochs = max(epochs, int(np.ceil(self.min_steps / steps_per_epoch)))
        opt = torch.optim.Adam(self.net_.parameters(), lr=lr, weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        use_val = X_val is not None
        if use_val:
            Xv, yv = self._tensor(X_val), torch.tensor(np.asarray(y_val), dtype=torch.float32)
        best, best_state, stale = np.inf, None, 0
        for _ in range(epochs):
            self.net_.train()
            perm = torch.randperm(n, generator=gen)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                opt.zero_grad()
                loss = (loss_fn(self.net_(Xt[idx]).squeeze(1), yt[idx]) * wt[idx]).sum() / wt[idx].sum()
                loss.backward()
                opt.step()
            if use_val:
                self.net_.eval()
                with torch.no_grad():
                    v = nn.functional.binary_cross_entropy_with_logits(self.net_(Xv).squeeze(1), yv).item()
                if v < best - 1e-5:
                    best, best_state, stale = v, copy.deepcopy(self.net_.state_dict()), 0
                else:
                    stale += 1
                    if stale >= self.patience:
                        break
        if best_state is not None:
            self.net_.load_state_dict(best_state)
        self.net_.eval()

    def fit(self, X, y, X_val=None, y_val=None, sample_weight=None) -> "MLPClassifier":
        X = np.asarray(X, dtype=np.float32)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-6
        torch.manual_seed(self.seed)
        self.net_ = self._build(X.shape[1])
        self._train(X, y, X_val, y_val, sample_weight, self.epochs, self.lr)
        return self

    def fine_tune(self, X, y, epochs: int = 30, lr: float = 5e-4, sample_weight=None) -> "MLPClassifier":
        """Return a copy continued on local data with a small learning rate; the source scaler is kept."""
        new = copy.deepcopy(self)
        new._train(np.asarray(X, dtype=np.float32), y, None, None, sample_weight, epochs, lr)
        return new

    def predict_proba(self, X) -> np.ndarray:
        self.net_.eval()
        with torch.no_grad():
            return torch.sigmoid(self.net_(self._tensor(X)).squeeze(1)).numpy().astype(float)


def make_model(name: str, seed: int = 0):
    if name == "lightgbm":
        return GBMClassifier("lightgbm", seed=seed)
    if name == "xgboost":
        return GBMClassifier("xgboost", seed=seed)
    if name == "mlp":
        return MLPClassifier(seed=seed)
    raise ValueError(f"unknown model: {name}")


def fit_model(name: str, train: pd.DataFrame, valid: pd.DataFrame, seed: int = 0):
    """Pooled model with country indicators, early-stopped on the validation split."""
    model = make_model(name, seed)
    model.fit(design_matrix(train), labels(train), design_matrix(valid), labels(valid))
    return model


def predict_df(model, df: pd.DataFrame, with_country: bool = True) -> np.ndarray:
    return model.predict_proba(design_matrix(df, with_country))
