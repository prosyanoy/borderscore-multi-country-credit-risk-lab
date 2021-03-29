"""File-based champion registry and the promotion gate."""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib


@dataclass
class GateDecision:
    promote: bool
    reasons: list[str] = field(default_factory=list)
    challenger: dict = field(default_factory=dict)
    champion: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def decide(challenger: dict[str, float], champion: dict[str, float] | None,
           min_auc_gain: float = 0.0, min_ece_gain: float = 0.0) -> GateDecision:
    """Promote only if the challenger has a strictly higher AUC *and* strictly lower ECE
    (each by more than the configured margin). With no champion, the challenger is promoted."""
    if champion is None:
        return GateDecision(True, ["no champion yet: bootstrap promotion"], challenger)
    auc_ok = challenger["auc"] > champion["auc"] + min_auc_gain
    ece_ok = challenger["ece"] < champion["ece"] - min_ece_gain
    reasons = [
        f"AUC {challenger['auc']:.4f} vs champion {champion['auc']:.4f} (need > +{min_auc_gain}): {'ok' if auc_ok else 'FAIL'}",
        f"ECE {challenger['ece']:.4f} vs champion {champion['ece']:.4f} (need < -{min_ece_gain}): {'ok' if ece_ok else 'FAIL'}",
    ]
    return GateDecision(auc_ok and ece_ok, reasons, challenger, champion)


class Registry:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.champion_dir = self.root / "champion"

    @property
    def model_path(self) -> Path:
        return self.champion_dir / "model.joblib"

    def meta(self) -> dict | None:
        p = self.champion_dir / "meta.json"
        return json.loads(p.read_text()) if p.exists() and self.model_path.exists() else None

    def load_champion(self):
        return joblib.load(self.model_path) if self.meta() else None

    def promote(self, model_file: Path, meta: dict) -> None:
        self.champion_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(model_file, self.model_path)
        meta = {**meta, "promoted_at": datetime.now(timezone.utc).isoformat()}
        (self.champion_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    def log_decision(self, record: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self.root / "history.jsonl", "a") as f:
            f.write(json.dumps({**record, "at": datetime.now(timezone.utc).isoformat()}) + "\n")
