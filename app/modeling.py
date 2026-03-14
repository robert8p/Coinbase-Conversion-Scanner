from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import os
import joblib
import numpy as np

from .features import FEATURE_NAMES


@dataclass
class ModelBundle:
    pipeline: object
    calibrator: object
    feature_names: List[str]
    mu: Dict[str, float]
    sigma: Dict[str, float]
    meta: Dict[str, object]


def risk_bucket_from_X(X: np.ndarray) -> np.ndarray:
    out = np.array(["OK"] * X.shape[0], dtype=object)
    downside = downside_risk_score_from_X(X)
    out[downside >= 0.65] = "HIGH"
    return out


def event_risk_mask_from_X(X: np.ndarray) -> np.ndarray:
    # volume_z, wickiness, rv_1h
    return (X[:, 12] > 2.8) | (X[:, 14] > 2.2) | (X[:, 9] > 0.05)


def downside_risk_score_from_X(X: np.ndarray) -> np.ndarray:
    dd = -np.minimum(X[:, 19], 0.0) * 2.5
    weak = -np.minimum(X[:, 16], 0.0) * 2.0
    choppy = np.clip(X[:, 14] / 3.0, 0, 1)
    return np.clip(0.5 * dd + 0.3 * weak + 0.2 * choppy, 0.0, 1.0)


def stage1_diagnostics_from_X(X: np.ndarray) -> np.ndarray:
    score = (
        1.3 * np.clip(X[:, 3], 0, 0.08) * 100
        + 0.8 * np.clip(X[:, 4], 0, 0.2) * 100
        + 0.6 * np.clip(X[:, 17], 0, 2)
        + 0.5 * np.clip(X[:, 27] + X[:, 28], -0.05, 0.1) * 100
        + 0.4 * np.clip(1.2 - X[:, 14], 0, 1)
    )
    return score


def uncertainty_from_X(X: np.ndarray, mu: Dict[str, float], sigma: Dict[str, float], z_thresh: float) -> tuple[np.ndarray, np.ndarray]:
    zcounts = np.zeros(X.shape[0], dtype=int)
    for i, f in enumerate(FEATURE_NAMES):
        m = float(mu.get(f, 0.0))
        s = max(1e-6, float(sigma.get(f, 1.0)))
        zcounts += (np.abs((X[:, i] - m) / s) > z_thresh).astype(int)
    levels = np.where(zcounts >= 6, "HIGH", np.where(zcounts >= 3, "MED", "LOW"))
    return levels, zcounts


def try_load_bundle(model_dir: str) -> Optional[ModelBundle]:
    p = os.path.join(model_dir, "pt2", "bundle.joblib")
    if not os.path.exists(p):
        return None
    b = joblib.load(p)
    if getattr(b, "feature_names", None) != list(FEATURE_NAMES):
        return None
    return b


def predict_probs(bundle: Optional[ModelBundle], X: np.ndarray) -> np.ndarray:
    if bundle is None:
        return np.clip(0.5 + 1.2 * X[:, 3] + 0.8 * X[:, 27] - 0.6 * X[:, 14], 0.02, 0.95)
    raw = bundle.pipeline.predict_proba(X)[:, 1]
    if bundle.calibrator is None:
        return raw
    return bundle.calibrator.predict(raw)
