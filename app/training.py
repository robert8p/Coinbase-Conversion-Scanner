from __future__ import annotations
import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Dict, List

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .alpaca import CoinbaseClient
from .config import Settings
from .features import FEATURE_NAMES, compute_features_from_5m


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _precision(y: np.ndarray, p: np.ndarray, thr: float) -> tuple[float | None, int]:
    m = p >= thr
    n = int(np.sum(m))
    if n == 0:
        return None, 0
    return float(np.mean(y[m])), n


def _label_from_bars(bars: List[dict], i: int, horizon_bars: int, move_pct: float) -> int:
    p0 = float(bars[i]["c"])
    future = bars[i + 1:i + 1 + horizon_bars]
    if not future:
        return 0
    h = max(float(x["h"]) for x in future)
    return int(h >= p0 * (1.0 + move_pct))


def run_training(settings: Settings, symbols: List[str], sector_map: Dict[str, str]) -> Dict[str, object]:
    if settings.demo_mode:
        raise RuntimeError("Training requires DEMO_MODE=false.")
    client = CoinbaseClient(base_url=settings.coinbase_api_base)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=settings.train_lookback_days)
    horizon_bars = max(1, settings.target_horizon_minutes // 5)

    if settings.train_max_symbols > 0:
        symbols = symbols[:settings.train_max_symbols]

    bars_by_sym, err, _ = client.get_bars(symbols, start_utc=start, end_utc=end, granularity_s=300)
    if err and not bars_by_sym:
        raise RuntimeError(f"Coinbase data error: {err}")

    btc = bars_by_sym.get("BTC-USD", [])
    eth = bars_by_sym.get("ETH-USD", [])

    X_rows, y_rows, day_idx = [], [], []
    for sym, bars in bars_by_sym.items():
        if len(bars) < max(settings.min_bars_5m, horizon_bars + 250):
            continue
        for i in range(220, len(bars) - horizon_bars - 1):
            try:
                feats = compute_features_from_5m(bars[:i + 1], btc_bars=btc[:i + 1], eth_bars=eth[:i + 1])
            except Exception:
                continue
            X_rows.append([float(feats[f]) for f in FEATURE_NAMES])
            y_rows.append(_label_from_bars(bars, i, horizon_bars, settings.target_move_pct))
            ts = datetime.fromisoformat(str(bars[i]["t"]).replace("Z", "+00:00"))
            day_idx.append(int(ts.strftime("%Y%m%d")))

    if len(y_rows) < 1200:
        raise RuntimeError(f"Not enough training samples ({len(y_rows)}). Increase TRAIN_LOOKBACK_DAYS.")

    X = np.asarray(X_rows, dtype=float)
    y = np.asarray(y_rows, dtype=int)
    days = sorted(np.unique(day_idx))
    split = max(5, int(len(days) * 0.8))
    train_days = set(days[:split])
    hold_days = set(days[split:])
    mtr = np.array([d in train_days for d in day_idx], dtype=bool)
    mho = np.array([d in hold_days for d in day_idx], dtype=bool)

    best = None
    for c in settings.enet_c_values:
        for l1 in settings.enet_l1_values:
            pipe = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(solver="saga", penalty="elasticnet", l1_ratio=l1, C=c, max_iter=3000, class_weight="balanced"))])
            pipe.fit(X[mtr], y[mtr])
            p_tr = pipe.predict_proba(X[mtr])[:, 1]
            cal = IsotonicRegression(out_of_bounds="clip")
            cal.fit(p_tr, y[mtr])
            p_ho = cal.predict(pipe.predict_proba(X[mho])[:, 1])
            prec75, cnt75 = _precision(y[mho], p_ho, 0.75)
            score = (prec75 or 0.0) + min(cnt75, 40) / 200.0
            if best is None or score > best["score"]:
                best = {"pipeline": pipe, "cal": cal, "p": p_ho, "score": score, "c": c, "l1": l1}

    p = best["p"]
    metrics = {
        "auc_val": float(roc_auc_score(y[mho], p)) if len(np.unique(y[mho])) > 1 else None,
        "brier_val": _brier(y[mho], p),
        "precision_at": {},
    }
    for thr in (0.60, 0.70, 0.75, 0.80):
        pr, ct = _precision(y[mho], p, thr)
        metrics["precision_at"][str(thr)] = {"precision": pr, "count": ct}

    pt2_dir = os.path.join(settings.model_dir, "pt2")
    os.makedirs(pt2_dir, exist_ok=True)
    mu = {f: float(np.mean(X[mtr, i])) for i, f in enumerate(FEATURE_NAMES)}
    sigma = {f: float(np.std(X[mtr, i]) + 1e-9) for i, f in enumerate(FEATURE_NAMES)}
    bundle = SimpleNamespace(pipeline=best["pipeline"], calibrator=best["cal"], feature_names=list(FEATURE_NAMES), mu=mu, sigma=sigma, meta={**metrics, "target": "pt2", "horizon_minutes": settings.target_horizon_minutes, "move_pct": settings.target_move_pct, "model": {"C": best["c"], "l1_ratio": best["l1"]}})
    joblib.dump(bundle, os.path.join(pt2_dir, "bundle.joblib"))

    return {"pt2": metrics, "samples": int(X.shape[0]), "train_days": len(train_days), "holdout_days": len(hold_days), "trained_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")}
