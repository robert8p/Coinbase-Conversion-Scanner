from __future__ import annotations
from datetime import datetime
from typing import Dict, List
import math
import numpy as np

FEATURE_NAMES = [
    "ret_5m", "ret_15m", "ret_30m", "ret_60m", "ret_6h", "ret_24h", "ret_3d", "ret_7d",
    "atr_pct", "rv_1h", "rv_6h", "bb_width", "volume_z", "dollar_volume", "wickiness",
    "ema_fast_spread", "ema_slow_spread", "adx_proxy", "path_smoothness", "drawdown_24h",
    "dist_24h_high", "dist_7d_high", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "weekend",
    "btc_rel_1h", "eth_rel_1h",
]


def _ret(c: np.ndarray, n: int) -> float:
    if c.size <= n or c[-n - 1] <= 0:
        return 0.0
    return float(c[-1] / c[-n - 1] - 1.0)


def compute_features_from_5m(bars: List[dict], btc_bars: List[dict] | None = None, eth_bars: List[dict] | None = None) -> Dict[str, float]:
    c = np.array([float(b["c"]) for b in bars], dtype=float)
    h = np.array([float(b["h"]) for b in bars], dtype=float)
    l = np.array([float(b["l"]) for b in bars], dtype=float)
    o = np.array([float(b["o"]) for b in bars], dtype=float)
    v = np.array([float(b["v"]) for b in bars], dtype=float)
    if c.size < 220 or c[-1] <= 0:
        raise ValueError("insufficient bars")
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.r_[c[0], c[:-1]]), np.abs(l - np.r_[c[0], c[:-1]])))
    atr = float(np.mean(tr[-24:]))
    rets = np.diff(np.log(np.maximum(c, 1e-12)))
    ema_fast = float(np.mean(c[-12:]))
    ema_slow = float(np.mean(c[-48:]))
    ma = float(np.mean(c[-24:]))
    bb = float(np.std(c[-24:]))
    ts = datetime.fromisoformat(str(bars[-1]["t"]).replace("Z", "+00:00"))
    hour = ts.hour + ts.minute / 60.0
    dow = ts.weekday()
    r1h = _ret(c, 12)

    def _rel(other: List[dict] | None) -> float:
        if not other or len(other) < 13:
            return 0.0
        oc = np.array([float(x["c"]) for x in other], dtype=float)
        return r1h - _ret(oc, 12)

    return {
        "ret_5m": _ret(c, 1), "ret_15m": _ret(c, 3), "ret_30m": _ret(c, 6), "ret_60m": r1h,
        "ret_6h": _ret(c, 72), "ret_24h": _ret(c, 288), "ret_3d": _ret(c, 864), "ret_7d": _ret(c, 2016),
        "atr_pct": atr / c[-1], "rv_1h": float(np.std(rets[-12:])), "rv_6h": float(np.std(rets[-72:])),
        "bb_width": (2.0 * bb / ma) if ma > 0 else 0.0,
        "volume_z": float((v[-1] - np.mean(v[-288:])) / (np.std(v[-288:]) + 1e-9)),
        "dollar_volume": float(np.mean(c[-24:] * v[-24:])),
        "wickiness": float(np.mean((h[-24:] - l[-24:]) / np.maximum(np.abs(c[-24:] - o[-24:]), 1e-9))),
        "ema_fast_spread": (c[-1] / ema_fast - 1.0) if ema_fast > 0 else 0.0,
        "ema_slow_spread": (c[-1] / ema_slow - 1.0) if ema_slow > 0 else 0.0,
        "adx_proxy": float(np.mean(np.abs(np.diff(c[-25:]))) / (np.mean(tr[-24:]) + 1e-9)),
        "path_smoothness": float(abs(c[-1] - c[-24]) / (np.sum(np.abs(np.diff(c[-24:]))) + 1e-9)),
        "drawdown_24h": float(c[-1] / np.max(h[-288:]) - 1.0),
        "dist_24h_high": float(c[-1] / np.max(h[-288:]) - 1.0),
        "dist_7d_high": float(c[-1] / np.max(h[-2016:]) - 1.0) if c.size >= 2016 else 0.0,
        "hour_sin": math.sin(2 * math.pi * hour / 24.0), "hour_cos": math.cos(2 * math.pi * hour / 24.0),
        "dow_sin": math.sin(2 * math.pi * dow / 7.0), "dow_cos": math.cos(2 * math.pi * dow / 7.0),
        "weekend": 1.0 if dow >= 5 else 0.0,
        "btc_rel_1h": _rel(btc_bars), "eth_rel_1h": _rel(eth_bars),
    }
