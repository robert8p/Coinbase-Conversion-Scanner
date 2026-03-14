from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List


def _bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
    except Exception:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)).strip())
    except Exception:
        return default


def _csv(name: str, default_csv: str) -> List[str]:
    raw = os.getenv(name, default_csv)
    vals = [x.strip().upper() for x in (raw or "").split(",") if x.strip()]
    return vals


@dataclass(frozen=True)
class Settings:
    model_dir: str
    demo_mode: bool
    disable_scheduler: bool
    scan_interval_minutes: int
    admin_password: str
    train_lookback_days: int
    train_max_symbols: int
    min_bars_5m: int

    coinbase_api_base: str
    coinbase_max_products: int
    quote_currencies: List[str]
    universe_mode: str
    universe_top_n: int
    min_listing_days: int
    min_rolling_dollar_volume: float
    max_wickiness: float
    min_recent_activity: float
    universe_exclude_symbols: List[str]

    target_horizon_minutes: int
    target_move_pct: float
    stage1_candidate_cap: int
    stage1_min_score: float

    uncertainty_z_thresh: float
    uncertainty_prob_cap: float
    downside_high_threshold: float
    downside_medium_threshold: float
    downside_prob_cap_high: float
    downside_prob_cap_medium: float
    event_prob_cap: float

    enet_c_values: List[float]
    enet_l1_values: List[float]

    @staticmethod
    def from_env() -> "Settings":
        return Settings(
            model_dir=os.getenv("MODEL_DIR", "./runtime/model"),
            demo_mode=_bool("DEMO_MODE", False),
            disable_scheduler=_bool("DISABLE_SCHEDULER", False),
            scan_interval_minutes=max(1, _int("SCAN_INTERVAL_MINUTES", 5)),
            admin_password=os.getenv("ADMIN_PASSWORD", ""),
            train_lookback_days=max(10, _int("TRAIN_LOOKBACK_DAYS", 60)),
            train_max_symbols=max(0, _int("TRAIN_MAX_SYMBOLS", 0)),
            min_bars_5m=max(120, _int("MIN_BARS_5M", 360)),

            coinbase_api_base=os.getenv("COINBASE_API_BASE", "https://api.exchange.coinbase.com"),
            coinbase_max_products=max(50, _int("COINBASE_MAX_PRODUCTS", 400)),
            quote_currencies=_csv("COINBASE_QUOTE_CURRENCIES", "USD,USDC"),
            universe_mode=os.getenv("UNIVERSE_MODE", "top_n").strip().lower(),
            universe_top_n=max(20, _int("UNIVERSE_TOP_N", 120)),
            min_listing_days=max(1, _int("MIN_LISTING_DAYS", 30)),
            min_rolling_dollar_volume=max(1.0, _float("MIN_ROLLING_DOLLAR_VOLUME", 750000.0)),
            max_wickiness=max(0.1, _float("MAX_WICKINESS", 0.75)),
            min_recent_activity=max(0.0, _float("MIN_RECENT_ACTIVITY", 0.65)),
            universe_exclude_symbols=_csv("UNIVERSE_EXCLUDE_SYMBOLS", ""),

            target_horizon_minutes=max(30, _int("TARGET_HORIZON_MINUTES", 120)),
            target_move_pct=max(0.005, _float("TARGET_MOVE_PCT", 0.02)),
            stage1_candidate_cap=max(10, _int("STAGE1_CANDIDATE_CAP", 90)),
            stage1_min_score=_float("STAGE1_MIN_SCORE", 1.3),

            uncertainty_z_thresh=max(1.0, _float("UNCERTAINTY_Z_THRESH", 3.2)),
            uncertainty_prob_cap=_float("UNCERTAINTY_PROB_CAP", 0.58),
            downside_high_threshold=_float("DOWNSIDE_HIGH_THRESHOLD", 0.72),
            downside_medium_threshold=_float("DOWNSIDE_MEDIUM_THRESHOLD", 0.5),
            downside_prob_cap_high=_float("DOWNSIDE_PROB_CAP_HIGH", 0.44),
            downside_prob_cap_medium=_float("DOWNSIDE_PROB_CAP_MEDIUM", 0.64),
            event_prob_cap=_float("EVENT_PROB_CAP", 0.4),

            enet_c_values=[float(x) for x in os.getenv("ENET_C_VALUES", "0.25,0.5,1.0,2.0").split(",") if x.strip()],
            enet_l1_values=[float(x) for x in os.getenv("ENET_L1_VALUES", "0.0,0.25,0.5,0.75").split(",") if x.strip()],
        )
