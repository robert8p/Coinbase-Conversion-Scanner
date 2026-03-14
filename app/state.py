from __future__ import annotations
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TrainingStatus:
    running: bool = False
    started_at_utc: Optional[str] = None
    finished_at_utc: Optional[str] = None
    last_result: Optional[Dict[str, Any]] = None
    last_error: Optional[str] = None


@dataclass
class DataSourceStatus:
    ok: bool = False
    message: str = "Not checked"
    last_request_utc: Optional[str] = None
    last_bar_timestamp: Optional[str] = None
    rate_limit_warn: Optional[str] = None


@dataclass
class UniverseStatus:
    source: str = "fallback"
    warning: Optional[str] = None
    count: int = 0


@dataclass
class ModelThresholdStatus:
    trained: bool = False
    path: str = ""
    auc_val: Optional[float] = None
    brier_val: Optional[float] = None
    calibrator: Optional[str] = None


@dataclass
class ModelStatus:
    pt2: ModelThresholdStatus = field(default_factory=ModelThresholdStatus)


@dataclass
class CoverageStatus:
    universe_count: int = 0
    symbols_requested_count: int = 0
    symbols_returned_with_bars_count: int = 0
    symbols_with_sufficient_bars_count: int = 0
    symbols_scored_count: int = 0
    top_skip_reasons: Dict[str, int] = field(default_factory=dict)
    profile_symbols_available: int = 0
    profile_symbols_missing: int = 0
    profile_note: Optional[str] = None
    stage1_candidate_count: int = 0
    stage2_scored_count: int = 0
    stage1_blocked_count: int = 0
    capped_by_downside_count: int = 0
    capped_by_uncertainty_count: int = 0
    capped_by_event_count: int = 0
    threshold_counts: Dict[str, int] = field(default_factory=dict)
    guardrail_stats: Dict[str, int] = field(default_factory=dict)
    universe_policy_summary: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkippedSymbol:
    symbol: str
    reason: str
    last_bar_timestamp: Optional[str] = None


@dataclass
class ScoreRow:
    symbol: str
    price: float
    prob_2: float
    risk: str
    risk_reasons: str
    downside_risk: Optional[float] = None
    uncertainty: str = "LOW"
    uncertainty_reasons: str = ""
    btc_relative: Optional[float] = None
    reasons: str = ""
    prob_2_raw: Optional[float] = None
    target: str = "pt2"


@dataclass
class AppState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    data_source: DataSourceStatus = field(default_factory=DataSourceStatus)
    constituents: UniverseStatus = field(default_factory=UniverseStatus)
    model: ModelStatus = field(default_factory=ModelStatus)
    training: TrainingStatus = field(default_factory=TrainingStatus)
    coverage: CoverageStatus = field(default_factory=CoverageStatus)
    last_run_utc: Optional[str] = None
    scores: List[ScoreRow] = field(default_factory=list)
    last_error: Optional[str] = None
    skipped: List[SkippedSymbol] = field(default_factory=list)

    def set_scores(self, rows: List[ScoreRow], run_utc: str) -> None:
        with self.lock:
            self.scores = rows
            self.last_run_utc = run_utc
            self.last_error = None

    def set_error(self, msg: str) -> None:
        with self.lock:
            self.last_error = msg

    def set_coverage(self, cov: CoverageStatus, skipped: List[SkippedSymbol]) -> None:
        with self.lock:
            self.coverage = cov
            self.skipped = skipped[:200]

    def snapshot_scores(self) -> Dict[str, Any]:
        with self.lock:
            return {"last_run_utc": self.last_run_utc, "rows": [r.__dict__ for r in self.scores], "last_error": self.last_error}

    def snapshot_status(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "data_source": self.data_source.__dict__,
                "constituents": self.constituents.__dict__,
                "model": {"pt2": self.model.pt2.__dict__},
                "training": self.training.__dict__,
                "coverage": self.coverage.__dict__,
                "last_run_utc": self.last_run_utc,
                "last_error": self.last_error,
            }
