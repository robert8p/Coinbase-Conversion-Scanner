from __future__ import annotations
import threading
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Set

import numpy as np

from .alpaca import CoinbaseClient
from .config import Settings
from .constituents import Constituent, discover_coinbase_products, load_fallback
from .features import FEATURE_NAMES, compute_features_from_5m
from .modeling import downside_risk_score_from_X, event_risk_mask_from_X, predict_probs, risk_bucket_from_X, stage1_diagnostics_from_X, try_load_bundle, uncertainty_from_X
from .state import AppState, CoverageStatus, ScoreRow, SkippedSymbol


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Scanner:
    def __init__(self, settings: Settings, state: AppState):
        self.settings = settings
        self.state = state
        self.constituents: List[Constituent] = []
        self._stop = threading.Event()
        self._thread = None

    def load_constituents(self) -> None:
        found, err = discover_coinbase_products(self.settings.coinbase_api_base, self.settings.quote_currencies)
        if not found:
            found = load_fallback()
        exclusions: Set[str] = set(self.settings.universe_exclude_symbols)
        eligible = [c for c in found if c.listing_age_days >= self.settings.min_listing_days and c.symbol not in exclusions]
        max_pool = max(1, min(self.settings.coinbase_max_products, len(eligible)))
        if self.settings.universe_mode == "top_n":
            max_pool = max(1, min(max_pool, self.settings.universe_top_n))
        self.constituents = eligible[:max_pool]
        with self.state.lock:
            self.state.constituents.source = "coinbase" if err is None else "fallback"
            self.state.constituents.warning = err
            self.state.constituents.count = len(self.constituents)
            self.state.coverage.universe_count = len(self.constituents)
            self.state.coverage.symbols_requested_count = len(self.constituents)

    def start(self) -> None:
        if self.settings.disable_scheduler:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.scan_once(_utc_now())
            except Exception as e:
                self.state.set_error(str(e))
            self._stop.wait(max(5, self.settings.scan_interval_minutes * 60))

    def _demo_scores(self) -> List[ScoreRow]:
        rows = [
            ScoreRow("BTC-USD", 64000, 0.79, "OK", "DEMO", 0.24, "LOW", "", 0.02, "Trend + participation"),
            ScoreRow("ETH-USD", 3400, 0.73, "OK", "DEMO", 0.28, "LOW", "", 0.01, "Impulse + BTC alignment"),
            ScoreRow("SOL-USD", 145, 0.67, "CAUTION", "DEMO", 0.44, "MED", "OOD-lite", 0.03, "Momentum + wider risk"),
        ]
        return rows

    def scan_once(self, now_utc: datetime) -> None:
        run_utc = now_utc.isoformat().replace("+00:00", "Z")
        if not self.constituents:
            self.load_constituents()
        universe = [c.symbol for c in self.constituents]
        cov = CoverageStatus(universe_count=len(universe), symbols_requested_count=len(universe))

        if self.settings.demo_mode:
            rows = self._demo_scores()
            cov.symbols_scored_count = len(rows)
            cov.stage1_candidate_count = len(rows)
            cov.stage2_scored_count = len(rows)
            cov.threshold_counts = {"ge_0_60": 3, "ge_0_70": 2, "ge_0_75": 1, "ge_0_80": 0}
            cov.universe_policy_summary = {"mode": self.settings.universe_mode, "min_listing_days": self.settings.min_listing_days}
            with self.state.lock:
                self.state.data_source.ok = True
                self.state.data_source.message = "DEMO_MODE"
                self.state.data_source.last_request_utc = run_utc
                self.state.data_source.last_bar_timestamp = run_utc
            self.state.set_scores(rows, run_utc)
            self.state.set_coverage(cov, [])
            return

        client = CoinbaseClient(base_url=self.settings.coinbase_api_base)
        start = now_utc - timedelta(days=8)
        with self.state.lock:
            self.state.data_source.message = f"Scanning {len(universe)} symbols..."
            self.state.data_source.last_request_utc = run_utc
            self.state.data_source.ok = False
        bars_by_sym, err, warn = client.get_bars(universe, start, now_utc, granularity_s=300)
        with self.state.lock:
            self.state.data_source.ok = len(bars_by_sym) > 0
            self.state.data_source.message = err or "OK"
            self.state.data_source.last_request_utc = run_utc
            self.state.data_source.rate_limit_warn = warn

        btc = bars_by_sym.get("BTC-USD", [])
        eth = bars_by_sym.get("ETH-USD", [])
        skipped: List[SkippedSymbol] = []
        feat_rows, symbols, prices = [], [], []
        for sym in universe:
            bars = bars_by_sym.get(sym, [])
            if not bars:
                skipped.append(SkippedSymbol(sym, "no_bars"))
                continue
            cov.symbols_returned_with_bars_count += 1
            if len(bars) < self.settings.min_bars_5m:
                skipped.append(SkippedSymbol(sym, "insufficient_bars", bars[-1]["t"]))
                continue
            cov.symbols_with_sufficient_bars_count += 1
            try:
                f = compute_features_from_5m(bars, btc_bars=btc, eth_bars=eth)
            except Exception:
                skipped.append(SkippedSymbol(sym, "feature_error", bars[-1]["t"]))
                continue
            if f["dollar_volume"] < self.settings.min_rolling_dollar_volume:
                skipped.append(SkippedSymbol(sym, "illiquid", bars[-1]["t"]))
                continue
            feat_rows.append([f[k] for k in FEATURE_NAMES])
            symbols.append(sym)
            prices.append(float(bars[-1]["c"]))
            self.state.data_source.last_bar_timestamp = bars[-1]["t"]

        if not feat_rows:
            cov.top_skip_reasons = dict(Counter([s.reason for s in skipped]).most_common(8))
            self.state.set_coverage(cov, skipped)
            self.state.set_scores([], run_utc)
            return

        X = np.asarray(feat_rows, dtype=float)
        bundle = try_load_bundle(self.settings.model_dir)
        stage1 = stage1_diagnostics_from_X(X)
        blocked = (risk_bucket_from_X(X) == "HIGH") | event_risk_mask_from_X(X) | (X[:, 14] > self.settings.max_wickiness)
        candidates_mask = (stage1 >= self.settings.stage1_min_score) & (~blocked)
        idx = np.where(candidates_mask)[0]
        if idx.size > self.settings.stage1_candidate_cap:
            idx = idx[np.argsort(stage1[idx])[::-1][:self.settings.stage1_candidate_cap]]
        cov.stage1_candidate_count = int(idx.size)
        cov.stage1_blocked_count = int(np.sum(blocked))

        probs = predict_probs(bundle, X[idx]) if idx.size else np.array([])
        downside = downside_risk_score_from_X(X[idx]) if idx.size else np.array([])
        unc_levels, unc_count = uncertainty_from_X(X[idx], getattr(bundle, "mu", {}), getattr(bundle, "sigma", {}), self.settings.uncertainty_z_thresh) if idx.size else (np.array([]), np.array([]))
        events = event_risk_mask_from_X(X[idx]) if idx.size else np.array([])

        rows = []
        for j, i in enumerate(idx):
            p = float(probs[j])
            d = float(downside[j])
            unc = str(unc_levels[j])
            reasons = []
            if events[j]:
                p = min(p, self.settings.event_prob_cap)
                cov.capped_by_event_count += 1
                reasons.append("EVENT_RISK")
            if d >= self.settings.downside_high_threshold:
                p = min(p, self.settings.downside_prob_cap_high)
                cov.capped_by_downside_count += 1
                reasons.append("DOWNSIDE_HIGH")
            elif d >= self.settings.downside_medium_threshold:
                p = min(p, self.settings.downside_prob_cap_medium)
                reasons.append("DOWNSIDE_MED")
            if unc == "HIGH":
                p = min(p, self.settings.uncertainty_prob_cap)
                cov.capped_by_uncertainty_count += 1
                reasons.append("OOD")
            risk = "BLOCKED" if blocked[i] else ("HIGH" if d > self.settings.downside_high_threshold else "CAUTION" if d > self.settings.downside_medium_threshold else "OK")
            rows.append(ScoreRow(symbols[i], prices[i], p, risk, ";".join(reasons) or "OK", d, unc, f"z_count={int(unc_count[j])}", float(X[i, 27]), f"stage1={stage1[i]:.2f}", float(probs[j])))

        rows.sort(key=lambda r: r.prob_2, reverse=True)
        cov.stage2_scored_count = len(rows)
        cov.symbols_scored_count = len(rows)
        cov.top_skip_reasons = dict(Counter([s.reason for s in skipped]).most_common(8))
        cov.threshold_counts = {
            "ge_0_60": int(sum(r.prob_2 >= 0.60 for r in rows)),
            "ge_0_70": int(sum(r.prob_2 >= 0.70 for r in rows)),
            "ge_0_75": int(sum(r.prob_2 >= 0.75 for r in rows)),
            "ge_0_80": int(sum(r.prob_2 >= 0.80 for r in rows)),
        }
        cov.guardrail_stats = {"blocked_in_universe": int(np.sum(blocked)), "event_in_candidates": int(np.sum(events)) if idx.size else 0}
        cov.universe_policy_summary = {"mode": self.settings.universe_mode, "top_n": self.settings.universe_top_n, "min_listing_days": self.settings.min_listing_days, "min_rolling_dollar_volume": self.settings.min_rolling_dollar_volume}

        self.state.set_scores(rows, run_utc)
        self.state.set_coverage(cov, skipped)
