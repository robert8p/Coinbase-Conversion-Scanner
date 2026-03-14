from __future__ import annotations
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import requests


def _to_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class CoinbaseClient:
    base_url: str = "https://api.exchange.coinbase.com"
    max_workers: int = 12

    def _get(self, path: str, params: Dict[str, str], timeout_s: int = 20) -> Tuple[Optional[object], Optional[str], Optional[str]]:
        url = self.base_url.rstrip("/") + path
        backoff = 0.6
        warn = None
        err = None
        for _ in range(6):
            try:
                r = requests.get(url, params=params, timeout=timeout_s, headers={"User-Agent": "coinbase-crypto-prob-scanner/1.0"})
                if r.status_code == 429:
                    warn = "HTTP 429 rate-limited; retrying"
                    time.sleep(backoff)
                    backoff = min(10.0, backoff * 1.8)
                    continue
                if r.status_code >= 400:
                    return None, f"HTTP {r.status_code}: {r.text[:300]}", warn
                return r.json(), None, warn
            except Exception as e:
                err = str(e)
                time.sleep(backoff)
                backoff = min(10.0, backoff * 1.8)
        return None, err or "request failed", warn

    def _fetch_symbol_bars(self, sym: str, start_utc: datetime, end_utc: datetime, granularity_s: int) -> Tuple[str, List[dict], Optional[str], Optional[str]]:
        bars: List[dict] = []
        err_any = None
        warn_any = None
        max_chunk = timedelta(seconds=granularity_s * 300)
        chunk_start = start_utc
        while chunk_start < end_utc:
            chunk_end = min(end_utc, chunk_start + max_chunk)
            params = {"start": _to_iso(chunk_start), "end": _to_iso(chunk_end), "granularity": str(granularity_s)}
            js, err, warn = self._get(f"/products/{sym}/candles", params)
            if warn:
                warn_any = warn_any or warn
            if err:
                err_any = err_any or err
                break
            if isinstance(js, list):
                for c in js:
                    if isinstance(c, list) and len(c) >= 6:
                        ts = datetime.fromtimestamp(int(c[0]), tz=timezone.utc).isoformat().replace("+00:00", "Z")
                        bars.append({"t": ts, "o": float(c[3]), "h": float(c[2]), "l": float(c[1]), "c": float(c[4]), "v": float(c[5])})
            chunk_start = chunk_end
            time.sleep(0.005)
        if bars:
            bars.sort(key=lambda x: x["t"])
        return sym, bars, err_any, warn_any

    def get_bars(self, symbols: List[str], start_utc: datetime, end_utc: datetime, granularity_s: int = 300) -> Tuple[Dict[str, List[dict]], Optional[str], Optional[str]]:
        out: Dict[str, List[dict]] = {}
        warn_any = None
        err_any = None
        if not symbols:
            return out, None, None

        workers = max(1, min(self.max_workers, len(symbols)))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(self._fetch_symbol_bars, sym, start_utc, end_utc, granularity_s) for sym in symbols]
            for fut in as_completed(futs):
                sym, bars, err, warn = fut.result()
                if warn:
                    warn_any = warn_any or warn
                if err:
                    err_any = err_any or f"{sym}: {err}"
                if bars:
                    out[sym] = bars
        return None, err, warn

    def get_bars(self, symbols: List[str], start_utc: datetime, end_utc: datetime, granularity_s: int = 300) -> Tuple[Dict[str, List[dict]], Optional[str], Optional[str]]:
        out: Dict[str, List[dict]] = {}
        warn_any = None
        err_any = None
        max_chunk = timedelta(seconds=granularity_s * 300)
        for sym in symbols:
            bars: List[dict] = []
            chunk_start = start_utc
            while chunk_start < end_utc:
                chunk_end = min(end_utc, chunk_start + max_chunk)
                params = {"start": _to_iso(chunk_start), "end": _to_iso(chunk_end), "granularity": str(granularity_s)}
                js, err, warn = self._get(f"/products/{sym}/candles", params)
                if warn:
                    warn_any = warn_any or warn
                if err:
                    err_any = err_any or err
                    break
                if isinstance(js, list):
                    for c in js:
                        if isinstance(c, list) and len(c) >= 6:
                            ts = datetime.fromtimestamp(int(c[0]), tz=timezone.utc).isoformat().replace("+00:00", "Z")
                            bars.append({"t": ts, "o": float(c[3]), "h": float(c[2]), "l": float(c[1]), "c": float(c[4]), "v": float(c[5])})
                chunk_start = chunk_end
                time.sleep(0.01)
            if bars:
                bars.sort(key=lambda x: x["t"])
                out[sym] = bars
        return out, err_any, warn_any
