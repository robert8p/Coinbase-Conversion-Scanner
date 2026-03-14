from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import requests


@dataclass
class Constituent:
    symbol: str
    name: str
    sector: str
    industry: str
    listing_age_days: int = 9999


def _lev_or_synth(base: str) -> bool:
    x = base.upper()
    bad = ["UP", "DOWN", "BULL", "BEAR", "3L", "3S", "5L", "5S", "LEV", "INDEX"]
    return any(t in x for t in bad)


def discover_coinbase_products(base_url: str, quote_currencies: List[str], timeout_s: int = 15) -> Tuple[Optional[List[Constituent]], Optional[str]]:
    url = base_url.rstrip("/") + "/products"
    try:
        r = requests.get(url, timeout=timeout_s, headers={"User-Agent": "coinbase-crypto-prob-scanner/1.0"})
        r.raise_for_status()
        rows = r.json()
        out: List[Constituent] = []
        now = datetime.now(timezone.utc)
        for p in rows:
            if not isinstance(p, dict):
                continue
            quote = str(p.get("quote_currency", "")).upper()
            if quote not in {q.upper() for q in quote_currencies}:
                continue
            if not p.get("id") or not p.get("base_currency"):
                continue
            if str(p.get("status", "")).lower() not in {"online", "active"}:
                continue
            if bool(p.get("trading_disabled")):
                continue
            if bool(p.get("cancel_only")) or bool(p.get("post_only")) or bool(p.get("auction_mode")):
                continue
            base = str(p.get("base_currency")).upper()
            if _lev_or_synth(base):
                continue
            age = 9999
            created_at = p.get("created_at")
            if created_at:
                try:
                    t = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
                    age = max(1, int((now - t).total_seconds() // 86400))
                except Exception:
                    pass
            out.append(Constituent(symbol=str(p.get("id")).upper(), name=f"{base}/{quote}", sector="CRYPTO", industry="SPOT", listing_age_days=age))
        if len(out) < 10:
            return None, f"too few eligible Coinbase products: {len(out)}"
        return out, None
    except Exception as e:
        return None, str(e)


def load_fallback() -> List[Constituent]:
    # Minimal demo-safe fallback.
    syms = ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "LINK-USD", "MATIC-USD"]
    return [Constituent(symbol=s, name=s, sector="CRYPTO", industry="SPOT") for s in syms]
