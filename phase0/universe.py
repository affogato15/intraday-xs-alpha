"""Phase 0 seed universe.

Scope v1 (long-only, exploratory): S&P 500 constituents + SPY + SPDR sector ETFs.
This is a *current-membership* universe. Per the roadmap it is acceptable for Phase 0
hypothesis discovery only; it cannot be used to estimate historical strategy
performance because of survivorship bias.

The full liquidity filter (ADV, price) is applied downstream, after bar data is
fetched.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from common.paths import PHASE0_DATA, ensure_dirs

SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

SPDR_SECTOR_ETFS: tuple[str, ...] = (
    "XLK",   # Technology
    "XLF",   # Financials
    "XLV",   # Health Care
    "XLE",   # Energy
    "XLY",   # Consumer Discretionary
    "XLP",   # Consumer Staples
    "XLI",   # Industrials
    "XLB",   # Materials
    "XLU",   # Utilities
    "XLRE",  # Real Estate
    "XLC",   # Communication Services
)

BENCHMARKS: tuple[str, ...] = ("SPY",)


@dataclass(frozen=True)
class UniverseSnapshot:
    version: str
    fetched_at_utc: str
    source_url: str
    sp500_tickers: tuple[str, ...]
    sector_etfs: tuple[str, ...]
    benchmarks: tuple[str, ...]
    notes: str = ""

    @property
    def all_tickers(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for t in (*self.sp500_tickers, *self.sector_etfs, *self.benchmarks):
            seen.setdefault(t, None)
        return tuple(seen)

    def content_hash(self) -> str:
        payload = json.dumps(
            {
                "sp500_tickers": list(self.sp500_tickers),
                "sector_etfs": list(self.sector_etfs),
                "benchmarks": list(self.benchmarks),
            },
            sort_keys=True,
        ).encode()
        return hashlib.sha256(payload).hexdigest()[:12]


def _normalize_ticker_for_yahoo(ticker: str) -> str:
    return ticker.strip().upper().replace(".", "-")


def fetch_sp500_from_wikipedia() -> list[str]:
    """Scrape current S&P 500 constituents from Wikipedia.

    Yahoo-normalized (dots -> dashes, e.g. BRK.B -> BRK-B).
    """
    from io import StringIO

    import pandas as pd
    import requests

    headers = {
        "User-Agent": (
            "intraday-alpha-research/0.1 (Phase 0 exploratory universe fetch; "
            "contact: aleksandrmalyuk15@gmail.com)"
        )
    }
    resp = requests.get(SP500_WIKI_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))
    for tbl in tables:
        cols_lower = {str(c).lower() for c in tbl.columns}
        if "symbol" in cols_lower:
            symbol_col = next(c for c in tbl.columns if str(c).lower() == "symbol")
            tickers = [_normalize_ticker_for_yahoo(str(s)) for s in tbl[symbol_col].tolist()]
            return sorted({t for t in tickers if t and t.isascii()})
    raise RuntimeError("Could not locate S&P 500 constituents table on Wikipedia.")


def build_seed_universe(version: str = "v1") -> UniverseSnapshot:
    sp500 = tuple(fetch_sp500_from_wikipedia())
    return UniverseSnapshot(
        version=version,
        fetched_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        source_url=SP500_WIKI_URL,
        sp500_tickers=sp500,
        sector_etfs=SPDR_SECTOR_ETFS,
        benchmarks=BENCHMARKS,
        notes=(
            "Current-membership S&P 500 + SPDR sector ETFs + SPY. "
            "Acceptable for Phase 0 exploratory hypothesis discovery only. "
            "Survivorship-biased; not to be used for historical performance."
        ),
    )


def snapshot_path(version: str) -> Path:
    return PHASE0_DATA / f"universe_{version}.parquet"


def snapshot_meta_path(version: str) -> Path:
    return PHASE0_DATA / f"universe_{version}.meta.json"


def save_snapshot(snap: UniverseSnapshot) -> Path:
    ensure_dirs()
    rows: list[dict[str, str]] = []
    for t in snap.sp500_tickers:
        rows.append({"ticker": t, "role": "sp500_member"})
    for t in snap.sector_etfs:
        rows.append({"ticker": t, "role": "sector_etf"})
    for t in snap.benchmarks:
        rows.append({"ticker": t, "role": "benchmark"})

    df = pl.DataFrame(rows, schema={"ticker": pl.Utf8, "role": pl.Utf8})
    path = snapshot_path(snap.version)
    df.write_parquet(path)

    meta = asdict(snap)
    meta["content_hash"] = snap.content_hash()
    meta["n_sp500"] = len(snap.sp500_tickers)
    meta["n_sector_etfs"] = len(snap.sector_etfs)
    meta["n_benchmarks"] = len(snap.benchmarks)
    meta["n_total_unique"] = len(snap.all_tickers)
    snapshot_meta_path(snap.version).write_text(json.dumps(meta, indent=2))
    return path


def load_snapshot(version: str) -> UniverseSnapshot:
    meta = json.loads(snapshot_meta_path(version).read_text())
    return UniverseSnapshot(
        version=meta["version"],
        fetched_at_utc=meta["fetched_at_utc"],
        source_url=meta["source_url"],
        sp500_tickers=tuple(meta["sp500_tickers"]),
        sector_etfs=tuple(meta["sector_etfs"]),
        benchmarks=tuple(meta["benchmarks"]),
        notes=meta.get("notes", ""),
    )


if __name__ == "__main__":
    snap = build_seed_universe(version="v1")
    path = save_snapshot(snap)
    print(f"Saved universe v={snap.version} hash={snap.content_hash()}")
    print(f"  sp500={len(snap.sp500_tickers)} sectors={len(snap.sector_etfs)} "
          f"benchmarks={len(snap.benchmarks)} total_unique={len(snap.all_tickers)}")
    print(f"  parquet: {path}")
    print(f"  meta:    {snapshot_meta_path(snap.version)}")
