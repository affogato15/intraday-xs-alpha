"""GICS sector -> SPDR sector ETF mapping for the Phase 0 universe.

Scrapes the same Wikipedia table used by phase0.universe to get each S&P 500
ticker's GICS Sector, then maps GICS Sector names to the 11 SPDR sector ETF
tickers used as sector-relative benchmarks in Phase 0B.

Snapshot is saved to data/phase0_yahoo/sector_map_v1.parquet so downstream
modules can join on ticker deterministically.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import polars as pl

from common.paths import PHASE0_DATA, ensure_dirs
from phase0.universe import SP500_WIKI_URL, _normalize_ticker_for_yahoo

SECTOR_MAP_PATH = PHASE0_DATA / "sector_map_v1.parquet"
SECTOR_MAP_META_PATH = PHASE0_DATA / "sector_map_v1.meta.json"

GICS_TO_ETF: dict[str, str] = {
    "Information Technology": "XLK",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Energy": "XLE",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}


def fetch_sector_map() -> pl.DataFrame:
    import pandas as pd
    import requests

    headers = {
        "User-Agent": (
            "intraday-alpha-research/0.1 (Phase 0B sector map fetch)"
        )
    }
    resp = requests.get(SP500_WIKI_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))
    for tbl in tables:
        cols_lower = {str(c).lower() for c in tbl.columns}
        if "symbol" in cols_lower and any("sector" in c for c in cols_lower):
            sym_col = next(c for c in tbl.columns if str(c).lower() == "symbol")
            sec_col = next(
                c for c in tbl.columns if str(c).lower() == "gics sector"
            )
            tickers = [
                _normalize_ticker_for_yahoo(str(s)) for s in tbl[sym_col].tolist()
            ]
            sectors = [str(s) for s in tbl[sec_col].tolist()]
            etfs = [GICS_TO_ETF.get(s) for s in sectors]
            df = pl.DataFrame(
                {"ticker": tickers, "gics_sector": sectors, "sector_etf": etfs}
            )
            unknown = df.filter(pl.col("sector_etf").is_null())
            if unknown.height:
                raise RuntimeError(
                    f"Unmapped GICS sectors: {unknown['gics_sector'].unique().to_list()}"
                )
            return df.unique(subset=["ticker"]).sort("ticker")
    raise RuntimeError("Could not locate a table with both Symbol and GICS Sector.")


def save_sector_map(df: pl.DataFrame) -> Path:
    ensure_dirs()
    df.write_parquet(SECTOR_MAP_PATH)
    meta = {
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_url": SP500_WIKI_URL,
        "n_tickers": df.height,
        "n_gics_sectors": df["gics_sector"].n_unique(),
        "gics_to_etf": GICS_TO_ETF,
    }
    SECTOR_MAP_META_PATH.write_text(json.dumps(meta, indent=2))
    return SECTOR_MAP_PATH


def load_sector_map() -> pl.DataFrame:
    return pl.read_parquet(SECTOR_MAP_PATH)


if __name__ == "__main__":
    df = fetch_sector_map()
    save_sector_map(df)
    print(f"tickers: {df.height}")
    print(f"sectors: {df['gics_sector'].n_unique()}")
    print(df.group_by("sector_etf").agg(pl.len().alias("n")).sort("n", descending=True))
    print(f"\nSaved: {SECTOR_MAP_PATH}")
