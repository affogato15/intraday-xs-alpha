"""Phase 1 hfdatalibrary adapter.

Reads the provider's per-ticker parquet files and normalizes them to the same
canonical schema used across the project (see Phase 0 downloader). Preserves
the provider's `source` column so that the pitrading -> iex boundary (around
2022-03-07) can be tracked at every downstream step.

Timezone-of-record on disk is naive ET wall-clock; the adapter localizes to
`America/New_York`, converts to UTC, and materializes both. No forward-fill.

Ticker rebrands: the provider kept the pre-rebrand symbol on disk in some
cases. `TICKER_ALIASES` maps the canonical post-rebrand ticker (used
everywhere else in the project) to the on-disk provider ticker.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import polars as pl

from common.paths import DATA_ROOT, PHASE1_DATA, PROJECT_ROOT, ensure_dirs

DATASET_DIR = DATA_ROOT / "hfdatalibrary_clean_5min_1391tickers_2026-09-28"
FILE_SUFFIX_STEM = "_clean_5min"
FILE_SUFFIX = f"{FILE_SUFFIX_STEM}.parquet"

INVENTORY_PATH = PHASE1_DATA / "hf_inventory.parquet"
INVENTORY_META_PATH = PHASE1_DATA / "hf_inventory.meta.json"

# Canonical -> provider symbol. Consumers request the canonical (post-rebrand) name;
# the adapter opens the provider's on-disk file for the aliased symbol.
#
# The provider is INCONSISTENT about ticker renames:
#   * Sometimes keeps the pre-rebrand symbol as canonical (FB, FISV, NLOK)
#     — these need aliases so callers can use the modern name.
#   * Sometimes backfills the old history into the new symbol
#     (VTRS <- MYL, NLOK <- SYMC, DOW <- DWDP) — no alias needed; callers
#     use the current name and get the full backfilled series.
#   * Some rebrands are truncated (post-rebrand data missing entirely; see
#     KNOWN_TRUNCATED_SERIES below).
TICKER_ALIASES: dict[str, str] = {
    "META": "FB",    # Facebook -> Meta on 2022-06-09; provider retained FB (ends 2026-07-31)
    "FI":   "FISV",  # Fiserv rebrand on 2024-07-11; provider retained FISV (still current)
    "GEN":  "NLOK",  # NortonLifeLock -> Gen Digital on 2022-11-08; NLOK data truncated at rebrand date
}

# Series that end BEFORE the dataset's overall last date for reasons unrelated
# to a real delisting. Callers must not treat these as "delisted at end date".
KNOWN_TRUNCATED_SERIES: dict[str, dict] = {
    "META": {
        "provider_symbol": "FB",
        "truncation_ts_et": "2022-06-08",
        "reason": (
            "FB is continuous 2012-05-18 → 2022-06-08 (day before FB→META rebrand); "
            "provider added sporadic partial-coverage bars 2025-06-27 → 2026-07-31 "
            "that are unreliable (5-16 sessions/month vs 20). Adapter treats "
            "everything past 2022-06-08 as unavailable."
        ),
    },
    "GEN": {
        "provider_symbol": "NLOK",
        "truncation_ts_et": "2022-11-07",
        "reason": "provider truncated at NortonLifeLock->Gen Digital rebrand; no continuation series",
    },
}

# Non-terminal internal gaps discovered by phase1.data_quality gap detector.
# Downstream eligibility will naturally reject rows inside these windows, but
# recording them here documents the provider quirks explicitly.
KNOWN_INTERNAL_GAPS: dict[str, list[dict]] = {
    "O": [{"gap_after": "2008-02-12", "gap_before": "2008-11-14", "n_missing_sessions": 192}],
    "BF-B": [{"gap_after": "2026-03-27", "gap_before": "2026-09-08", "n_missing_sessions": 111}],
    "BRK-B": [{"gap_after": "2026-03-27", "gap_before": "2026-09-08", "n_missing_sessions": 111}],
}

# S&P 500 members entirely missing from the provider inventory as of the
# 2026-09-28 snapshot. Mostly recent spinoffs and post-2022 additions.
KNOWN_MISSING_SP500: tuple[str, ...] = (
    "GEHC",   # GE Healthcare spin 2023-01-04
    "KVUE",   # Kenvue spin from JNJ 2023-08-23
    "GEV",    # GE Vernova spin 2024-04-02
    "SOLV",   # Solventum spin from MMM 2024-04-01
    "VLTO",   # Veralto spin from DHR 2023-09-30
    "WBD",    # Warner Bros Discovery merger 2022-04-08
    # ...and other recent additions we detected earlier (see KNOWN_MISSING_SP500_SEE_INVENTORY).
    # This tuple is not exhaustive; consult phase1_dataset memory for the full list.
)

# Inverse for provider-symbol -> canonical, built once.
_PROVIDER_TO_CANONICAL: dict[str, str] = {v: k for k, v in TICKER_ALIASES.items()}

CANONICAL_SCHEMA: dict[str, pl.DataType] = {
    "ticker": pl.Utf8,
    "ts_utc": pl.Datetime(time_unit="us", time_zone="UTC"),
    "ts_et": pl.Datetime(time_unit="us", time_zone="America/New_York"),
    "session_date_et": pl.Date,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Int64,
    "source": pl.Utf8,
}

# Provider source markers.
SOURCE_PITRADING = "pitrading"
SOURCE_IEX = "iex"

# Regular session bar-of-day expectations.
BARS_PER_SESSION_REGULAR = 78
SESSION_OPEN_ET = (9, 30)
SESSION_LAST_BAR_OPEN_ET = (15, 55)


def _provider_symbol(canonical: str) -> str:
    return TICKER_ALIASES.get(canonical, canonical)


def _canonical_symbol(provider: str) -> str:
    return _PROVIDER_TO_CANONICAL.get(provider, provider)


def provider_file(canonical: str) -> Path:
    return DATASET_DIR / f"{_provider_symbol(canonical)}{FILE_SUFFIX}"


def available_provider_tickers() -> list[str]:
    """Provider symbols as they appear on disk."""
    return sorted(f.stem.removesuffix(FILE_SUFFIX_STEM) for f in DATASET_DIR.glob(f"*{FILE_SUFFIX}"))


def available_canonical_tickers() -> list[str]:
    """Canonical (post-alias) tickers available in the dataset."""
    return sorted(_canonical_symbol(p) for p in available_provider_tickers())


def _normalize_frame(df: pl.DataFrame, canonical: str) -> pl.DataFrame:
    """Rename/localize/select to the canonical schema. Deduplicate by ts_utc."""
    if df.is_empty():
        return pl.DataFrame(schema=CANONICAL_SCHEMA)

    df = df.rename(
        {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )

    df = df.with_columns(
        pl.col("datetime")
        .cast(pl.Datetime(time_unit="us"))
        .dt.replace_time_zone("America/New_York")
        .alias("ts_et")
    ).with_columns(
        pl.col("ts_et").dt.convert_time_zone("UTC").alias("ts_utc"),
        pl.col("ts_et").dt.date().alias("session_date_et"),
        pl.lit(canonical, dtype=pl.Utf8).alias("ticker"),
    )

    df = df.select(list(CANONICAL_SCHEMA.keys()))
    # Deduplicate defensively — providers occasionally emit dupes at session boundaries.
    return df.sort("ts_utc").unique(subset=["ts_utc"], keep="last", maintain_order=True)


def load_ticker(canonical_ticker: str) -> pl.DataFrame:
    """Load one ticker's full history in the canonical schema.

    Raises FileNotFoundError if the ticker (or its provider alias) is absent.
    """
    path = provider_file(canonical_ticker)
    if not path.exists():
        raise FileNotFoundError(
            f"No parquet for {canonical_ticker} (looked for provider symbol "
            f"{_provider_symbol(canonical_ticker)} at {path})"
        )
    raw = pl.read_parquet(path)
    return _normalize_frame(raw, canonical_ticker)


def load_universe(canonical_tickers: Iterable[str]) -> pl.DataFrame:
    """Load and concatenate multiple tickers. Skips missing tickers silently but
    returns the set of ones actually loaded via `df['ticker'].unique()`.

    For very large universes prefer iterating with `load_ticker` and processing
    per ticker rather than concatenating everything in memory.
    """
    frames: list[pl.DataFrame] = []
    for tk in canonical_tickers:
        path = provider_file(tk)
        if not path.exists():
            continue
        frames.append(_normalize_frame(pl.read_parquet(path), tk))
    if not frames:
        return pl.DataFrame(schema=CANONICAL_SCHEMA)
    return pl.concat(frames, how="vertical_relaxed").sort(["ticker", "ts_utc"])


# ---- Inventory ------------------------------------------------------------


@dataclass(frozen=True)
class TickerInventoryRow:
    ticker: str
    provider_symbol: str
    file: str
    n_bars: int
    first_ts_et: str
    last_ts_et: str
    n_sessions: int
    sources: str  # "|"-joined for parquet-friendliness


def build_inventory() -> pl.DataFrame:
    """Walk the dataset directory once, return per-ticker inventory row."""
    rows: list[dict] = []
    for path in sorted(DATASET_DIR.glob(f"*{FILE_SUFFIX}")):
        provider = path.stem.removesuffix(FILE_SUFFIX_STEM)
        canonical = _canonical_symbol(provider)
        df = pl.read_parquet(path, columns=["datetime", "source"])
        if df.is_empty():
            continue
        first = df["datetime"].min()
        last = df["datetime"].max()
        n_sessions = df.select(pl.col("datetime").dt.date().n_unique()).item()
        srcs = "|".join(df["source"].unique().sort().to_list())
        rows.append(
            {
                "ticker": canonical,
                "provider_symbol": provider,
                "file": str(path.relative_to(PROJECT_ROOT)),
                "n_bars": df.height,
                "first_ts_et": first.isoformat(),
                "last_ts_et": last.isoformat(),
                "n_sessions": n_sessions,
                "sources": srcs,
            }
        )
    return pl.DataFrame(rows).sort("ticker")


def save_inventory(inv: pl.DataFrame | None = None) -> Path:
    ensure_dirs()
    if inv is None:
        inv = build_inventory()
    inv.write_parquet(INVENTORY_PATH)
    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_dir": str(DATASET_DIR.relative_to(PROJECT_ROOT)),
        "n_files": inv.height,
        "n_canonical_tickers": inv["ticker"].n_unique(),
        "n_delisted_pre_2026_07": int(
            (inv["last_ts_et"].str.slice(0, 10) < "2026-07-01").sum()
        ),
        "ticker_aliases": TICKER_ALIASES,
        "known_truncated_series": KNOWN_TRUNCATED_SERIES,
        "known_missing_sp500_partial": list(KNOWN_MISSING_SP500),
    }
    INVENTORY_META_PATH.write_text(json.dumps(meta, indent=2))
    return INVENTORY_PATH


@lru_cache(maxsize=1)
def load_inventory() -> pl.DataFrame:
    if not INVENTORY_PATH.exists():
        save_inventory()
    return pl.read_parquet(INVENTORY_PATH)


if __name__ == "__main__":
    inv = build_inventory()
    save_inventory(inv)
    print(f"Wrote inventory: {inv.height} rows -> {INVENTORY_PATH}")
    print(inv.head(5))

    # Quick smoke check across all alias paths + representative delisted / current.
    for canon in ["SPY", "META", "FI", "GEN", "VTRS", "DOW", "TSLA", "SIVB", "XLK", "WBD", "GEHC"]:
        try:
            df = load_ticker(canon)
            src = df["source"].unique().sort().to_list()
            prov = _provider_symbol(canon)
            alias_note = f" (via {prov})" if prov != canon else ""
            print(
                f"  {canon:<6}{alias_note:<15}: rows={df.height:>7} "
                f"{df['ts_et'].min()} → {df['ts_et'].max()}  sources={src}"
            )
        except FileNotFoundError:
            print(f"  {canon:<6}: MISSING in provider inventory")
