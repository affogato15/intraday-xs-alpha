"""Phase 1 eligibility helper.

Thin wrapper on top of the Phase 0 features + eligibility primitives that (a)
streams per-ticker to keep memory bounded on the full 24-year history and (b)
respects Phase 1 constraints — provider truncation dates, optional date range
restriction, and the pitrading→iex source boundary.

Downstream code (Phase 2 confirmatory test) requests an already-eligibility-
filtered feature frame with the frozen-hypothesis L/H parameters. It should
never receive raw bars from the adapter directly.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Iterable

import polars as pl

from phase0.eligibility import (
    DECISION_TIME_BAR_SEQ,
    DEFAULT_MIN_COVERAGE,
    EligibilitySpec,
    apply_eligibility,
)
from phase0.features import (
    add_forward_returns,
    add_intra_session_ret_5m,
    add_lookback_returns,
    reindex_bars_to_grid,
)
from common.paths import PROJECT_ROOT
from phase1.hfdatalibrary_adapter import (
    KNOWN_TRUNCATED_SERIES,
    load_ticker,
)


def frozen_universe_on(as_of: date) -> list[str]:
    """Apply the Phase 1 frozen universe rule: PIT ∩ HF ∖ Real Estate.

    Returns the ordered list of canonical tickers eligible for cross-section
    formation on `as_of`. Used by Phase 2 to build daily cross-sections.
    """
    membership = pl.read_parquet(PROJECT_ROOT / "data" / "phase1_pit" / "membership_daily.parquet")
    inv = pl.read_parquet(PROJECT_ROOT / "data" / "phase1" / "hf_inventory.parquet")
    sector_map = pl.read_parquet(PROJECT_ROOT / "data" / "phase0_yahoo" / "sector_map_v1.parquet")

    pit_tickers = set(
        membership.filter(pl.col("date") == as_of)["ticker"].to_list()
    )
    hf_canonical = set(inv["ticker"].to_list())
    hf_provider = set(inv["provider_symbol"].to_list())
    hf_all = hf_canonical | hf_provider
    real_estate = set(
        sector_map.filter(pl.col("sector_etf") == "XLRE")["ticker"].to_list()
    )
    return sorted((pit_tickers & hf_all) - real_estate)


def _apply_truncation_guard(bars: pl.DataFrame, canonical_ticker: str) -> pl.DataFrame:
    """Drop rows past a known provider truncation date so downstream logic does
    not mistake truncation for delisting."""
    truncation = KNOWN_TRUNCATED_SERIES.get(canonical_ticker)
    if truncation is None:
        return bars
    cutoff = date.fromisoformat(truncation["truncation_ts_et"])
    return bars.filter(pl.col("session_date_et") <= cutoff)


def build_eligible_features(
    canonical_tickers: Iterable[str],
    lookback_minutes: tuple[int, ...] = (15,),
    horizon_minutes: tuple[int, ...] = (60,),
    decision_bar_seqs: tuple[int, ...] = (DECISION_TIME_BAR_SEQ["13:00"],),
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    date_range: tuple[date, date] | None = None,
    verbose: bool = False,
) -> pl.DataFrame:
    """Stream each ticker → reindex → features → eligibility filter → concat.

    Only eligible (ticker, session, decision_bar) rows are returned. Frozen
    hypothesis defaults: L=15m, H=60m, decision 13:00 ET.
    """
    if len(horizon_minutes) != 1:
        raise ValueError(
            "Only a single primary horizon is supported per call; H30/H120 must be "
            "computed as diagnostics in a separate module."
        )
    (h_min,) = horizon_minutes
    frames: list[pl.DataFrame] = []
    tickers_list = list(canonical_tickers)
    n = len(tickers_list)
    for i, tk in enumerate(tickers_list, 1):
        if verbose and i % 100 == 0:
            print(f"  {i}/{n} tickers processed")
        try:
            bars = load_ticker(tk)
        except FileNotFoundError:
            continue
        bars = _apply_truncation_guard(bars, tk)
        if date_range is not None:
            start, end = date_range
            bars = bars.filter(
                (pl.col("session_date_et") >= start)
                & (pl.col("session_date_et") <= end)
            )
        if bars.is_empty():
            continue
        session_dates = sorted(bars["session_date_et"].unique().to_list())
        reindexed = reindex_bars_to_grid(bars, session_dates)
        reindexed = add_intra_session_ret_5m(reindexed)
        reindexed = add_lookback_returns(reindexed, lookback_minutes=lookback_minutes)
        reindexed = add_forward_returns(reindexed, horizons_minutes=(h_min,))

        for L in lookback_minutes:
            spec = EligibilitySpec.from_minutes(L, h_min, min_coverage=min_coverage)
            elig = apply_eligibility(reindexed, spec, decision_bars=decision_bar_seqs)
            keep = elig.filter(pl.col("is_eligible")).with_columns(
                pl.lit(L, dtype=pl.Int32).alias("_lookback_min")
            )
            if not keep.is_empty():
                frames.append(keep)
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="vertical_relaxed").sort(
        ["_lookback_min", "session_date_et", "bar_seq", "ticker"]
    )
