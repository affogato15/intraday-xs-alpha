"""Phase 0B market/sector-relative return construction.

Adds SPY-relative (β=1) and sector-relative (β=1 vs SPDR sector ETF) variants
of each lookback and forward return already present on the feature frame. Full
beta-adjusted residuals are deferred — this Phase 0B pass fixes β=1 to sidestep
window-choice / lookahead concerns in the exploratory stage. Beta-adjusted
variants can be added later as an extension of the preregistered grid, not as a
post-hoc replacement of these.

Output columns are named:
    ret_{L}m_spy_rel      = ret_{L}m - ret_{L}m_SPY
    fwd_ret_{H}m_spy_rel  = fwd_ret_{H}m - fwd_ret_{H}m_SPY
    ret_{L}m_sec_rel      = ret_{L}m - ret_{L}m_{sector_etf(ticker)}
    fwd_ret_{H}m_sec_rel  = fwd_ret_{H}m - fwd_ret_{H}m_{sector_etf(ticker)}
"""
from __future__ import annotations

from typing import Iterable

import polars as pl

from phase0.sector_map import load_sector_map

SPY_TICKER = "SPY"


def _benchmark_returns(
    feats: pl.DataFrame,
    benchmark_tickers: Iterable[str],
    lookback_minutes: Iterable[int],
    horizons_minutes: Iterable[int],
) -> pl.DataFrame:
    """Extract benchmark-only (session, bar_seq, bench_ticker) return columns."""
    cols = ["ticker", "session_date_et", "bar_seq"]
    for L in lookback_minutes:
        cols.append(f"ret_{L}m")
    for H in horizons_minutes:
        cols.append(f"fwd_ret_{H}m")
    bench = (
        feats.filter(pl.col("ticker").is_in(list(benchmark_tickers)))
        .select(cols)
        .rename({"ticker": "bench_ticker"})
    )
    return bench


def add_spy_relative(
    feats: pl.DataFrame,
    lookback_minutes: Iterable[int],
    horizons_minutes: Iterable[int],
) -> pl.DataFrame:
    """Add `ret_{L}m_spy_rel` and `fwd_ret_{H}m_spy_rel` columns."""
    lookback_minutes = list(lookback_minutes)
    horizons_minutes = list(horizons_minutes)
    spy = (
        feats.filter(pl.col("ticker") == SPY_TICKER)
        .select(
            ["session_date_et", "bar_seq"]
            + [f"ret_{L}m" for L in lookback_minutes]
            + [f"fwd_ret_{H}m" for H in horizons_minutes]
        )
        .rename(
            {f"ret_{L}m": f"_spy_ret_{L}m" for L in lookback_minutes}
            | {f"fwd_ret_{H}m": f"_spy_fwd_ret_{H}m" for H in horizons_minutes}
        )
    )
    joined = feats.join(spy, on=["session_date_et", "bar_seq"], how="left")
    exprs = []
    for L in lookback_minutes:
        exprs.append((pl.col(f"ret_{L}m") - pl.col(f"_spy_ret_{L}m")).alias(f"ret_{L}m_spy_rel"))
    for H in horizons_minutes:
        exprs.append(
            (pl.col(f"fwd_ret_{H}m") - pl.col(f"_spy_fwd_ret_{H}m")).alias(
                f"fwd_ret_{H}m_spy_rel"
            )
        )
    out = joined.with_columns(exprs)
    drop_cols = [f"_spy_ret_{L}m" for L in lookback_minutes] + [
        f"_spy_fwd_ret_{H}m" for H in horizons_minutes
    ]
    return out.drop(drop_cols)


def add_sector_relative(
    feats: pl.DataFrame,
    lookback_minutes: Iterable[int],
    horizons_minutes: Iterable[int],
    sector_map: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Add `ret_{L}m_sec_rel` and `fwd_ret_{H}m_sec_rel` columns.

    Each ticker's sector ETF is looked up from `sector_map` (default: fresh load
    from disk). Non-S&P-500 rows (SPY, sector ETFs themselves) receive NULL for
    sector-relative columns.
    """
    lookback_minutes = list(lookback_minutes)
    horizons_minutes = list(horizons_minutes)
    if sector_map is None:
        sector_map = load_sector_map()

    etf_tickers = sector_map["sector_etf"].unique().to_list()
    etf_returns = (
        feats.filter(pl.col("ticker").is_in(etf_tickers))
        .select(
            ["ticker", "session_date_et", "bar_seq"]
            + [f"ret_{L}m" for L in lookback_minutes]
            + [f"fwd_ret_{H}m" for H in horizons_minutes]
        )
        .rename({"ticker": "sector_etf"})
        .rename(
            {f"ret_{L}m": f"_sec_ret_{L}m" for L in lookback_minutes}
            | {f"fwd_ret_{H}m": f"_sec_fwd_ret_{H}m" for H in horizons_minutes}
        )
    )
    # Attach each ticker's sector ETF then join on (session, bar_seq, sector_etf).
    feats_with_sec = feats.join(
        sector_map.select(["ticker", "sector_etf"]),
        on="ticker",
        how="left",
    )
    joined = feats_with_sec.join(
        etf_returns,
        on=["sector_etf", "session_date_et", "bar_seq"],
        how="left",
    )
    exprs = []
    for L in lookback_minutes:
        exprs.append(
            (pl.col(f"ret_{L}m") - pl.col(f"_sec_ret_{L}m")).alias(f"ret_{L}m_sec_rel")
        )
    for H in horizons_minutes:
        exprs.append(
            (pl.col(f"fwd_ret_{H}m") - pl.col(f"_sec_fwd_ret_{H}m")).alias(
                f"fwd_ret_{H}m_sec_rel"
            )
        )
    out = joined.with_columns(exprs)
    drop_cols = [f"_sec_ret_{L}m" for L in lookback_minutes] + [
        f"_sec_fwd_ret_{H}m" for H in horizons_minutes
    ]
    return out.drop(drop_cols)


def add_all_relatives(
    feats: pl.DataFrame,
    lookback_minutes: Iterable[int],
    horizons_minutes: Iterable[int],
) -> pl.DataFrame:
    with_spy = add_spy_relative(feats, lookback_minutes, horizons_minutes)
    with_sec = add_sector_relative(with_spy, lookback_minutes, horizons_minutes)
    return with_sec
