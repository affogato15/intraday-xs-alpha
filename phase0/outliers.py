"""Phase 0 outlier diagnostics.

Reads the audit artifacts and drills into the three flagged groups:
  (A) low mean coverage (<0.90 / <0.50)
  (B) extreme 5-min returns (|r| > 20%)
  (C) intra-session gaps (>5 min inside a single session)

Prints a compact human-readable report; no exclusions applied here.
Decisions about universe_v2 are made after reading this report.
"""
from __future__ import annotations

import polars as pl

from phase0.audit import (
    COVERAGE_BY_SESSION_PATH,
    COVERAGE_BY_TICKER_PATH,
    EXTREME_RETURN_ABS,
)
from phase0.download import BARS_DIR

LOW_COVERAGE_HARD = 0.50
LOW_COVERAGE_SOFT = 0.90


def _load():
    per_ticker = pl.read_parquet(COVERAGE_BY_TICKER_PATH)
    per_session = pl.read_parquet(COVERAGE_BY_SESSION_PATH)
    return per_ticker, per_session


def report_low_coverage(per_ticker: pl.DataFrame, per_session: pl.DataFrame) -> None:
    print("\n=== (A) LOW COVERAGE ===")
    low = per_ticker.filter(pl.col("mean_coverage") < LOW_COVERAGE_SOFT).sort("mean_coverage")
    if low.is_empty():
        print("  none")
        return
    print(low.select([
        "ticker", "mean_coverage", "min_coverage",
        "n_sessions_with_data", "n_missing_sessions",
    ]))

    for tk in low["ticker"].to_list():
        tk_sessions = per_session.filter(pl.col("ticker") == tk).sort("session_date_et")
        present = tk_sessions.filter(pl.col("observed_bars") > 0)
        if present.is_empty():
            print(f"  {tk}: no sessions with data")
            continue
        first = present["session_date_et"].min()
        last = present["session_date_et"].max()
        n_with_data = present.height
        n_total = tk_sessions.height
        print(f"  {tk}: {n_with_data}/{n_total} sessions with data, "
              f"first={first}, last={last}")


def report_extreme_returns(per_ticker: pl.DataFrame) -> None:
    print("\n=== (B) EXTREME 5-MIN RETURNS ===")
    flagged = per_ticker.filter(pl.col("n_sessions_with_extreme_bar") > 0).sort(
        "n_sessions_with_extreme_bar", descending=True
    )
    if flagged.is_empty():
        print("  none")
        return
    print(flagged.select([
        "ticker", "mean_coverage", "n_sessions_with_extreme_bar",
        "total_observed_bars",
    ]))

    print("\n  Details (session, max |ret_5m|, close before/after):")
    for tk in flagged["ticker"].to_list():
        bars_path = BARS_DIR / f"{tk}_5m.parquet"
        if not bars_path.exists():
            continue
        bars = pl.read_parquet(bars_path).sort("ts_utc")
        bars = bars.with_columns(
            (pl.col("close") / pl.col("close").shift(1) - 1.0).alias("ret_5m")
        ).filter(pl.col("ret_5m").abs() > EXTREME_RETURN_ABS)
        if bars.is_empty():
            continue
        print(f"  {tk}:")
        for row in bars.iter_rows(named=True):
            print(f"    {row['ts_et']}  ret_5m={row['ret_5m']*100:+.2f}%  "
                  f"close={row['close']:.4f}  vol={row['volume']}")


def report_intra_session_gaps(per_ticker: pl.DataFrame, per_session: pl.DataFrame) -> None:
    print("\n=== (C) INTRA-SESSION GAPS (>5 min) ===")
    flagged = per_ticker.filter(pl.col("n_sessions_with_gap") > 0)
    print(f"  n tickers flagged: {flagged.height}")
    if flagged.is_empty():
        return
    dist = flagged["n_sessions_with_gap"].describe()
    print("  distribution of n_sessions_with_gap per ticker:")
    print(dist)

    gap_sessions = per_session.filter(pl.col("has_intra_session_gap"))
    if gap_sessions.is_empty():
        return
    print("\n  gap magnitude distribution across (ticker, session) rows:")
    print(gap_sessions.select("max_intra_session_gap_min").describe())

    worst = gap_sessions.sort("max_intra_session_gap_min", descending=True).head(15)
    print("\n  worst 15 (ticker, session, max_gap_min, coverage):")
    print(worst.select([
        "ticker", "session_date_et", "max_intra_session_gap_min", "coverage",
    ]))

    print("\n  top 15 tickers by n_sessions_with_gap:")
    top = flagged.sort("n_sessions_with_gap", descending=True).head(15)
    print(top.select([
        "ticker", "n_sessions_with_gap", "mean_coverage",
    ]))


def main() -> None:
    per_ticker, per_session = _load()
    report_low_coverage(per_ticker, per_session)
    report_extreme_returns(per_ticker)
    report_intra_session_gaps(per_ticker, per_session)


if __name__ == "__main__":
    main()
