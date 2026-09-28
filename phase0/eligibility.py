"""Phase 0 eligibility gate: (ticker, session, decision bar) triple must pass.

Rules (no ticker-level blacklist):
  1. Lookback coverage: within the window (decision_bar - lookback_bars, decision_bar],
     the fraction of non-null closes must be >= min_coverage (default 0.85).
     This also enforces PIT availability automatically — a ticker whose history
     starts mid-window fails the gate.
  2. Decision bar itself: close at decision_bar is not null.
  3. Horizon bar: close at (decision_bar + horizon_bars) is not null. Otherwise
     the target cannot be formed and the pair cannot contribute to IC.

Preregistered decision-time grid (bar_seq indexing, bar_seq=0 -> 09:30 ET):
    10:00 -> 6
    11:00 -> 18
    12:00 -> 30
    13:00 -> 42
    14:00 -> 54
    15:00 -> 66
"""
from __future__ import annotations

from dataclasses import dataclass

import polars as pl

DECISION_TIME_BAR_SEQ: dict[str, int] = {
    "10:00": 6,
    "11:00": 18,
    "12:00": 30,
    "13:00": 42,
    "14:00": 54,
    "15:00": 66,
}

DEFAULT_MIN_COVERAGE = 0.85


@dataclass(frozen=True)
class EligibilitySpec:
    lookback_bars: int
    horizon_bars: int
    min_coverage: float = DEFAULT_MIN_COVERAGE

    @classmethod
    def from_minutes(
        cls, lookback_m: int, horizon_m: int, min_coverage: float = DEFAULT_MIN_COVERAGE
    ) -> "EligibilitySpec":
        if lookback_m % 5 or horizon_m % 5:
            raise ValueError("Lookback / horizon must be multiples of 5.")
        return cls(
            lookback_bars=lookback_m // 5,
            horizon_bars=horizon_m // 5,
            min_coverage=min_coverage,
        )


def _lookback_coverage_frac(reindexed: pl.DataFrame, lookback_bars: int) -> pl.Expr:
    """Fraction of non-null closes in the trailing `lookback_bars` bars (inclusive)."""
    return (
        pl.col("close")
        .is_not_null()
        .cast(pl.Float64)
        .rolling_mean(window_size=lookback_bars, min_samples=1)
        .over(["ticker", "session_date_et"])
    )


def apply_eligibility(
    reindexed: pl.DataFrame,
    spec: EligibilitySpec,
    decision_bars: tuple[int, ...] | None = None,
) -> pl.DataFrame:
    """Add `is_eligible` and `lookback_coverage` columns and optionally restrict to
    the requested decision-bar rows.

    Parameters
    ----------
    reindexed : polars.DataFrame
        Output of `phase0.features.reindex_bars_to_grid` (must be sorted by
        ticker, session_date_et, bar_seq).
    spec : EligibilitySpec
        Lookback / horizon / min-coverage configuration.
    decision_bars : tuple of ints, optional
        If provided, only rows whose `bar_seq` is in this set are returned;
        others are dropped. Passing None returns all bar_seq rows with the flag.
    """
    fwd_close = pl.col("close").shift(-spec.horizon_bars).over(
        ["ticker", "session_date_et"]
    )
    with_flags = reindexed.with_columns(
        _lookback_coverage_frac(reindexed, spec.lookback_bars).alias(
            "lookback_coverage"
        ),
        fwd_close.alias("_fwd_close"),
    ).with_columns(
        (
            (pl.col("lookback_coverage") >= spec.min_coverage)
            & pl.col("close").is_not_null()
            & pl.col("_fwd_close").is_not_null()
        ).alias("is_eligible")
    ).drop("_fwd_close")

    if decision_bars is not None:
        with_flags = with_flags.filter(pl.col("bar_seq").is_in(list(decision_bars)))
    return with_flags


def eligibility_summary(eligible_frame: pl.DataFrame) -> pl.DataFrame:
    """Roll up eligibility counts per (bar_seq, session_date_et)."""
    return (
        eligible_frame.group_by(["session_date_et", "bar_seq"])
        .agg(
            pl.col("is_eligible").sum().alias("n_eligible"),
            pl.col("ticker").n_unique().alias("n_tickers_total"),
        )
        .with_columns(
            (pl.col("n_eligible") / pl.col("n_tickers_total")).alias("eligible_frac")
        )
        .sort(["session_date_et", "bar_seq"])
    )
