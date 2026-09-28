"""Phase 0C state / conditioning features.

Preregistered state variables computed on the reindexed 5-min feature frame.
Kept intentionally small (3) to prevent combinatorial bucket search downstream.

Each variable is a trailing-L-bar cross-sectional statistic evaluated at the
decision timestamp. Cross-sectional buckets (quintiles) at decision time will
be used in the conditional IC map.

Variables:
    trailing_volume_{L}m       sum of volume over trailing L/5 bars
    abs_ret_{L}m               |ret_{L}m|
    realized_vol_{L}m          std of intra-session ret_5m over trailing L/5 bars
"""
from __future__ import annotations

from typing import Iterable

import polars as pl


def add_state_variables(
    df: pl.DataFrame,
    lookback_minutes: Iterable[int],
) -> pl.DataFrame:
    """Add trailing-L state variables. Requires reindexed frame with `ret_5m`.

    Requires the frame to have been produced by phase0.features.reindex_bars_to_grid
    (so shift/rolling by k bars corresponds to time shift of 5k minutes).
    """
    lookback_minutes = list(lookback_minutes)
    exprs: list[pl.Expr] = []
    for L in lookback_minutes:
        if L % 5 != 0:
            raise ValueError(f"Lookback {L}m is not a multiple of 5.")
        k = L // 5
        exprs.append(
            pl.col("volume")
            .rolling_sum(window_size=k, min_samples=k)
            .over(["ticker", "session_date_et"])
            .alias(f"trailing_volume_{L}m")
        )
        exprs.append(pl.col(f"ret_{L}m").abs().alias(f"abs_ret_{L}m"))
        # Realized vol needs at least 2 observations for std; skip L=5m entirely.
        if k >= 2:
            exprs.append(
                pl.col("ret_5m")
                .rolling_std(window_size=k, min_samples=min(k, max(2, k // 2)))
                .over(["ticker", "session_date_et"])
                .alias(f"realized_vol_{L}m")
            )
    return df.with_columns(exprs)


STATE_VAR_NAMES: tuple[str, ...] = (
    "trailing_volume",
    "abs_ret",
    "realized_vol",
)


def state_col(name: str, L: int) -> str:
    return f"{name}_{L}m"
