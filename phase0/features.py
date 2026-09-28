"""Phase 0 timestamp-safe feature construction.

The reindexer builds the canonical 78-bar regular-session ET grid
(09:30–15:55 in 5-min steps) per ticker per session, so a `shift(N)` in Polars
becomes a shift of exactly `N*5` minutes and cannot silently jump across missing
bars. Missing bars stay NULL; no OHLCV forward-fill is applied anywhere.

All lookback returns are partitioned by (ticker, session_date_et) so they never
cross the overnight boundary. Overnight moves are exposed only through the
`overnight_gap` feature.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta

import polars as pl

from common.timezones import NY

BARS_PER_SESSION = 78  # 09:30..15:55 inclusive, 5-min steps
SESSION_OPEN = time(9, 30)


def _session_ts_et(session_date: date, bar_seq: int) -> datetime:
    return datetime.combine(session_date, SESSION_OPEN, tzinfo=NY) + timedelta(
        minutes=5 * bar_seq
    )


def expected_intraday_grid(session_dates: list[date]) -> pl.DataFrame:
    """Full (session_date_et, bar_seq, ts_et) grid across the given sessions."""
    rows: list[dict] = []
    for d in session_dates:
        for k in range(BARS_PER_SESSION):
            rows.append(
                {
                    "session_date_et": d,
                    "bar_seq": k,
                    "ts_et": _session_ts_et(d, k),
                }
            )
    return pl.DataFrame(
        rows,
        schema={
            "session_date_et": pl.Date,
            "bar_seq": pl.Int32,
            "ts_et": pl.Datetime(time_unit="us", time_zone="America/New_York"),
        },
    )


def reindex_bars_to_grid(
    bars: pl.DataFrame,
    session_dates: list[date] | None = None,
) -> pl.DataFrame:
    """Left-join every ticker to the full intraday grid. Missing bars -> NULL OHLCV.

    Output schema:
        ticker, session_date_et, bar_seq, ts_et,
        open, high, low, close, volume  (nullable)
    """
    if session_dates is None:
        session_dates = sorted(bars["session_date_et"].unique().to_list())
    grid = expected_intraday_grid(session_dates)
    tickers = bars.select("ticker").unique().sort("ticker")
    full = tickers.join(grid, how="cross")
    ohlcv = bars.select(
        ["ticker", "session_date_et", "ts_et", "open", "high", "low", "close", "volume"]
    )
    return (
        full.join(ohlcv, on=["ticker", "session_date_et", "ts_et"], how="left")
        .sort(["ticker", "session_date_et", "bar_seq"])
    )


def add_bar_return(df: pl.DataFrame) -> pl.DataFrame:
    """Self-contained within-bar return: close/open - 1. Safe on first bar of session."""
    return df.with_columns(
        (pl.col("close") / pl.col("open") - 1.0).alias("ret_bar")
    )


def add_intra_session_ret_5m(df: pl.DataFrame) -> pl.DataFrame:
    """close-to-close 5m return, partitioned by (ticker, session) so never crosses overnight.

    First bar of every session is NULL by construction.
    """
    return df.with_columns(
        (
            pl.col("close")
            / pl.col("close").shift(1).over(["ticker", "session_date_et"])
            - 1.0
        ).alias("ret_5m")
    )


def add_lookback_returns(
    df: pl.DataFrame,
    lookback_minutes: tuple[int, ...] = (5, 15, 30, 60, 120),
) -> pl.DataFrame:
    """Add lookback returns for each L in `lookback_minutes`.

    Requires the frame to have been produced by `reindex_bars_to_grid` — otherwise
    shift(k) does not correspond to a time shift of 5*k minutes.

    Column naming: `ret_{L}m` = close_t / close_{t-L/5 bars} - 1, partitioned by
    (ticker, session_date_et). First L/5 bars of each session are NULL.
    """
    exprs = []
    for L in lookback_minutes:
        if L % 5 != 0:
            raise ValueError(f"Lookback {L}m is not a multiple of 5.")
        k = L // 5
        exprs.append(
            (
                pl.col("close")
                / pl.col("close").shift(k).over(["ticker", "session_date_et"])
                - 1.0
            ).alias(f"ret_{L}m")
        )
    return df.with_columns(exprs)


def add_forward_returns(
    df: pl.DataFrame,
    horizons_minutes: tuple[int, ...] = (30, 60, 120),
) -> pl.DataFrame:
    """Add forward returns for each H in `horizons_minutes`.

    Requires `reindex_bars_to_grid` frame. Partitioned by (ticker, session) so we
    never predict across overnight. Last H/5 bars of every session are NULL.

    Column naming: `fwd_ret_{H}m` = close_{t+H/5 bars} / close_t - 1.
    """
    exprs = []
    for H in horizons_minutes:
        if H % 5 != 0:
            raise ValueError(f"Horizon {H}m is not a multiple of 5.")
        k = H // 5
        exprs.append(
            (
                pl.col("close").shift(-k).over(["ticker", "session_date_et"])
                / pl.col("close")
                - 1.0
            ).alias(f"fwd_ret_{H}m")
        )
    return df.with_columns(exprs)


def add_skip_forward_returns(
    df: pl.DataFrame,
    horizons_minutes: tuple[int, ...] = (60,),
    skip_bars: int = 1,
) -> pl.DataFrame:
    """Forward returns with a leading gap of `skip_bars` 5-min bars.

    `fwd_ret_{H}m_skip{skip_bars}` = close_{t + skip_bars + H/5} / close_{t + skip_bars} - 1.

    Falsification diagnostic for bid-ask bounce on 5-min closes: a genuine
    economic reversal should survive one-bar separation between the decision
    close and the start of the forward window; a microstructure artifact should
    collapse.

    Diagnostic only. Not to be treated as a primary target specification.
    """
    exprs = []
    for H in horizons_minutes:
        if H % 5 != 0:
            raise ValueError(f"Horizon {H}m is not a multiple of 5.")
        k = H // 5
        start = pl.col("close").shift(-skip_bars).over(["ticker", "session_date_et"])
        end = pl.col("close").shift(-(skip_bars + k)).over(
            ["ticker", "session_date_et"]
        )
        exprs.append((end / start - 1.0).alias(f"fwd_ret_{H}m_skip{skip_bars}"))
    return df.with_columns(exprs)


def compute_overnight_gap(df: pl.DataFrame) -> pl.DataFrame:
    """Per (ticker, session): overnight_gap = first_open / prev_last_close - 1.

    First session per ticker has NULL. Uses first non-null open / last non-null close.
    """
    firsts = (
        df.filter(pl.col("open").is_not_null())
        .sort(["ticker", "session_date_et", "bar_seq"])
        .group_by(["ticker", "session_date_et"])
        .agg(pl.col("open").first().alias("first_open"))
    )
    lasts = (
        df.filter(pl.col("close").is_not_null())
        .sort(["ticker", "session_date_et", "bar_seq"])
        .group_by(["ticker", "session_date_et"])
        .agg(pl.col("close").last().alias("last_close"))
    )
    per_session = (
        firsts.join(lasts, on=["ticker", "session_date_et"])
        .sort(["ticker", "session_date_et"])
        .with_columns(
            (
                pl.col("first_open")
                / pl.col("last_close").shift(1).over("ticker")
                - 1.0
            ).alias("overnight_gap")
        )
        .select(["ticker", "session_date_et", "overnight_gap"])
    )
    return per_session


def build_feature_frame(
    bars: pl.DataFrame,
    lookback_minutes: tuple[int, ...] = (5, 15, 30, 60, 120),
    horizons_minutes: tuple[int, ...] = (30, 60, 120),
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Convenience: reindex + bar/intra/lookback/forward returns + overnight gap.

    Returns
    -------
    features : polars.DataFrame
        Bar-grid feature frame.
    overnight : polars.DataFrame
        (ticker, session_date_et, overnight_gap) side table.
    """
    reindexed = reindex_bars_to_grid(bars)
    reindexed = add_bar_return(reindexed)
    reindexed = add_intra_session_ret_5m(reindexed)
    reindexed = add_lookback_returns(reindexed, lookback_minutes=lookback_minutes)
    reindexed = add_forward_returns(reindexed, horizons_minutes=horizons_minutes)
    overnight = compute_overnight_gap(reindexed)
    return reindexed, overnight
