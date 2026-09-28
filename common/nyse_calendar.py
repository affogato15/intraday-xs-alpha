"""NYSE / Nasdaq trading calendar with early-close awareness.

Wraps `pandas_market_calendars` and returns the exact expected 5-min bar count
per session (78 on full days, 42 on 1 PM early closes, etc.), plus tz-aware
open/close timestamps. Callers must use this instead of assuming 78 bars/day.

Cached to `data/phase1/nyse_calendar.parquet` for reproducibility.
"""
from __future__ import annotations

from datetime import date, datetime

import polars as pl

from common.paths import PHASE1_DATA, ensure_dirs
from common.timezones import NY

REG_SESSION_MINUTES_FULL = 6 * 60 + 30  # 09:30 → 16:00 = 390 min
BAR_MINUTES = 5

NYSE_CALENDAR_PATH = PHASE1_DATA / "nyse_calendar.parquet"


def _load_pmc_schedule(start: date, end: date):
    """pandas_market_calendars NYSE schedule between dates (inclusive)."""
    import pandas_market_calendars as pmc

    nyse = pmc.get_calendar("NYSE")
    return nyse.schedule(start_date=start, end_date=end)


def build_calendar(start: date = date(1990, 1, 1), end: date = date(2027, 1, 1)) -> pl.DataFrame:
    """Return one row per trading session with the exact expected bar count.

    Columns:
        session_date_et       (Date)
        market_open_utc       (Datetime[us, UTC])
        market_close_utc      (Datetime[us, UTC])
        market_open_et_hhmm   (Utf8) — e.g. "09:30"
        market_close_et_hhmm  (Utf8) — e.g. "16:00" or "13:00"
        session_minutes       (Int64) — close - open in minutes
        expected_bars_5m      (Int64) — session_minutes / 5
        is_early_close        (Boolean)
    """
    schedule = _load_pmc_schedule(start, end)
    schedule = schedule.copy()
    open_utc = schedule["market_open"].dt.tz_convert("UTC")
    close_utc = schedule["market_close"].dt.tz_convert("UTC")
    open_et = schedule["market_open"].dt.tz_convert("America/New_York")
    close_et = schedule["market_close"].dt.tz_convert("America/New_York")

    minutes = ((close_utc - open_utc).dt.total_seconds() / 60).astype("int64")

    df = pl.DataFrame(
        {
            "session_date_et": [d.date() for d in close_et],
            "market_open_utc": [d.to_pydatetime() for d in open_utc],
            "market_close_utc": [d.to_pydatetime() for d in close_utc],
            "market_open_et_hhmm": [d.strftime("%H:%M") for d in open_et],
            "market_close_et_hhmm": [d.strftime("%H:%M") for d in close_et],
            "session_minutes": minutes.tolist(),
        },
        schema={
            "session_date_et": pl.Date,
            "market_open_utc": pl.Datetime(time_unit="us", time_zone="UTC"),
            "market_close_utc": pl.Datetime(time_unit="us", time_zone="UTC"),
            "market_open_et_hhmm": pl.Utf8,
            "market_close_et_hhmm": pl.Utf8,
            "session_minutes": pl.Int64,
        },
    )
    return df.with_columns(
        (pl.col("session_minutes") // BAR_MINUTES).alias("expected_bars_5m"),
        (pl.col("session_minutes") < REG_SESSION_MINUTES_FULL).alias("is_early_close"),
    ).sort("session_date_et")


def save_calendar() -> pl.DataFrame:
    ensure_dirs()
    df = build_calendar()
    df.write_parquet(NYSE_CALENDAR_PATH)
    return df


def load_calendar() -> pl.DataFrame:
    if not NYSE_CALENDAR_PATH.exists():
        save_calendar()
    return pl.read_parquet(NYSE_CALENDAR_PATH)


if __name__ == "__main__":
    df = save_calendar()
    print(f"Sessions: {df.height}, range: {df['session_date_et'].min()} → {df['session_date_et'].max()}")
    early = df.filter(pl.col("is_early_close"))
    print(f"\nEarly-close sessions: {early.height}")
    print("Distribution of early-close bar counts:")
    print(early.group_by("expected_bars_5m").agg(pl.len().alias("n")).sort("expected_bars_5m"))
    print("\nEarly closes 2022-2026:")
    print(
        early.filter(
            pl.col("session_date_et") >= pl.date(2022, 1, 1)
        ).select(["session_date_et", "market_close_et_hhmm", "expected_bars_5m"])
    )
