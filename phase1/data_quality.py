"""Phase 1 data-quality audit for the hfdatalibrary dataset.

Runs a streaming per-ticker audit (raw bar frame is 24 years × 1388 tickers ×
78 bars/day ≈ 640M rows — well beyond RAM). Each parquet is read once,
aggregated to per-(ticker, session_date_et) statistics, then dropped. Session-
level artifacts are combined at the end.

Artifacts written to `data/phase1_audit/`:
    coverage_by_ticker_session.parquet   (ticker × session grid, session-level)
    coverage_by_ticker.parquet           (per-ticker roll-up)
    session_calendar.parquet             (canonical dataset session calendar)
    source_switch_by_ticker.parquet      (pitrading → iex crossover per ticker)
    audit_report.json                    (top-level scalar report)

The audit tracks the provider's `source` column so downstream code can guard
against the volume-regime shift at the 2022-03-07 pitrading→iex boundary.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from common.nyse_calendar import load_calendar
from common.paths import PHASE1_AUDIT, ensure_dirs
from phase1.hfdatalibrary_adapter import (
    DATASET_DIR,
    FILE_SUFFIX,
    FILE_SUFFIX_STEM,
    KNOWN_INTERNAL_GAPS,
    KNOWN_TRUNCATED_SERIES,
    SOURCE_IEX,
    SOURCE_PITRADING,
    TICKER_ALIASES,
    _canonical_symbol,
    _normalize_frame,
)

COVERAGE_BY_SESSION_PATH = PHASE1_AUDIT / "coverage_by_ticker_session.parquet"
COVERAGE_BY_TICKER_PATH = PHASE1_AUDIT / "coverage_by_ticker.parquet"
SESSION_CALENDAR_PATH = PHASE1_AUDIT / "session_calendar.parquet"
SOURCE_SWITCH_PATH = PHASE1_AUDIT / "source_switch_by_ticker.parquet"
AUDIT_REPORT_PATH = PHASE1_AUDIT / "audit_report.json"

BAR_INTERVAL_MIN = 5
EXTREME_INTRA_SESSION_ABS = 0.20  # 20% within a single 5-min bar = halt / split artifact
BENCHMARK_TICKER = "SPY"
SOURCE_BOUNDARY_TS_ET = datetime(2022, 3, 7)


def _per_ticker_session_stats(bars: pl.DataFrame) -> pl.DataFrame:
    """Aggregate one row per (ticker, session_date_et). Intra-session only for
    `max_abs_ret_5m` (partitioned by session so overnight gaps don't pollute).
    """
    with_ret = bars.sort(["ticker", "ts_utc"]).with_columns(
        (
            pl.col("close")
            / pl.col("close").shift(1).over(["ticker", "session_date_et"])
            - 1.0
        ).alias("ret_5m_intra"),
        pl.col("ts_utc")
        .diff()
        .over(["ticker", "session_date_et"])
        .dt.total_minutes()
        .alias("gap_min"),
    )
    return (
        with_ret.group_by(["ticker", "session_date_et"])
        .agg(
            pl.len().alias("observed_bars"),
            (pl.col("volume") == 0).sum().alias("zero_volume_bars"),
            pl.col("gap_min").max().alias("max_intra_session_gap_min"),
            pl.col("ret_5m_intra").abs().max().alias("max_abs_ret_5m"),
            pl.col("volume").sum().alias("session_volume"),
            (pl.col("source") == SOURCE_PITRADING).any().alias("has_pitrading"),
            (pl.col("source") == SOURCE_IEX).any().alias("has_iex"),
        )
        .sort(["ticker", "session_date_et"])
    )


def _overnight_gaps(bars: pl.DataFrame) -> pl.DataFrame:
    """Per (ticker, session): overnight_gap = first_open / prev_last_close - 1."""
    ordered = bars.sort(["ticker", "ts_utc"])
    firsts = ordered.group_by(["ticker", "session_date_et"]).agg(
        pl.col("open").first().alias("first_open")
    )
    lasts = ordered.group_by(["ticker", "session_date_et"]).agg(
        pl.col("close").last().alias("last_close")
    )
    joined = firsts.join(lasts, on=["ticker", "session_date_et"]).sort(
        ["ticker", "session_date_et"]
    )
    return joined.with_columns(
        (
            pl.col("first_open")
            / pl.col("last_close").shift(1).over("ticker")
            - 1.0
        ).alias("overnight_gap")
    ).select(["ticker", "session_date_et", "overnight_gap"])


def _source_switch_for_ticker(session_stats: pl.DataFrame) -> dict:
    """Find the first session where each ticker's source becomes IEX-only."""
    first_iex = session_stats.filter(pl.col("has_iex")).select(
        pl.col("session_date_et").min().alias("first_iex_session")
    ).item()
    last_pitrading = session_stats.filter(pl.col("has_pitrading")).select(
        pl.col("session_date_et").max().alias("last_pitrading_session")
    ).item()
    mixed_sessions = int(
        session_stats.filter(pl.col("has_pitrading") & pl.col("has_iex")).height
    )
    return {
        "first_iex_session": first_iex,
        "last_pitrading_session": last_pitrading,
        "n_mixed_source_sessions": mixed_sessions,
    }


def _process_one_file(path: Path) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    provider = path.stem.removesuffix(FILE_SUFFIX_STEM)
    canonical = _canonical_symbol(provider)
    raw = pl.read_parquet(path)
    bars = _normalize_frame(raw, canonical)
    if bars.is_empty():
        return pl.DataFrame(), pl.DataFrame(), {
            "ticker": canonical, "provider_symbol": provider, **{"empty": True}
        }
    per_ts_session = _per_ticker_session_stats(bars)
    overnight = _overnight_gaps(bars)
    per_ts_session = per_ts_session.join(
        overnight, on=["ticker", "session_date_et"], how="left"
    )
    switch = _source_switch_for_ticker(per_ts_session)
    switch["ticker"] = canonical
    switch["provider_symbol"] = provider
    return per_ts_session, overnight, switch


def _session_calendar(per_ts_session: pl.DataFrame) -> pl.DataFrame:
    """Canonical dataset session calendar from NYSE (early-close aware) intersected
    with sessions actually present in the data.

    `expected_bars` is taken from `common.nyse_calendar` (78 on full days, 42 on
    1 PM early closes, etc.) — NOT inferred from the data. This lets us
    distinguish "ticker missing bars on a 78-bar regular day" from "ticker missing
    bars on a 42-bar early close" downstream.
    """
    nyse = load_calendar().select(
        [
            "session_date_et",
            pl.col("expected_bars_5m").alias("expected_bars"),
            pl.col("is_early_close"),
            pl.col("market_close_et_hhmm"),
        ]
    )
    present = (
        per_ts_session.group_by("session_date_et")
        .agg(pl.col("ticker").n_unique().alias("tickers_with_any_bars"))
    )
    return (
        present.join(nyse, on="session_date_et", how="left")
        .with_columns(
            pl.col("expected_bars").fill_null(0),
            pl.col("is_early_close").fill_null(False),
        )
        .sort("session_date_et")
    )


def _coverage_grid(
    per_ts_session: pl.DataFrame, sessions: pl.DataFrame
) -> pl.DataFrame:
    """(ticker × session) grid with coverage and issue flags. Drops sessions
    that aren't on the NYSE calendar (weekend spillover from provider)."""
    valid_sessions = sessions.filter(pl.col("expected_bars") > 0)
    tickers = per_ts_session.select("ticker").unique().sort("ticker")
    full_grid = tickers.join(valid_sessions, how="cross")
    return (
        full_grid.join(per_ts_session, on=["ticker", "session_date_et"], how="left")
        .with_columns(
            pl.col("observed_bars").fill_null(0),
            pl.col("zero_volume_bars").fill_null(0),
        )
        .with_columns(
            (pl.col("observed_bars") / pl.col("expected_bars")).alias("coverage"),
            (pl.col("observed_bars") == 0).alias("missing_session"),
            (pl.col("max_abs_ret_5m") > EXTREME_INTRA_SESSION_ABS).alias(
                "has_extreme_bar"
            ),
            (pl.col("max_intra_session_gap_min") > BAR_INTERVAL_MIN).alias(
                "has_intra_session_gap"
            ),
        )
        .sort(["ticker", "session_date_et"])
    )


def _per_ticker_rollup(
    grid: pl.DataFrame, n_sessions: int, per_ts_session: pl.DataFrame
) -> pl.DataFrame:
    """Per-ticker roll-up.

    Distinguishes:
      * `mean_coverage_all_sessions` — over the dataset's whole calendar (low for
        tickers with short life spans).
      * `mean_coverage_active` — averaged only over sessions between the ticker's
        first and last observed session (fair comparison across tickers).
    """
    active_ranges = per_ts_session.group_by("ticker").agg(
        pl.col("session_date_et").min().alias("first_active_session"),
        pl.col("session_date_et").max().alias("last_active_session"),
    )
    grid_active = grid.join(active_ranges, on="ticker", how="left").filter(
        (pl.col("session_date_et") >= pl.col("first_active_session"))
        & (pl.col("session_date_et") <= pl.col("last_active_session"))
    )
    active_agg = grid_active.group_by("ticker").agg(
        pl.col("coverage").mean().alias("mean_coverage_active"),
        pl.col("coverage").median().alias("median_coverage_active"),
        pl.col("missing_session").sum().alias("n_missing_within_active"),
        pl.len().alias("n_sessions_active"),
    )
    all_agg = (
        grid.group_by("ticker")
        .agg(
            pl.col("coverage").mean().alias("mean_coverage_all_sessions"),
            pl.col("coverage").min().alias("min_coverage"),
            pl.col("missing_session").sum().alias("n_missing_sessions"),
            pl.col("has_intra_session_gap").sum().alias("n_sessions_with_gap"),
            pl.col("has_extreme_bar").sum().alias("n_sessions_with_extreme_bar"),
            pl.col("zero_volume_bars").sum().alias("total_zero_volume_bars"),
            pl.col("observed_bars").sum().alias("total_observed_bars"),
            pl.col("session_volume").sum().alias("total_volume"),
        )
        .with_columns(
            (n_sessions - pl.col("n_missing_sessions")).alias("n_sessions_with_data"),
        )
    )
    return (
        all_agg.join(active_agg, on="ticker", how="left")
        .join(active_ranges, on="ticker", how="left")
        .sort("mean_coverage_active")
    )


def run_audit(subset: list[str] | None = None) -> dict:
    """Stream through all provider files (or the given subset of canonical tickers)."""
    ensure_dirs()
    PHASE1_AUDIT.mkdir(parents=True, exist_ok=True)

    if subset is None:
        files = sorted(DATASET_DIR.glob(f"*{FILE_SUFFIX}"))
    else:
        wanted_provider = {TICKER_ALIASES.get(t, t) for t in subset}
        files = [
            p for p in sorted(DATASET_DIR.glob(f"*{FILE_SUFFIX}"))
            if p.stem.removesuffix(FILE_SUFFIX_STEM) in wanted_provider
        ]
    print(f"Auditing {len(files)} ticker files...")

    per_ts_session_chunks: list[pl.DataFrame] = []
    switch_rows: list[dict] = []
    for i, path in enumerate(files, 1):
        if i % 200 == 0:
            print(f"  processed {i}/{len(files)}")
        stats, _overnight, switch = _process_one_file(path)
        if not stats.is_empty():
            per_ts_session_chunks.append(stats)
        if not switch.get("empty"):
            switch_rows.append(switch)

    per_ts_session = pl.concat(per_ts_session_chunks, how="vertical_relaxed")
    sessions = _session_calendar(per_ts_session)
    grid = _coverage_grid(per_ts_session, sessions)
    per_ticker = _per_ticker_rollup(grid, n_sessions=sessions.height, per_ts_session=per_ts_session)
    switches = pl.DataFrame(switch_rows).sort("ticker")

    grid.write_parquet(COVERAGE_BY_SESSION_PATH)
    per_ticker.write_parquet(COVERAGE_BY_TICKER_PATH)
    sessions.write_parquet(SESSION_CALENDAR_PATH)
    switches.write_parquet(SOURCE_SWITCH_PATH)

    session_dates = sessions["session_date_et"].to_list()
    boundary_sessions = int(
        switches.filter(
            pl.col("first_iex_session") == datetime(2022, 3, 7).date()
        ).height
    )
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_dir": str(DATASET_DIR),
        "n_files_processed": len(files),
        "n_tickers_with_bars": per_ticker.height,
        "n_sessions": sessions.height,
        "session_min": session_dates[0].isoformat() if session_dates else None,
        "session_max": session_dates[-1].isoformat() if session_dates else None,
        "expected_bars_per_session": {
            "min": int(sessions["expected_bars"].min()),
            "median": int(sessions["expected_bars"].median()),
            "max": int(sessions["expected_bars"].max()),
            "n_full_day_78_bars": int(
                (sessions["expected_bars"] == 78).sum()
            ),
            "n_early_close": int(sessions["is_early_close"].sum()),
        },
        "coverage_by_ticker": {
            "mean_active_coverage": float(per_ticker["mean_coverage_active"].mean()),
            "median_active_coverage": float(per_ticker["mean_coverage_active"].median()),
            "tickers_below_0p90_active": int(
                (per_ticker["mean_coverage_active"] < 0.90).sum()
            ),
            "tickers_below_0p50_active": int(
                (per_ticker["mean_coverage_active"] < 0.50).sum()
            ),
            "note": (
                "'active' = averaged over each ticker's own first-to-last session "
                "range, not the full dataset calendar."
            ),
        },
        "issues": {
            "total_zero_volume_bars": int(per_ticker["total_zero_volume_bars"].sum()),
            "tickers_with_intra_session_gap": int(
                (per_ticker["n_sessions_with_gap"] > 0).sum()
            ),
            "tickers_with_intra_session_extreme_bar": int(
                (per_ticker["n_sessions_with_extreme_bar"] > 0).sum()
            ),
        },
        "source_boundary": {
            "expected_boundary_ts_et": SOURCE_BOUNDARY_TS_ET.date().isoformat(),
            "n_tickers_switching_on_expected_date": boundary_sessions,
            "n_tickers_with_iex_data": int(
                switches["first_iex_session"].is_not_null().sum()
            ),
            "n_tickers_with_pitrading_data": int(
                switches["last_pitrading_session"].is_not_null().sum()
            ),
        },
        "known_truncated_series": KNOWN_TRUNCATED_SERIES,
        "known_internal_gaps": KNOWN_INTERNAL_GAPS,
        "artifacts": {
            "coverage_by_ticker_session": str(COVERAGE_BY_SESSION_PATH),
            "coverage_by_ticker": str(COVERAGE_BY_TICKER_PATH),
            "session_calendar": str(SESSION_CALENDAR_PATH),
            "source_switch_by_ticker": str(SOURCE_SWITCH_PATH),
        },
    }
    AUDIT_REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
    return report


if __name__ == "__main__":
    report = run_audit()
    print("=== Phase 1 audit summary ===")
    print(f"tickers: {report['n_tickers_with_bars']}")
    print(f"sessions: {report['n_sessions']} "
          f"({report['session_min']} → {report['session_max']})")
    print(f"expected_bars/session (min/med/max): "
          f"{report['expected_bars_per_session']['min']}/"
          f"{report['expected_bars_per_session']['median']}/"
          f"{report['expected_bars_per_session']['max']}")
    print(f"mean coverage (within active period): "
          f"{report['coverage_by_ticker']['mean_active_coverage']:.3f}")
    print(f"tickers <0.90 active coverage: "
          f"{report['coverage_by_ticker']['tickers_below_0p90_active']}")
    print(f"tickers <0.50 active coverage: "
          f"{report['coverage_by_ticker']['tickers_below_0p50_active']}")
    print(f"zero-volume bars: {report['issues']['total_zero_volume_bars']}")
    print(f"tickers with intra-session gap: "
          f"{report['issues']['tickers_with_intra_session_gap']}")
    print(f"tickers with intra-session extreme (|r|>20%): "
          f"{report['issues']['tickers_with_intra_session_extreme_bar']}")
    print(f"source boundary: {report['source_boundary']}")
    print(f"Report: {AUDIT_REPORT_PATH}")
