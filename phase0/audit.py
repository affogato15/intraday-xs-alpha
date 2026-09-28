"""Phase 0.2 coverage & data-quality audit.

Reads the per-ticker 5-minute Parquet files produced by phase0.download and
produces:

- data/phase0_audit/coverage_by_ticker_session.parquet   (grid: ticker × session)
- data/phase0_audit/coverage_by_ticker.parquet           (per-ticker roll-up)
- data/phase0_audit/session_calendar.parquet             (canonical session set)
- data/phase0_audit/audit_report.json                    (top-level report)

Coverage is defined per the roadmap as observed_bars / expected_bars.
ExpectedBars per session is inferred as the max bars observed by any ticker on
that session — this handles regular sessions (78 bars) and early closes (~42
bars) without an external market calendar.

No forward-filling anywhere. Missing bars stay missing.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from common.paths import PHASE0_AUDIT, PHASE0_DATA, ensure_dirs
from phase0.download import BARS_DIR, MANIFEST_PATH

COVERAGE_BY_SESSION_PATH = PHASE0_AUDIT / "coverage_by_ticker_session.parquet"
COVERAGE_BY_TICKER_PATH = PHASE0_AUDIT / "coverage_by_ticker.parquet"
SESSION_CALENDAR_PATH = PHASE0_AUDIT / "session_calendar.parquet"
AUDIT_REPORT_PATH = PHASE0_AUDIT / "audit_report.json"

BAR_INTERVAL_MIN = 5
EXTREME_RETURN_ABS = 0.20  # 20% in 5 minutes -> likely halt/split artifact
BENCHMARK_TICKER = "SPY"


def _load_all_bars() -> pl.DataFrame:
    files = sorted(BARS_DIR.glob("*_5m.parquet"))
    if not files:
        raise FileNotFoundError(f"No bar files in {BARS_DIR}")
    return pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")


def _per_ticker_session_stats(bars: pl.DataFrame) -> pl.DataFrame:
    """Aggregate one row per (ticker, session_date_et).

    `max_abs_ret_5m` is intra-session only (partition by session), so overnight
    gaps do not pollute the intraday-extreme diagnostic. Overnight gaps are
    reported separately.
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
        )
        .sort(["ticker", "session_date_et"])
    )


def _overnight_gaps(bars: pl.DataFrame) -> pl.DataFrame:
    """One row per (ticker, session) with overnight_gap = first_open / prev_last_close - 1.

    First session per ticker gets a null overnight_gap (no prior session available).
    """
    ordered = bars.sort(["ticker", "ts_utc"])
    firsts = (
        ordered.group_by(["ticker", "session_date_et"])
        .agg(pl.col("open").first().alias("first_open"))
    )
    lasts = (
        ordered.group_by(["ticker", "session_date_et"])
        .agg(pl.col("close").last().alias("last_close"))
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


def _session_calendar(per_ts_session: pl.DataFrame) -> pl.DataFrame:
    """Canonical session set with inferred expected bar count.

    expected_bars = max observed_bars across tickers on that session.
    """
    return (
        per_ts_session.group_by("session_date_et")
        .agg(
            pl.col("observed_bars").max().alias("expected_bars"),
            pl.col("ticker").n_unique().alias("tickers_with_any_bars"),
        )
        .sort("session_date_et")
    )


def _coverage_grid(per_ts_session: pl.DataFrame, sessions: pl.DataFrame) -> pl.DataFrame:
    tickers = per_ts_session.select("ticker").unique().sort("ticker")
    full_grid = tickers.join(sessions, how="cross")
    return (
        full_grid.join(
            per_ts_session, on=["ticker", "session_date_et"], how="left"
        )
        .with_columns(
            pl.col("observed_bars").fill_null(0),
            pl.col("zero_volume_bars").fill_null(0),
        )
        .with_columns(
            (pl.col("observed_bars") / pl.col("expected_bars")).alias("coverage"),
            (pl.col("observed_bars") == 0).alias("missing_session"),
            (pl.col("max_abs_ret_5m") > EXTREME_RETURN_ABS).alias("has_extreme_bar"),
            (pl.col("max_intra_session_gap_min") > BAR_INTERVAL_MIN).alias(
                "has_intra_session_gap"
            ),
        )
        .sort(["ticker", "session_date_et"])
    )


def _per_ticker_rollup(grid: pl.DataFrame, n_sessions: int) -> pl.DataFrame:
    return (
        grid.group_by("ticker")
        .agg(
            pl.col("coverage").mean().alias("mean_coverage"),
            pl.col("coverage").median().alias("median_coverage"),
            pl.col("coverage").min().alias("min_coverage"),
            pl.col("missing_session").sum().alias("n_missing_sessions"),
            pl.col("has_intra_session_gap").sum().alias("n_sessions_with_gap"),
            pl.col("has_extreme_bar").sum().alias("n_sessions_with_extreme_bar"),
            pl.col("zero_volume_bars").sum().alias("total_zero_volume_bars"),
            pl.col("observed_bars").sum().alias("total_observed_bars"),
        )
        .with_columns(
            (n_sessions - pl.col("n_missing_sessions")).alias("n_sessions_with_data"),
        )
        .sort("mean_coverage", descending=False)
    )


def _benchmark_alignment(bars: pl.DataFrame) -> dict[str, int | float | None]:
    """Compare each non-benchmark ticker's timestamps to SPY's for a given session set.

    Reports fraction of ticker timestamps that also exist in SPY, averaged across tickers.
    """
    spy = bars.filter(pl.col("ticker") == BENCHMARK_TICKER)
    if spy.is_empty():
        return {"spy_available": False}
    spy_ts = set(spy["ts_utc"].to_list())
    non_bench = bars.filter(pl.col("ticker") != BENCHMARK_TICKER)
    if non_bench.is_empty():
        return {"spy_available": True, "tickers_checked": 0}
    per_ticker_align = (
        non_bench.group_by("ticker")
        .agg(pl.col("ts_utc").alias("all_ts"))
        .with_columns(
            pl.col("all_ts")
            .map_elements(
                lambda ts_list: sum(1 for t in ts_list if t in spy_ts) / max(len(ts_list), 1),
                return_dtype=pl.Float64,
            )
            .alias("frac_aligned_to_spy")
        )
        .select(["ticker", "frac_aligned_to_spy"])
    )
    return {
        "spy_available": True,
        "tickers_checked": per_ticker_align.height,
        "mean_frac_aligned": float(per_ticker_align["frac_aligned_to_spy"].mean()),
        "min_frac_aligned": float(per_ticker_align["frac_aligned_to_spy"].min()),
        "tickers_below_0p95": int(
            (per_ticker_align["frac_aligned_to_spy"] < 0.95).sum()
        ),
    }


def run_audit() -> dict:
    ensure_dirs()
    PHASE0_AUDIT.mkdir(parents=True, exist_ok=True)

    bars = _load_all_bars()
    per_ts_session = _per_ticker_session_stats(bars)
    overnight = _overnight_gaps(bars)
    per_ts_session = per_ts_session.join(
        overnight, on=["ticker", "session_date_et"], how="left"
    )
    sessions = _session_calendar(per_ts_session)
    grid = _coverage_grid(per_ts_session, sessions)
    per_ticker = _per_ticker_rollup(grid, n_sessions=sessions.height)

    grid.write_parquet(COVERAGE_BY_SESSION_PATH)
    per_ticker.write_parquet(COVERAGE_BY_TICKER_PATH)
    sessions.write_parquet(SESSION_CALENDAR_PATH)

    manifest = None
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text())

    align = _benchmark_alignment(bars)

    session_dates = sessions["session_date_et"].to_list()
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_manifest": str(MANIFEST_PATH) if manifest else None,
        "universe_version": manifest.get("universe_version") if manifest else None,
        "data_start_utc": manifest.get("data_start_utc") if manifest else None,
        "data_end_utc": manifest.get("data_end_utc") if manifest else None,
        "n_bar_files": len(list(BARS_DIR.glob("*_5m.parquet"))),
        "n_tickers_with_bars": bars["ticker"].n_unique(),
        "n_sessions": sessions.height,
        "session_min": session_dates[0].isoformat() if session_dates else None,
        "session_max": session_dates[-1].isoformat() if session_dates else None,
        "total_bar_rows": bars.height,
        "expected_bars_per_session": {
            "min": int(sessions["expected_bars"].min()),
            "median": int(sessions["expected_bars"].median()),
            "max": int(sessions["expected_bars"].max()),
        },
        "coverage_by_ticker": {
            "mean_mean_coverage": float(per_ticker["mean_coverage"].mean()),
            "median_mean_coverage": float(per_ticker["mean_coverage"].median()),
            "tickers_below_0p90": int((per_ticker["mean_coverage"] < 0.90).sum()),
            "tickers_below_0p50": int((per_ticker["mean_coverage"] < 0.50).sum()),
        },
        "issues": {
            "total_zero_volume_bars": int(per_ticker["total_zero_volume_bars"].sum()),
            "tickers_with_intra_session_gap": int(
                (per_ticker["n_sessions_with_gap"] > 0).sum()
            ),
            "tickers_with_intra_session_extreme_bar": int(
                (per_ticker["n_sessions_with_extreme_bar"] > 0).sum()
            ),
            "overnight_gap": {
                "n_gap_rows": int(grid["overnight_gap"].is_not_null().sum()),
                "n_abs_gap_gt_5pct": int(
                    (grid["overnight_gap"].abs() > 0.05).sum()
                ),
                "n_abs_gap_gt_10pct": int(
                    (grid["overnight_gap"].abs() > 0.10).sum()
                ),
                "n_abs_gap_gt_20pct": int(
                    (grid["overnight_gap"].abs() > 0.20).sum()
                ),
            },
        },
        "benchmark_alignment": align,
        "artifacts": {
            "coverage_by_ticker_session": str(COVERAGE_BY_SESSION_PATH),
            "coverage_by_ticker": str(COVERAGE_BY_TICKER_PATH),
            "session_calendar": str(SESSION_CALENDAR_PATH),
        },
    }
    AUDIT_REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
    return report


def _parse_args():
    p = argparse.ArgumentParser(description="Phase 0.2 coverage audit")
    return p.parse_args()


if __name__ == "__main__":
    _parse_args()
    report = run_audit()
    print("=== Phase 0.2 audit summary ===")
    print(f"tickers: {report['n_tickers_with_bars']}")
    print(f"sessions: {report['n_sessions']} "
          f"({report['session_min']} → {report['session_max']})")
    print(f"total bar rows: {report['total_bar_rows']}")
    print(f"expected_bars/session (min/med/max): "
          f"{report['expected_bars_per_session']['min']}/"
          f"{report['expected_bars_per_session']['median']}/"
          f"{report['expected_bars_per_session']['max']}")
    print(f"mean coverage across tickers: "
          f"{report['coverage_by_ticker']['mean_mean_coverage']:.3f}")
    print(f"tickers <0.90 mean coverage: "
          f"{report['coverage_by_ticker']['tickers_below_0p90']}")
    print(f"tickers <0.50 mean coverage: "
          f"{report['coverage_by_ticker']['tickers_below_0p50']}")
    print(f"zero-volume bars total: {report['issues']['total_zero_volume_bars']}")
    print(f"tickers with intra-session gap: "
          f"{report['issues']['tickers_with_intra_session_gap']}")
    print(f"tickers with intra-session extreme "
          f"(|r|>{int(EXTREME_RETURN_ABS*100)}%) bar: "
          f"{report['issues']['tickers_with_intra_session_extreme_bar']}")
    og = report['issues']['overnight_gap']
    print(f"overnight gaps: n_rows={og['n_gap_rows']}  "
          f">5%={og['n_abs_gap_gt_5pct']}  "
          f">10%={og['n_abs_gap_gt_10pct']}  "
          f">20%={og['n_abs_gap_gt_20pct']}")
    print(f"SPY alignment: {report['benchmark_alignment']}")
    print(f"Report: {AUDIT_REPORT_PATH}")
