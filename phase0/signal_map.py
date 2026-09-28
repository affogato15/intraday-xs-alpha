"""Phase 0A intraday return / IC map.

For each preregistered (L, H, decision_time) triple compute per-timestamp
cross-sectional Spearman Rank IC between the trailing L-minute return and the
forward H-minute return, restricted to eligible (ticker, session, decision_bar)
rows.

    IC_{L,H,t} = corr_spearman_cs( ret_L_{i,t},  fwd_ret_H_{i,t} )

The signal here is raw price returns. Market/sector-relative variants are Phase
0B and live in a separate module.

Artifacts written to data/phase0_audit/signal_map/:
    ic_per_timestamp.parquet   long: (L,H,session_date_et,bar_seq,decision_time,ic,n_names)
    ic_summary_L_H_dt.parquet  wide summary per (L,H,decision_time)
    ic_report.json             top-level scalar report
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from common.paths import PHASE0_AUDIT, ensure_dirs
from phase0.download import BARS_DIR
from phase0.eligibility import (
    DECISION_TIME_BAR_SEQ,
    DEFAULT_MIN_COVERAGE,
    EligibilitySpec,
    apply_eligibility,
)
from phase0.features import build_feature_frame

SIGNAL_MAP_DIR = PHASE0_AUDIT / "signal_map"
IC_PER_TIMESTAMP_PATH = SIGNAL_MAP_DIR / "ic_per_timestamp.parquet"
IC_SUMMARY_PATH = SIGNAL_MAP_DIR / "ic_summary_L_H_dt.parquet"
IC_REPORT_PATH = SIGNAL_MAP_DIR / "ic_report.json"

# Preregistered grids (roadmap Phase 0A).
LOOKBACKS_M: tuple[int, ...] = (5, 15, 30, 60, 120)
PRIMARY_HORIZON_M: int = 60
DIAG_HORIZONS_M: tuple[int, ...] = (30, 120)
ALL_HORIZONS_M: tuple[int, ...] = (PRIMARY_HORIZON_M, *DIAG_HORIZONS_M)

BAR_SEQ_TO_DECISION_TIME: dict[int, str] = {v: k for k, v in DECISION_TIME_BAR_SEQ.items()}


def _load_all_bars() -> pl.DataFrame:
    files = sorted(BARS_DIR.glob("*_5m.parquet"))
    if not files:
        raise FileNotFoundError(f"No bar files in {BARS_DIR}")
    return pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")


def _spearman_ic_per_group(
    df: pl.DataFrame, ret_col: str, target_col: str
) -> pl.DataFrame:
    """Per (session_date_et, bar_seq) compute Spearman IC and n_names.

    Spearman = Pearson of ranks; rank ties averaged. Only eligible rows are
    assumed to be in the input.
    """
    return (
        df.with_columns(
            pl.col(ret_col).rank(method="average").over(
                ["session_date_et", "bar_seq"]
            ).alias("_x_rank"),
            pl.col(target_col).rank(method="average").over(
                ["session_date_et", "bar_seq"]
            ).alias("_y_rank"),
        )
        .group_by(["session_date_et", "bar_seq"])
        .agg(
            pl.corr("_x_rank", "_y_rank").alias("ic"),
            pl.len().alias("n_names"),
        )
        .sort(["session_date_et", "bar_seq"])
    )


def compute_ic_map(min_coverage: float = DEFAULT_MIN_COVERAGE) -> pl.DataFrame:
    """Return the long-format IC-per-timestamp table for all preregistered (L,H,dt)."""
    bars = _load_all_bars()
    feats, _ = build_feature_frame(
        bars,
        lookback_minutes=LOOKBACKS_M,
        horizons_minutes=ALL_HORIZONS_M,
    )
    feats = feats.sort(["ticker", "session_date_et", "bar_seq"])

    decision_bars = tuple(DECISION_TIME_BAR_SEQ.values())
    long_rows: list[pl.DataFrame] = []
    for L in LOOKBACKS_M:
        for H in ALL_HORIZONS_M:
            spec = EligibilitySpec.from_minutes(L, H, min_coverage=min_coverage)
            elig = apply_eligibility(feats, spec, decision_bars=decision_bars)
            elig = elig.filter(pl.col("is_eligible"))
            if elig.is_empty():
                continue
            ret_col = f"ret_{L}m"
            tgt_col = f"fwd_ret_{H}m"
            ic = _spearman_ic_per_group(
                elig.select(
                    ["ticker", "session_date_et", "bar_seq", ret_col, tgt_col]
                ),
                ret_col=ret_col,
                target_col=tgt_col,
            ).with_columns(
                pl.lit(L, dtype=pl.Int32).alias("L_min"),
                pl.lit(H, dtype=pl.Int32).alias("H_min"),
                pl.col("bar_seq")
                .cast(pl.Int32)
                .replace_strict(BAR_SEQ_TO_DECISION_TIME, default=None)
                .alias("decision_time_et"),
            )
            long_rows.append(ic)
    if not long_rows:
        return pl.DataFrame()
    return pl.concat(long_rows, how="vertical_relaxed").sort(
        ["L_min", "H_min", "session_date_et", "bar_seq"]
    )


def summarize_ic(ic_long: pl.DataFrame) -> pl.DataFrame:
    """Per (L, H, decision_time_et): mean/median/std/t-stat/count of IC across sessions."""
    return (
        ic_long.group_by(["L_min", "H_min", "decision_time_et", "bar_seq"])
        .agg(
            pl.col("ic").mean().alias("ic_mean"),
            pl.col("ic").median().alias("ic_median"),
            pl.col("ic").std().alias("ic_std"),
            pl.col("ic").count().alias("n_sessions"),
            pl.col("n_names").mean().alias("avg_n_names"),
        )
        .with_columns(
            (pl.col("ic_mean") / (pl.col("ic_std") / pl.col("n_sessions").sqrt()))
            .alias("ic_tstat_naive")
        )
        .sort(["H_min", "L_min", "bar_seq"])
    )


def run() -> dict:
    ensure_dirs()
    SIGNAL_MAP_DIR.mkdir(parents=True, exist_ok=True)

    ic_long = compute_ic_map()
    ic_long.write_parquet(IC_PER_TIMESTAMP_PATH)

    summary = summarize_ic(ic_long)
    summary.write_parquet(IC_SUMMARY_PATH)

    summary_clean = summary.filter(pl.col("ic_mean").is_not_nan())
    primary = summary_clean.filter(pl.col("H_min") == PRIMARY_HORIZON_M)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "min_coverage": DEFAULT_MIN_COVERAGE,
        "lookbacks_min": list(LOOKBACKS_M),
        "primary_horizon_min": PRIMARY_HORIZON_M,
        "diagnostic_horizons_min": list(DIAG_HORIZONS_M),
        "decision_times_et": list(DECISION_TIME_BAR_SEQ.keys()),
        "n_ic_observations_total": ic_long.height,
        "primary_top10_by_abs_mean": (
            primary.sort(pl.col("ic_mean").abs(), descending=True)
            .head(10)
            .to_dicts()
        ),
        "primary_by_L_avg_across_dt": (
            primary.group_by("L_min")
            .agg(
                pl.col("ic_mean").mean().alias("ic_mean_avg_dt"),
                pl.col("ic_tstat_naive").mean().alias("ic_tstat_avg_dt"),
                pl.col("avg_n_names").mean().alias("avg_n_names"),
            )
            .sort("L_min")
            .to_dicts()
        ),
        "artifacts": {
            "ic_per_timestamp": str(IC_PER_TIMESTAMP_PATH),
            "ic_summary_L_H_dt": str(IC_SUMMARY_PATH),
        },
    }
    IC_REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
    return report


def _print_primary_heatmap(summary: pl.DataFrame) -> None:
    print("\n=== Primary horizon H=60m — mean IC by (L, decision_time) ===")
    primary = summary.filter(pl.col("H_min") == PRIMARY_HORIZON_M)
    pivot = primary.pivot(
        values="ic_mean",
        index="L_min",
        on="decision_time_et",
        aggregate_function="first",
    ).sort("L_min")
    print(pivot)

    print("\n=== Primary H=60m — naive t-stat by (L, decision_time) ===")
    pivot_t = primary.pivot(
        values="ic_tstat_naive",
        index="L_min",
        on="decision_time_et",
        aggregate_function="first",
    ).sort("L_min")
    print(pivot_t)


if __name__ == "__main__":
    report = run()
    summary = pl.read_parquet(IC_SUMMARY_PATH)
    _print_primary_heatmap(summary)
    print("\n=== Top 10 non-NaN (L, H, decision_time) cells by |mean IC| ===")
    top10 = (
        pl.read_parquet(IC_SUMMARY_PATH)
        .filter(pl.col("ic_mean").is_not_nan())
        .sort(pl.col("ic_mean").abs(), descending=True)
        .head(10)
    )
    print(top10.select([
        "L_min", "H_min", "decision_time_et",
        "ic_mean", "ic_median", "ic_std", "n_sessions", "ic_tstat_naive", "avg_n_names",
    ]))
    print(f"\nReport: {IC_REPORT_PATH}")
