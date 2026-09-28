"""Phase 0A falsification diagnostic — H60 skip-1-bar reversal check.

Recomputes IC for the preregistered lookback grid × decision-time grid using a
forward return whose window starts one 5-min bar AFTER the decision timestamp
(i.e. `close_{t+13} / close_{t+1} - 1` instead of `close_{t+12} / close_{t} - 1`
for H=60m). If the midday reversal we saw in the primary Phase 0A map is
driven by bid-ask bounce on 5-min closes, the one-bar separation should
collapse it.

Diagnostic only. Never a candidate specification. Artifacts are written to a
separate directory so they cannot accidentally enter the winner-selection pool
in Phase 0F.

    IC_{L,H=60,skip=1,t} = corr_spearman_cs( ret_L_{i,t}, fwd_ret_60m_skip1_{i,t} )
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import polars as pl

from common.paths import PHASE0_AUDIT, ensure_dirs
from phase0.download import BARS_DIR
from phase0.eligibility import (
    DECISION_TIME_BAR_SEQ,
    DEFAULT_MIN_COVERAGE,
    EligibilitySpec,
    apply_eligibility,
)
from phase0.features import (
    add_skip_forward_returns,
    build_feature_frame,
)
from phase0.signal_map import (
    BAR_SEQ_TO_DECISION_TIME,
    LOOKBACKS_M,
    PRIMARY_HORIZON_M,
    _spearman_ic_per_group,
)

FALSIFICATION_DIR = PHASE0_AUDIT / "signal_map_falsification"
IC_LONG_PATH = FALSIFICATION_DIR / "ic_skip1_per_timestamp.parquet"
IC_SUMMARY_PATH = FALSIFICATION_DIR / "ic_skip1_summary_L_dt.parquet"
IC_REPORT_PATH = FALSIFICATION_DIR / "ic_skip1_report.json"

SKIP_BARS = 1
DIAG_HORIZON_M = PRIMARY_HORIZON_M  # H=60m only


def _load_all_bars() -> pl.DataFrame:
    files = sorted(BARS_DIR.glob("*_5m.parquet"))
    if not files:
        raise FileNotFoundError(f"No bar files in {BARS_DIR}")
    return pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")


def compute_falsification_ic(min_coverage: float = DEFAULT_MIN_COVERAGE) -> pl.DataFrame:
    bars = _load_all_bars()
    feats, _ = build_feature_frame(
        bars,
        lookback_minutes=LOOKBACKS_M,
        horizons_minutes=(),  # standard forward returns not needed here
    )
    feats = add_skip_forward_returns(
        feats, horizons_minutes=(DIAG_HORIZON_M,), skip_bars=SKIP_BARS
    ).sort(["ticker", "session_date_et", "bar_seq"])

    decision_bars = tuple(DECISION_TIME_BAR_SEQ.values())
    tgt_col = f"fwd_ret_{DIAG_HORIZON_M}m_skip{SKIP_BARS}"
    # Eligibility uses a horizon of (skip + H) minutes so that close_{t + skip + H/5}
    # is required to exist. close_{t + skip} being null will additionally null out
    # the target column, so we drop those rows afterwards.
    effective_horizon_m = 5 * SKIP_BARS + DIAG_HORIZON_M

    long_rows: list[pl.DataFrame] = []
    for L in LOOKBACKS_M:
        spec = EligibilitySpec.from_minutes(
            lookback_m=L,
            horizon_m=effective_horizon_m,
            min_coverage=min_coverage,
        )
        elig = apply_eligibility(feats, spec, decision_bars=decision_bars)
        elig = elig.filter(pl.col("is_eligible") & pl.col(tgt_col).is_not_null())
        if elig.is_empty():
            continue
        ret_col = f"ret_{L}m"
        ic = _spearman_ic_per_group(
            elig.select(
                ["ticker", "session_date_et", "bar_seq", ret_col, tgt_col]
            ),
            ret_col=ret_col,
            target_col=tgt_col,
        ).with_columns(
            pl.lit(L, dtype=pl.Int32).alias("L_min"),
            pl.lit(DIAG_HORIZON_M, dtype=pl.Int32).alias("H_min"),
            pl.lit(SKIP_BARS, dtype=pl.Int32).alias("skip_bars"),
            pl.col("bar_seq")
            .cast(pl.Int32)
            .replace_strict(BAR_SEQ_TO_DECISION_TIME, default=None)
            .alias("decision_time_et"),
        )
        long_rows.append(ic)
    if not long_rows:
        return pl.DataFrame()
    return pl.concat(long_rows, how="vertical_relaxed").sort(
        ["L_min", "session_date_et", "bar_seq"]
    )


def summarize(ic_long: pl.DataFrame) -> pl.DataFrame:
    return (
        ic_long.group_by(["L_min", "H_min", "skip_bars", "decision_time_et", "bar_seq"])
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
        .sort(["L_min", "bar_seq"])
    )


def run() -> dict:
    ensure_dirs()
    FALSIFICATION_DIR.mkdir(parents=True, exist_ok=True)

    ic_long = compute_falsification_ic()
    ic_long.write_parquet(IC_LONG_PATH)
    summary = summarize(ic_long)
    summary.write_parquet(IC_SUMMARY_PATH)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": (
            "Falsification of the H=60m midday reversal seen in the primary "
            "Phase 0A IC map. Forward window is shifted by one 5-min bar; if "
            "the effect is bid-ask bounce microstructure, IC magnitude should "
            "collapse. Diagnostic only, not a candidate specification."
        ),
        "skip_bars": SKIP_BARS,
        "horizon_min": DIAG_HORIZON_M,
        "lookbacks_min": list(LOOKBACKS_M),
        "decision_times_et": list(DECISION_TIME_BAR_SEQ.keys()),
        "n_ic_observations_total": ic_long.height,
        "artifacts": {
            "ic_skip1_per_timestamp": str(IC_LONG_PATH),
            "ic_skip1_summary_L_dt": str(IC_SUMMARY_PATH),
        },
    }
    IC_REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
    return report


def _print_comparison(summary: pl.DataFrame) -> None:
    from phase0.signal_map import IC_SUMMARY_PATH as PRIMARY_SUMMARY_PATH

    print("\n=== Falsification: H=60m skip-1-bar — mean IC by (L, decision_time) ===")
    pivot = summary.pivot(
        values="ic_mean",
        index="L_min",
        on="decision_time_et",
        aggregate_function="first",
    ).sort("L_min")
    print(pivot)

    print("\n=== Falsification: H=60m skip-1-bar — naive t-stat by (L, decision_time) ===")
    pivot_t = summary.pivot(
        values="ic_tstat_naive",
        index="L_min",
        on="decision_time_et",
        aggregate_function="first",
    ).sort("L_min")
    print(pivot_t)

    if PRIMARY_SUMMARY_PATH.exists():
        primary = pl.read_parquet(PRIMARY_SUMMARY_PATH).filter(
            pl.col("H_min") == DIAG_HORIZON_M
        )
        cmp = (
            primary.select(
                ["L_min", "decision_time_et", pl.col("ic_mean").alias("ic_primary")]
            )
            .join(
                summary.select(
                    ["L_min", "decision_time_et", pl.col("ic_mean").alias("ic_skip1")]
                ),
                on=["L_min", "decision_time_et"],
                how="inner",
            )
            .with_columns(
                (pl.col("ic_skip1") - pl.col("ic_primary")).alias("delta_skip1_minus_primary"),
                (
                    pl.col("ic_skip1").abs()
                    / pl.col("ic_primary").abs()
                ).alias("abs_ratio_skip1_over_primary"),
            )
            .sort(pl.col("ic_primary").abs(), descending=True)
        )
        print("\n=== Primary vs skip-1 side-by-side (sorted by |ic_primary|) ===")
        print(cmp)


if __name__ == "__main__":
    report = run()
    summary = pl.read_parquet(IC_SUMMARY_PATH)
    _print_comparison(summary)
    print(f"\nReport: {IC_REPORT_PATH}")
