"""Phase 0C conditional IC map (state-variable quintile buckets).

For each preregistered (L, decision_time) and state variable S in
{trailing_volume, abs_ret, realized_vol}, at each timestamp split the eligible
cross-section into cross-sectional quintiles of S, then compute the Rank IC
between ret_L and fwd_ret_60m within the top quintile (Q5) and bottom quintile
(Q1). Primary H=60m only (H30/H120 remain diagnostic-only per project rules).

Interpretation: if the primary signal is driven by high-activity or high-vol
names, IC(Q5) is meaningfully more negative than IC(Q1) (or vice versa).
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
from phase0.features import build_feature_frame
from phase0.sector_map import load_sector_map
from phase0.signal_map import (
    BAR_SEQ_TO_DECISION_TIME,
    LOOKBACKS_M,
    PRIMARY_HORIZON_M,
    _spearman_ic_per_group,
)
from phase0.state_features import STATE_VAR_NAMES, add_state_variables, state_col

CONDITIONAL_DIR = PHASE0_AUDIT / "signal_map_conditional"
IC_LONG_PATH = CONDITIONAL_DIR / "ic_conditional_per_timestamp.parquet"
IC_SUMMARY_PATH = CONDITIONAL_DIR / "ic_conditional_summary.parquet"
IC_REPORT_PATH = CONDITIONAL_DIR / "ic_conditional_report.json"

N_QUINTILES = 5


def _load_all_bars() -> pl.DataFrame:
    files = sorted(BARS_DIR.glob("*_5m.parquet"))
    if not files:
        raise FileNotFoundError(f"No bar files in {BARS_DIR}")
    return pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")


def _assign_quintiles(df: pl.DataFrame, state_column: str) -> pl.DataFrame:
    """Add `quintile` column with cross-sectional quintile of state_column at (session, bar_seq).

    Rows with null state values are dropped from the quintile assignment.
    """
    return (
        df.filter(pl.col(state_column).is_not_null())
        .with_columns(
            pl.col(state_column)
            .qcut(N_QUINTILES, labels=[str(i) for i in range(1, N_QUINTILES + 1)],
                  allow_duplicates=True)
            .over(["session_date_et", "bar_seq"])
            .cast(pl.String)
            .cast(pl.Int32)
            .alias("quintile")
        )
    )


def compute_conditional_ic(min_coverage: float = DEFAULT_MIN_COVERAGE) -> pl.DataFrame:
    bars = _load_all_bars()
    feats, _ = build_feature_frame(
        bars,
        lookback_minutes=LOOKBACKS_M,
        horizons_minutes=(PRIMARY_HORIZON_M,),
    )
    feats = add_state_variables(feats, lookback_minutes=LOOKBACKS_M).sort(
        ["ticker", "session_date_et", "bar_seq"]
    )

    # Stocks only (exclude SPY and sector ETFs from the cross-section).
    sector_map = load_sector_map()
    stock_tickers = set(sector_map["ticker"].to_list())
    feats = feats.filter(pl.col("ticker").is_in(list(stock_tickers)))

    decision_bars = tuple(DECISION_TIME_BAR_SEQ.values())
    H = PRIMARY_HORIZON_M
    tgt_col = f"fwd_ret_{H}m"

    long_rows: list[pl.DataFrame] = []
    for L in LOOKBACKS_M:
        ret_col = f"ret_{L}m"
        spec = EligibilitySpec.from_minutes(L, H, min_coverage=min_coverage)
        elig = apply_eligibility(feats, spec, decision_bars=decision_bars)
        elig = elig.filter(
            pl.col("is_eligible")
            & pl.col(ret_col).is_not_null()
            & pl.col(tgt_col).is_not_null()
        )
        if elig.is_empty():
            continue

        for state_name in STATE_VAR_NAMES:
            state_column = state_col(state_name, L)
            if state_column not in elig.columns:
                continue
            binned = _assign_quintiles(elig, state_column)
            for q in (1, N_QUINTILES):  # bottom + top quintiles only
                bucket = binned.filter(pl.col("quintile") == q)
                if bucket.is_empty():
                    continue
                ic = _spearman_ic_per_group(
                    bucket.select(
                        [
                            "ticker",
                            "session_date_et",
                            "bar_seq",
                            ret_col,
                            tgt_col,
                        ]
                    ),
                    ret_col=ret_col,
                    target_col=tgt_col,
                ).with_columns(
                    pl.lit(state_name).alias("state_var"),
                    pl.lit(q, dtype=pl.Int32).alias("quintile"),
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
        ["state_var", "L_min", "quintile", "session_date_et", "bar_seq"]
    )


def summarize(ic_long: pl.DataFrame) -> pl.DataFrame:
    return (
        ic_long.group_by(
            ["state_var", "L_min", "H_min", "quintile", "decision_time_et", "bar_seq"]
        )
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
        .sort(["state_var", "L_min", "bar_seq", "quintile"])
    )


def _q_diff_view(summary: pl.DataFrame) -> pl.DataFrame:
    """Wide view: for each (state_var, L, dt), show IC_Q1, IC_Q5, and Q5-Q1 diff."""
    keys = ["state_var", "L_min", "decision_time_et"]
    q1 = summary.filter(pl.col("quintile") == 1).select(
        keys + [pl.col("ic_mean").alias("ic_Q1"), pl.col("ic_tstat_naive").alias("t_Q1")]
    )
    q5 = summary.filter(pl.col("quintile") == N_QUINTILES).select(
        keys + [pl.col("ic_mean").alias("ic_Q5"), pl.col("ic_tstat_naive").alias("t_Q5")]
    )
    return (
        q1.join(q5, on=keys, how="full", coalesce=True)
        .with_columns((pl.col("ic_Q5") - pl.col("ic_Q1")).alias("ic_Q5_minus_Q1"))
        .sort(pl.col("ic_Q5_minus_Q1").abs(), descending=True, nulls_last=True)
    )


def run() -> dict:
    ensure_dirs()
    CONDITIONAL_DIR.mkdir(parents=True, exist_ok=True)

    ic_long = compute_conditional_ic()
    ic_long.write_parquet(IC_LONG_PATH)
    summary = summarize(ic_long)
    summary.write_parquet(IC_SUMMARY_PATH)

    q_diff = _q_diff_view(summary)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "min_coverage": DEFAULT_MIN_COVERAGE,
        "state_vars": list(STATE_VAR_NAMES),
        "lookbacks_min": list(LOOKBACKS_M),
        "primary_horizon_min": PRIMARY_HORIZON_M,
        "n_quintiles": N_QUINTILES,
        "n_ic_observations_total": ic_long.height,
        "top10_by_abs_Q5_minus_Q1": q_diff.head(10).to_dicts(),
        "artifacts": {
            "ic_conditional_per_timestamp": str(IC_LONG_PATH),
            "ic_conditional_summary": str(IC_SUMMARY_PATH),
        },
    }
    IC_REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
    return report


def _print_summary(summary: pl.DataFrame) -> None:
    for state in STATE_VAR_NAMES:
        print(f"\n=== state={state} — IC by (L, decision_time, quintile) ===")
        slc = summary.filter(pl.col("state_var") == state)
        for L in LOOKBACKS_M:
            for dt in DECISION_TIME_BAR_SEQ.keys():
                rows = slc.filter(
                    (pl.col("L_min") == L) & (pl.col("decision_time_et") == dt)
                )
                if rows.is_empty():
                    continue
                pieces = []
                for q in (1, N_QUINTILES):
                    r = rows.filter(pl.col("quintile") == q)
                    if r.height == 0:
                        pieces.append(f"Q{q}: n/a")
                    else:
                        ic = r["ic_mean"][0]
                        t = r["ic_tstat_naive"][0]
                        pieces.append(f"Q{q}={ic:+.4f}(t={t:+.2f})")
                print(f"  L={L:>3}m dt={dt:>5}  " + "  ".join(pieces))


if __name__ == "__main__":
    report = run()
    summary = pl.read_parquet(IC_SUMMARY_PATH)
    q_diff = _q_diff_view(summary)
    print("\n=== Top 15 by |IC_Q5 - IC_Q1| ===")
    print(q_diff.head(15))
    _print_summary(summary)
    print(f"\nReport: {IC_REPORT_PATH}")
