"""Phase 0B IC map on market/sector-relative returns.

Runs the same preregistered (L, H, decision_time) grid as Phase 0A but on
residualized (β=1) returns:
    variant = "spy_rel"  -> ret_L - ret_L_SPY,   fwd_ret_H - fwd_ret_H_SPY
    variant = "sec_rel"  -> ret_L - ret_L_ETF,   fwd_ret_H - fwd_ret_H_ETF
                            where ETF is the ticker's SPDR sector ETF.

Cross-section excludes SPY and the sector ETFs themselves (they are benchmarks,
not tradable names in the ranking). Only S&P 500 stocks contribute.

Full β-adjusted residuals (estimating β_{i,M}, β_{i,S}) are Phase 0B extension,
not part of this pass; keeping β=1 sidesteps window-choice / lookahead issues
in the exploratory stage.
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
from phase0.residuals import SPY_TICKER, add_all_relatives
from phase0.sector_map import load_sector_map
from phase0.signal_map import (
    ALL_HORIZONS_M,
    BAR_SEQ_TO_DECISION_TIME,
    LOOKBACKS_M,
    PRIMARY_HORIZON_M,
    _spearman_ic_per_group,
)

RELATIVE_DIR = PHASE0_AUDIT / "signal_map_relative"
IC_LONG_PATH = RELATIVE_DIR / "ic_per_timestamp.parquet"
IC_SUMMARY_PATH = RELATIVE_DIR / "ic_summary_variant_L_H_dt.parquet"
IC_REPORT_PATH = RELATIVE_DIR / "ic_report.json"


def _load_all_bars() -> pl.DataFrame:
    files = sorted(BARS_DIR.glob("*_5m.parquet"))
    if not files:
        raise FileNotFoundError(f"No bar files in {BARS_DIR}")
    return pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")


def _stock_universe_mask(feats: pl.DataFrame, sector_map: pl.DataFrame) -> pl.Series:
    """True for S&P 500 stocks (rows to include in the cross-section)."""
    stock_tickers = sector_map["ticker"].to_list()
    return feats["ticker"].is_in(stock_tickers)


def compute_relative_ic(min_coverage: float = DEFAULT_MIN_COVERAGE) -> pl.DataFrame:
    bars = _load_all_bars()
    feats, _ = build_feature_frame(
        bars,
        lookback_minutes=LOOKBACKS_M,
        horizons_minutes=ALL_HORIZONS_M,
    )
    feats = add_all_relatives(
        feats,
        lookback_minutes=LOOKBACKS_M,
        horizons_minutes=ALL_HORIZONS_M,
    ).sort(["ticker", "session_date_et", "bar_seq"])

    sector_map = load_sector_map()
    stock_mask = _stock_universe_mask(feats, sector_map)
    feats_stocks = feats.filter(stock_mask)

    decision_bars = tuple(DECISION_TIME_BAR_SEQ.values())
    long_rows: list[pl.DataFrame] = []
    for variant, ret_suffix, tgt_suffix in [
        ("spy_rel", "_spy_rel", "_spy_rel"),
        ("sec_rel", "_sec_rel", "_sec_rel"),
    ]:
        for L in LOOKBACKS_M:
            for H in ALL_HORIZONS_M:
                spec = EligibilitySpec.from_minutes(L, H, min_coverage=min_coverage)
                elig = apply_eligibility(feats_stocks, spec, decision_bars=decision_bars)
                ret_col = f"ret_{L}m{ret_suffix}"
                tgt_col = f"fwd_ret_{H}m{tgt_suffix}"
                elig = elig.filter(
                    pl.col("is_eligible")
                    & pl.col(ret_col).is_not_null()
                    & pl.col(tgt_col).is_not_null()
                )
                if elig.is_empty():
                    continue
                ic = _spearman_ic_per_group(
                    elig.select(
                        ["ticker", "session_date_et", "bar_seq", ret_col, tgt_col]
                    ),
                    ret_col=ret_col,
                    target_col=tgt_col,
                ).with_columns(
                    pl.lit(variant).alias("variant"),
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
        ["variant", "L_min", "H_min", "session_date_et", "bar_seq"]
    )


def summarize(ic_long: pl.DataFrame) -> pl.DataFrame:
    return (
        ic_long.group_by(["variant", "L_min", "H_min", "decision_time_et", "bar_seq"])
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
        .sort(["variant", "H_min", "L_min", "bar_seq"])
    )


def run() -> dict:
    ensure_dirs()
    RELATIVE_DIR.mkdir(parents=True, exist_ok=True)

    ic_long = compute_relative_ic()
    ic_long.write_parquet(IC_LONG_PATH)
    summary = summarize(ic_long)
    summary.write_parquet(IC_SUMMARY_PATH)

    summary_clean = summary.filter(pl.col("ic_mean").is_not_nan())
    primary = summary_clean.filter(pl.col("H_min") == PRIMARY_HORIZON_M)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "min_coverage": DEFAULT_MIN_COVERAGE,
        "lookbacks_min": list(LOOKBACKS_M),
        "primary_horizon_min": PRIMARY_HORIZON_M,
        "variants": ["spy_rel", "sec_rel"],
        "n_ic_observations_total": ic_long.height,
        "primary_top10_by_variant": {
            v: (
                primary.filter(pl.col("variant") == v)
                .sort(pl.col("ic_mean").abs(), descending=True)
                .head(10)
                .to_dicts()
            )
            for v in ["spy_rel", "sec_rel"]
        },
        "artifacts": {
            "ic_per_timestamp": str(IC_LONG_PATH),
            "ic_summary_variant_L_H_dt": str(IC_SUMMARY_PATH),
        },
    }
    IC_REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
    return report


def _print_variant_heatmap(summary: pl.DataFrame, variant: str) -> None:
    print(f"\n=== {variant.upper()} — Primary H=60m mean IC by (L, decision_time) ===")
    slc = summary.filter(
        (pl.col("variant") == variant) & (pl.col("H_min") == PRIMARY_HORIZON_M)
    )
    pivot = slc.pivot(
        values="ic_mean",
        index="L_min",
        on="decision_time_et",
        aggregate_function="first",
    ).sort("L_min")
    print(pivot)

    print(f"\n=== {variant.upper()} — Primary H=60m naive t-stat by (L, decision_time) ===")
    pivot_t = slc.pivot(
        values="ic_tstat_naive",
        index="L_min",
        on="decision_time_et",
        aggregate_function="first",
    ).sort("L_min")
    print(pivot_t)


def _print_side_by_side_with_raw(summary_rel: pl.DataFrame) -> None:
    from phase0.signal_map import IC_SUMMARY_PATH as RAW_SUMMARY_PATH

    if not RAW_SUMMARY_PATH.exists():
        return
    raw = pl.read_parquet(RAW_SUMMARY_PATH).filter(
        pl.col("H_min") == PRIMARY_HORIZON_M
    )
    rel = summary_rel.filter(pl.col("H_min") == PRIMARY_HORIZON_M)
    for variant in ["spy_rel", "sec_rel"]:
        v = rel.filter(pl.col("variant") == variant).select(
            ["L_min", "decision_time_et", pl.col("ic_mean").alias(f"ic_{variant}")]
        )
        raw = raw.join(v, on=["L_min", "decision_time_et"], how="left")
    print("\n=== Primary H=60m — raw vs SPY-rel vs sec-rel (mean IC) ===")
    print(
        raw.rename({"ic_mean": "ic_raw"})
        .select(
            ["L_min", "decision_time_et", "ic_raw", "ic_spy_rel", "ic_sec_rel"]
        )
        .sort(pl.col("ic_raw").abs(), descending=True)
        .head(20)
    )


if __name__ == "__main__":
    report = run()
    summary = pl.read_parquet(IC_SUMMARY_PATH)
    _print_variant_heatmap(summary, "spy_rel")
    _print_variant_heatmap(summary, "sec_rel")
    _print_side_by_side_with_raw(summary)
    print(f"\nReport: {IC_REPORT_PATH}")
