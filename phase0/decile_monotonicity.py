"""Phase 0D decile monotonicity diagnostic.

For each preregistered (L, decision_time) cell, rank the eligible cross-section
by the signal (ret_L for the raw variant, ret_L_sec_rel for the sector-relative
variant) into 10 buckets, then measure the average subsequent 60-minute return
per decile. Report:

    * mean forward return per decile (long-only economics: raw and SPY-relative)
    * monotonicity score = Spearman correlation between decile number and
      mean fwd return across sessions (1.0 = perfect momentum, -1.0 = perfect
      reversal)
    * Q10-Q1 spread and Q10-only mean (long-only view)
    * Q10 vs SPY expected excess return

Q10-Q1 spread remains a diagnostic; portfolio v1 is long-only Top-N.
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
from phase0.residuals import add_all_relatives
from phase0.sector_map import load_sector_map
from phase0.signal_map import (
    BAR_SEQ_TO_DECISION_TIME,
    LOOKBACKS_M,
    PRIMARY_HORIZON_M,
)

DECILE_DIR = PHASE0_AUDIT / "decile_monotonicity"
DECILE_LONG_PATH = DECILE_DIR / "decile_returns_per_timestamp.parquet"
DECILE_SUMMARY_PATH = DECILE_DIR / "decile_summary.parquet"
DECILE_REPORT_PATH = DECILE_DIR / "decile_report.json"

N_DECILES = 10


def _load_all_bars() -> pl.DataFrame:
    files = sorted(BARS_DIR.glob("*_5m.parquet"))
    if not files:
        raise FileNotFoundError(f"No bar files in {BARS_DIR}")
    return pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")


def _assign_deciles(df: pl.DataFrame, signal_col: str) -> pl.DataFrame:
    return (
        df.filter(pl.col(signal_col).is_not_null())
        .with_columns(
            pl.col(signal_col)
            .qcut(N_DECILES, labels=[str(i) for i in range(1, N_DECILES + 1)],
                  allow_duplicates=True)
            .over(["session_date_et", "bar_seq"])
            .cast(pl.String)
            .cast(pl.Int32)
            .alias("decile")
        )
    )


def _per_timestamp_decile_returns(
    df: pl.DataFrame, target_cols: list[str]
) -> pl.DataFrame:
    return (
        df.group_by(["session_date_et", "bar_seq", "decile"])
        .agg(
            *[pl.col(c).mean().alias(f"mean_{c}") for c in target_cols],
            pl.len().alias("n_names"),
        )
        .sort(["session_date_et", "bar_seq", "decile"])
    )


def _monotonicity_score(per_decile_means: pl.DataFrame, value_col: str) -> float:
    """Spearman of decile number vs mean value across deciles."""
    dfm = per_decile_means.sort("decile")
    if dfm.height < 2 or dfm[value_col].null_count() == dfm.height:
        return float("nan")
    return float(
        dfm.select(pl.corr(pl.col("decile").cast(pl.Float64), pl.col(value_col)))
        .item()
    )


def compute_deciles(
    variant: str,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (decile_long, decile_summary) for the requested variant.

    variant in {"raw", "sec_rel"}. Target returns reported: fwd_ret_60m (raw)
    and fwd_ret_60m_spy_rel (SPY-outperformance).
    """
    bars = _load_all_bars()
    feats, _ = build_feature_frame(
        bars,
        lookback_minutes=LOOKBACKS_M,
        horizons_minutes=(PRIMARY_HORIZON_M,),
    )
    feats = add_all_relatives(
        feats, lookback_minutes=LOOKBACKS_M, horizons_minutes=(PRIMARY_HORIZON_M,)
    ).sort(["ticker", "session_date_et", "bar_seq"])

    sector_map = load_sector_map()
    stock_tickers = sector_map["ticker"].to_list()
    feats = feats.filter(pl.col("ticker").is_in(stock_tickers))

    decision_bars = tuple(DECISION_TIME_BAR_SEQ.values())
    tgt_raw = f"fwd_ret_{PRIMARY_HORIZON_M}m"
    tgt_spy_rel = f"fwd_ret_{PRIMARY_HORIZON_M}m_spy_rel"
    target_cols = [tgt_raw, tgt_spy_rel]

    signal_suffix = "" if variant == "raw" else f"_{variant}"

    long_rows: list[pl.DataFrame] = []
    for L in LOOKBACKS_M:
        signal_col = f"ret_{L}m{signal_suffix}"
        spec = EligibilitySpec.from_minutes(
            L, PRIMARY_HORIZON_M, min_coverage=min_coverage
        )
        elig = apply_eligibility(feats, spec, decision_bars=decision_bars).filter(
            pl.col("is_eligible")
            & pl.col(signal_col).is_not_null()
            & pl.col(tgt_raw).is_not_null()
            & pl.col(tgt_spy_rel).is_not_null()
        )
        if elig.is_empty():
            continue
        binned = _assign_deciles(elig, signal_col)
        per_ts = _per_timestamp_decile_returns(binned, target_cols).with_columns(
            pl.lit(L, dtype=pl.Int32).alias("L_min"),
            pl.lit(variant).alias("variant"),
            pl.col("bar_seq")
            .cast(pl.Int32)
            .replace_strict(BAR_SEQ_TO_DECISION_TIME, default=None)
            .alias("decision_time_et"),
        )
        long_rows.append(per_ts)

    if not long_rows:
        return pl.DataFrame(), pl.DataFrame()
    long = pl.concat(long_rows, how="vertical_relaxed").sort(
        ["variant", "L_min", "session_date_et", "bar_seq", "decile"]
    )

    per_decile = (
        long.group_by(["variant", "L_min", "decision_time_et", "decile"])
        .agg(
            pl.col(f"mean_{tgt_raw}").mean().alias(f"avg_mean_{tgt_raw}"),
            pl.col(f"mean_{tgt_spy_rel}").mean().alias(f"avg_mean_{tgt_spy_rel}"),
            pl.col("n_names").mean().alias("avg_n_names"),
            pl.len().alias("n_sessions"),
        )
        .sort(["variant", "L_min", "decision_time_et", "decile"])
    )
    return long, per_decile


def cell_summary(per_decile: pl.DataFrame) -> pl.DataFrame:
    """One row per (variant, L, dt) with Q1/Q10/spread/monotonicity metrics."""
    keys = ["variant", "L_min", "decision_time_et"]
    q1_raw = f"avg_mean_fwd_ret_{PRIMARY_HORIZON_M}m"
    q1_rel = f"avg_mean_fwd_ret_{PRIMARY_HORIZON_M}m_spy_rel"

    q1 = per_decile.filter(pl.col("decile") == 1).select(
        keys
        + [pl.col(q1_raw).alias("q1_ret"), pl.col(q1_rel).alias("q1_ret_spy_rel")]
    )
    q10 = per_decile.filter(pl.col("decile") == N_DECILES).select(
        keys
        + [
            pl.col(q1_raw).alias("q10_ret"),
            pl.col(q1_rel).alias("q10_ret_spy_rel"),
        ]
    )
    summary = q1.join(q10, on=keys, how="inner").with_columns(
        (pl.col("q10_ret") - pl.col("q1_ret")).alias("q10_minus_q1_ret"),
        (pl.col("q10_ret_spy_rel") - pl.col("q1_ret_spy_rel")).alias(
            "q10_minus_q1_spy_rel"
        ),
    )

    # Monotonicity via Spearman of decile vs avg mean return.
    monot_rows: list[dict] = []
    for (variant, L, dt), grp in per_decile.group_by(keys):
        s_raw = _monotonicity_score(grp, q1_raw)
        s_rel = _monotonicity_score(grp, q1_rel)
        monot_rows.append(
            {
                "variant": variant,
                "L_min": L,
                "decision_time_et": dt,
                "monot_raw": s_raw,
                "monot_spy_rel": s_rel,
            }
        )
    monot = pl.DataFrame(monot_rows).with_columns(
        pl.col("L_min").cast(pl.Int32),
    )
    return summary.join(monot, on=keys, how="left").sort(
        pl.col("q10_minus_q1_spy_rel").abs(), descending=True, nulls_last=True
    )


def run() -> dict:
    ensure_dirs()
    DECILE_DIR.mkdir(parents=True, exist_ok=True)

    per_ts_raw, per_dec_raw = compute_deciles("raw")
    per_ts_rel, per_dec_rel = compute_deciles("sec_rel")
    per_ts = pl.concat([per_ts_raw, per_ts_rel], how="vertical_relaxed")
    per_dec = pl.concat([per_dec_raw, per_dec_rel], how="vertical_relaxed")

    per_ts.write_parquet(DECILE_LONG_PATH)
    per_dec.write_parquet(DECILE_SUMMARY_PATH)

    summary = cell_summary(per_dec)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_deciles": N_DECILES,
        "lookbacks_min": list(LOOKBACKS_M),
        "primary_horizon_min": PRIMARY_HORIZON_M,
        "variants": ["raw", "sec_rel"],
        "top10_cells_by_abs_q10_minus_q1_spy_rel": summary.head(10).to_dicts(),
        "artifacts": {
            "per_timestamp_deciles": str(DECILE_LONG_PATH),
            "per_decile_summary": str(DECILE_SUMMARY_PATH),
        },
    }
    DECILE_REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
    return report, summary, per_dec


def _print_top_cells(summary: pl.DataFrame, n: int = 8) -> None:
    print(f"\n=== Top {n} cells by |Q10-Q1 spread on SPY-rel fwd 60m return| ===")
    print(
        summary.select(
            [
                "variant",
                "L_min",
                "decision_time_et",
                (pl.col("q1_ret_spy_rel") * 10000).alias("q1_bps"),
                (pl.col("q10_ret_spy_rel") * 10000).alias("q10_bps"),
                (pl.col("q10_minus_q1_spy_rel") * 10000).alias("spread_bps"),
                pl.col("monot_spy_rel").round(3).alias("monot"),
            ]
        ).head(n)
    )


def _print_full_decile(per_dec: pl.DataFrame, variant: str, L: int, dt: str) -> None:
    slc = per_dec.filter(
        (pl.col("variant") == variant)
        & (pl.col("L_min") == L)
        & (pl.col("decision_time_et") == dt)
    ).sort("decile")
    if slc.is_empty():
        return
    print(f"\n--- variant={variant}  L={L}m  dt={dt}  full decile breakdown ---")
    print(
        slc.select(
            [
                "decile",
                (pl.col(f"avg_mean_fwd_ret_{PRIMARY_HORIZON_M}m") * 10000)
                .round(2)
                .alias("avg_fwd60m_bps"),
                (
                    pl.col(f"avg_mean_fwd_ret_{PRIMARY_HORIZON_M}m_spy_rel") * 10000
                )
                .round(2)
                .alias("avg_fwd60m_spy_rel_bps"),
                "avg_n_names",
                "n_sessions",
            ]
        )
    )


if __name__ == "__main__":
    report, summary, per_dec = run()
    _print_top_cells(summary, n=12)
    # Full decile breakdown for the three strongest primary cells we've been
    # tracking (from Phase 0A raw):
    for (variant, L, dt) in [
        ("raw", 15, "13:00"),
        ("sec_rel", 15, "13:00"),
        ("sec_rel", 30, "10:00"),
        ("sec_rel", 5, "13:00"),
    ]:
        _print_full_decile(per_dec, variant, L, dt)
    print(f"\nReport: {DECILE_REPORT_PATH}")
