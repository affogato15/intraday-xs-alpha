"""Phase 0E weak stability preview.

Red-flag filter for two persistent themes in Phase 0A/B/D:
    A. midday reversal          — L=15m,  decision=13:00
    B. morning momentum          — L=30m,  decision=10:00

For each candidate the IC time series (from phase0.signal_map.py and
signal_map_relative.py output) is aggregated by:
    * ISO week
    * day of week
    * SPY realized-volatility regime for the session

Only H=60m is used (per project rule: H30/H120 are diagnostic-only).

This is a *preview*, not a stability confirmation. 60 sessions cannot support
inferential claims about persistence — the point here is to look for effects
that collapse to a single week / one weekday / one vol regime, which would
downgrade the candidate.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import polars as pl

from common.paths import PHASE0_AUDIT, ensure_dirs
from phase0.download import BARS_DIR
from phase0.signal_map import (
    IC_PER_TIMESTAMP_PATH as RAW_IC_LONG_PATH,
    PRIMARY_HORIZON_M,
)
from phase0.signal_map_relative import (
    IC_LONG_PATH as REL_IC_LONG_PATH,
)

STABILITY_DIR = PHASE0_AUDIT / "stability_preview"
STABILITY_WEEKLY_PATH = STABILITY_DIR / "stability_weekly.parquet"
STABILITY_DOW_PATH = STABILITY_DIR / "stability_dow.parquet"
STABILITY_VOL_PATH = STABILITY_DIR / "stability_volregime.parquet"
STABILITY_REPORT_PATH = STABILITY_DIR / "stability_report.json"

# Candidate cells from Phase 0A/B/D (all H=60m).
CANDIDATES: list[dict] = [
    {"label": "midday_reversal_raw",       "variant": "raw",     "L_min": 15, "decision_time_et": "13:00"},
    {"label": "midday_reversal_sec_rel",   "variant": "sec_rel", "L_min": 15, "decision_time_et": "13:00"},
    {"label": "morning_momentum_raw",      "variant": "raw",     "L_min": 30, "decision_time_et": "10:00"},
    {"label": "morning_momentum_sec_rel",  "variant": "sec_rel", "L_min": 30, "decision_time_et": "10:00"},
]


def _load_ic_long() -> pl.DataFrame:
    raw = pl.read_parquet(RAW_IC_LONG_PATH).with_columns(pl.lit("raw").alias("variant"))
    rel = pl.read_parquet(REL_IC_LONG_PATH)
    keep = ["variant", "L_min", "H_min", "decision_time_et", "session_date_et", "bar_seq", "ic", "n_names"]
    return pl.concat(
        [raw.select(keep), rel.select(keep)], how="vertical_relaxed"
    ).filter(pl.col("H_min") == PRIMARY_HORIZON_M)


def _spy_session_realized_vol() -> pl.DataFrame:
    """Per-session SPY realized vol = std of intra-session 5-min close returns."""
    spy = pl.read_parquet(BARS_DIR / "SPY_5m.parquet").sort("ts_utc")
    return (
        spy.with_columns(
            (pl.col("close") / pl.col("close").shift(1).over("session_date_et") - 1.0)
            .alias("ret_5m")
        )
        .group_by("session_date_et")
        .agg(pl.col("ret_5m").std().alias("spy_rv"))
        .with_columns(
            pl.col("spy_rv")
            .qcut(2, labels=["low", "high"], allow_duplicates=True)
            .cast(pl.String)
            .alias("vol_regime")
        )
        .sort("session_date_et")
    )


def _candidate_ic_series(ic_long: pl.DataFrame, cand: dict) -> pl.DataFrame:
    return ic_long.filter(
        (pl.col("variant") == cand["variant"])
        & (pl.col("L_min") == cand["L_min"])
        & (pl.col("decision_time_et") == cand["decision_time_et"])
    ).sort("session_date_et")


def stability_by_week(ic_series: pl.DataFrame) -> pl.DataFrame:
    return (
        ic_series.with_columns(pl.col("session_date_et").dt.week().alias("iso_week"))
        .group_by("iso_week")
        .agg(
            pl.col("ic").mean().alias("ic_mean"),
            pl.col("ic").count().alias("n"),
        )
        .sort("iso_week")
    )


def stability_by_dow(ic_series: pl.DataFrame) -> pl.DataFrame:
    dow_labels = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri"}
    return (
        ic_series.with_columns(pl.col("session_date_et").dt.weekday().alias("dow_int"))
        .group_by("dow_int")
        .agg(
            pl.col("ic").mean().alias("ic_mean"),
            pl.col("ic").std().alias("ic_std"),
            pl.col("ic").count().alias("n"),
        )
        .with_columns(
            pl.col("dow_int").replace_strict(dow_labels, default="?").alias("dow"),
            (pl.col("ic_mean") / (pl.col("ic_std") / pl.col("n").sqrt())).alias("t"),
        )
        .sort("dow_int")
    )


def stability_by_volregime(
    ic_series: pl.DataFrame, spy_rv: pl.DataFrame
) -> pl.DataFrame:
    return (
        ic_series.join(
            spy_rv.select(["session_date_et", "vol_regime"]),
            on="session_date_et",
            how="left",
        )
        .group_by("vol_regime")
        .agg(
            pl.col("ic").mean().alias("ic_mean"),
            pl.col("ic").std().alias("ic_std"),
            pl.col("ic").count().alias("n"),
        )
        .with_columns(
            (pl.col("ic_mean") / (pl.col("ic_std") / pl.col("n").sqrt())).alias("t")
        )
        .sort("vol_regime")
    )


def run() -> dict:
    ensure_dirs()
    STABILITY_DIR.mkdir(parents=True, exist_ok=True)

    ic_long = _load_ic_long()
    spy_rv = _spy_session_realized_vol()

    weekly_rows: list[pl.DataFrame] = []
    dow_rows: list[pl.DataFrame] = []
    vol_rows: list[pl.DataFrame] = []

    per_candidate_summary = []
    for cand in CANDIDATES:
        series = _candidate_ic_series(ic_long, cand)
        wk = stability_by_week(series).with_columns(pl.lit(cand["label"]).alias("candidate"))
        dw = stability_by_dow(series).with_columns(pl.lit(cand["label"]).alias("candidate"))
        vr = stability_by_volregime(series, spy_rv).with_columns(pl.lit(cand["label"]).alias("candidate"))
        weekly_rows.append(wk)
        dow_rows.append(dw)
        vol_rows.append(vr)

        max_abs_week = wk.select(pl.col("ic_mean").abs().max()).item() if wk.height else float("nan")
        overall_ic = float(series["ic"].mean()) if series.height else float("nan")
        n_weeks_same_sign = int(
            wk.filter(
                pl.col("ic_mean").sign() == pl.lit(overall_ic).sign()
            ).height
        ) if wk.height else 0
        per_candidate_summary.append(
            {
                "label": cand["label"],
                "variant": cand["variant"],
                "L_min": cand["L_min"],
                "decision_time_et": cand["decision_time_et"],
                "n_sessions": series.height,
                "overall_ic_mean": overall_ic,
                "max_abs_weekly_ic": max_abs_week,
                "n_weeks_total": wk.height,
                "n_weeks_same_sign_as_overall": n_weeks_same_sign,
            }
        )

    weekly = pl.concat(weekly_rows, how="vertical_relaxed")
    dow = pl.concat(dow_rows, how="vertical_relaxed")
    vol = pl.concat(vol_rows, how="vertical_relaxed")

    weekly.write_parquet(STABILITY_WEEKLY_PATH)
    dow.write_parquet(STABILITY_DOW_PATH)
    vol.write_parquet(STABILITY_VOL_PATH)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "candidates": CANDIDATES,
        "per_candidate": per_candidate_summary,
        "artifacts": {
            "stability_weekly": str(STABILITY_WEEKLY_PATH),
            "stability_dow": str(STABILITY_DOW_PATH),
            "stability_volregime": str(STABILITY_VOL_PATH),
        },
    }
    STABILITY_REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
    return report, weekly, dow, vol


def _print_stability(weekly, dow, vol) -> None:
    for cand in CANDIDATES:
        label = cand["label"]
        print(f"\n=== {label}  (variant={cand['variant']} L={cand['L_min']}m dt={cand['decision_time_et']}) ===")
        w = weekly.filter(pl.col("candidate") == label).sort("iso_week")
        if w.height:
            print("weekly IC:")
            print(w.select(["iso_week", "ic_mean", "n"]))
        d = dow.filter(pl.col("candidate") == label)
        if d.height:
            print("day-of-week IC:")
            print(d.select(["dow", "ic_mean", "t", "n"]))
        v = vol.filter(pl.col("candidate") == label)
        if v.height:
            print("SPY vol-regime IC:")
            print(v.select(["vol_regime", "ic_mean", "t", "n"]))


if __name__ == "__main__":
    report, weekly, dow, vol = run()
    _print_stability(weekly, dow, vol)
    print("\n=== Per-candidate summary ===")
    print(pl.DataFrame(report["per_candidate"]))
    print(f"\nReport: {STABILITY_REPORT_PATH}")
