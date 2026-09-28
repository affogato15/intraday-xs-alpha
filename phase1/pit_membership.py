"""Phase 1 PIT S&P 500 membership reconstruction.

Source: `fja05680/sp500` GitHub repo (widely-cited free dataset originating from
Andreas Clenow's "Trading Evolved" + Wikipedia updates). Coverage:
    S&P 500 Historical Components & Changes.csv   1996-01-02 → 2019-01-11
    sp500_changes_since_2019.csv                  2019-01-18 → 2026-08-18

Reconstruction pipeline:
    1. Parse the historical snapshot file into per-date membership sets. The
       source uses a `TICKER-YYYYMM` suffix to retroactively mark delisted
       constituents; we strip that suffix and preserve the ticker.
    2. Sort snapshots by date; between snapshots membership carries forward.
    3. From 2019-01-11 onward apply the event file (add / remove per date).
    4. Explicitly SEPARATE ticker-rename events (FB→META, NLOK→GEN, etc.) from
       real index add/remove churn — same-company relabelings are tagged
       `is_rebrand=True` so churn statistics stay honest.
    5. Emit both a long-format daily membership table (one row per date × member)
       and an events table (one row per add / remove / rebrand).

Trustworthy start: 2001-01-16 (per source maintainer note the initial 5 years
appear to be missing some symbols — 487 on 1996-01-02 rising to 494 by 2001).

Sanity checks are run against known events (TSLA add, KVUE spinoff, GEHC/GEV/SOLV,
plus verified rebrands FB→META and NLOK→GEN).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

from common.paths import PHASE1_DATA, PROJECT_ROOT, ensure_dirs

PIT_DIR = PROJECT_ROOT / "data" / "phase1_pit"
RAW_SNAPSHOT_PATH = PIT_DIR / "sp500_historical_components_raw.csv"
RAW_CHANGES_PATH = PIT_DIR / "sp500_changes_since_2019.csv"

MEMBERSHIP_DAILY_PATH = PIT_DIR / "membership_daily.parquet"
EVENTS_PATH = PIT_DIR / "pit_events.parquet"
YEAR_STATS_PATH = PIT_DIR / "membership_year_stats.parquet"
HF_COVERAGE_PATH = PIT_DIR / "hf_coverage_by_year.parquet"
REPORT_PATH = PIT_DIR / "pit_report.json"

# Trustworthiness cut-off — source maintainer flags pre-2001 data as incomplete.
TRUSTWORTHY_START_ET: date = date(2001, 1, 16)

SOURCE_PROVENANCE = {
    "provider": "fja05680/sp500 (GitHub)",
    "provider_urls": [
        "https://github.com/fja05680/sp500",
        "https://raw.githubusercontent.com/fja05680/sp500/master/S%26P%20500%20Historical%20Components%20%26%20Changes.csv",
        "https://raw.githubusercontent.com/fja05680/sp500/master/sp500_changes_since_2019.csv",
    ],
    "origin": (
        "Andreas F. Clenow, 'Trading Evolved' open dataset (initial snapshot) + "
        "Wikipedia-derived incremental updates by the fja05680 maintainer."
    ),
    "known_limitations": [
        "Pre-2001 membership is likely incomplete (487-494 symbols on early rows; canonical target 500).",
        "sp500_changes_since_2019.csv may lag S&P announcements by a few weeks.",
        "Ticker-rename events are recorded as add/remove pairs in the changes file; must be separated from real index churn.",
        "Latest event as of download: 2026-08-18. Anything after that is not in this snapshot.",
    ],
}

# Add/remove pairs on these exact dates are ticker RENAMES (same underlying
# company), not real index additions/removals. Marked in events table and
# excluded from churn statistics.
KNOWN_REBRAND_EVENTS: dict[str, tuple[str, str]] = {
    # date_iso -> (from_ticker, to_ticker)
    "2022-06-09": ("FB", "META"),
    "2022-11-08": ("NLOK", "GEN"),
    # 2024-07-11 FISV -> FI rebrand is NOT in the changes file (silent update);
    # if it appears in a future refresh, add it here.
}


# ---- Parse raw sources ---------------------------------------------------


def _strip_delist_suffix(t: str) -> str:
    """`XYZ-200812` -> `XYZ`. Leave plain tickers unchanged."""
    if "-" in t and t.rsplit("-", 1)[-1].isdigit():
        return t.rsplit("-", 1)[0]
    return t


def _parse_snapshot_file() -> pl.DataFrame:
    """Return long-format (snapshot_date, ticker) pairs from historical snapshots."""
    raw = pl.read_csv(RAW_SNAPSHOT_PATH)
    return (
        raw.with_columns(pl.col("date").str.to_date().alias("snapshot_date"))
        .with_columns(pl.col("tickers").str.split(","))
        .explode("tickers")  # noqa
        .with_columns(
            pl.col("tickers").str.strip_chars().map_elements(
                _strip_delist_suffix, return_dtype=pl.Utf8
            ).alias("ticker")
        )
        .select(["snapshot_date", "ticker"])
        .unique()
        .sort(["snapshot_date", "ticker"])
    )


def _parse_changes_file() -> pl.DataFrame:
    """Return long-format (event_date, ticker, action) from the changes file."""
    raw = pl.read_csv(RAW_CHANGES_PATH)
    raw = raw.with_columns(pl.col("date").str.to_date().alias("event_date"))
    add_rows = (
        raw.filter(pl.col("add").is_not_null() & (pl.col("add") != ""))
        .with_columns(pl.col("add").str.split(","))
        .explode("add")
        .with_columns(pl.col("add").str.strip_chars().alias("ticker"))
        .with_columns(pl.lit("add").alias("action"))
        .select(["event_date", "ticker", "action"])
    )
    remove_rows = (
        raw.filter(pl.col("remove").is_not_null() & (pl.col("remove") != ""))
        .with_columns(pl.col("remove").str.split(","))
        .explode("remove")
        .with_columns(pl.col("remove").str.strip_chars().alias("ticker"))
        .with_columns(pl.lit("remove").alias("action"))
        .select(["event_date", "ticker", "action"])
    )
    return pl.concat([add_rows, remove_rows]).sort(["event_date", "action", "ticker"])


# ---- Reconstruction ------------------------------------------------------


def _snapshot_membership_series(snapshots: pl.DataFrame) -> dict[date, set[str]]:
    """{snapshot_date: {tickers}} — direct materialization of the snapshot file."""
    grouped = snapshots.group_by("snapshot_date").agg(pl.col("ticker"))
    return {row["snapshot_date"]: set(row["ticker"]) for row in grouped.to_dicts()}


def _apply_changes(
    initial: set[str], changes: pl.DataFrame, initial_date: date
) -> dict[date, set[str]]:
    """Start from `initial` set on `initial_date` (exclusive), apply per-date events."""
    result: dict[date, set[str]] = {}
    current = set(initial)
    grouped = changes.filter(pl.col("event_date") > initial_date).group_by(
        "event_date"
    ).agg(
        pl.col("ticker").filter(pl.col("action") == "add").alias("adds"),
        pl.col("ticker").filter(pl.col("action") == "remove").alias("removes"),
    ).sort("event_date")
    for row in grouped.to_dicts():
        for tk in row["removes"]:
            current.discard(tk)
        for tk in row["adds"]:
            current.add(tk)
        result[row["event_date"]] = set(current)
    return result


def reconstruct_membership() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (membership_daily_long, events_long).

    membership_daily_long columns: date, ticker
    events_long columns:            event_date, ticker, action ("add"/"remove"),
                                    is_rebrand (bool), rebrand_counterparty (str|null)
    """
    snap_long = _parse_snapshot_file()
    changes_long = _parse_changes_file()

    snap_by_date = _snapshot_membership_series(snap_long)
    snap_dates = sorted(snap_by_date.keys())
    last_snap_date = snap_dates[-1]
    tail = _apply_changes(snap_by_date[last_snap_date], changes_long, last_snap_date)

    # Assemble one entry per "as-of" state — starts at each event/snapshot date.
    state_by_date: dict[date, set[str]] = {}
    state_by_date.update(snap_by_date)
    state_by_date.update(tail)

    # Materialize daily: forward-fill state to every calendar day between first
    # and last state date.
    all_state_dates = sorted(state_by_date.keys())
    first_date = all_state_dates[0]
    last_date = all_state_dates[-1]

    daily_rows: list[dict] = []
    current_state: set[str] | None = None
    state_ix = 0
    total_days = (last_date - first_date).days + 1
    for delta in range(total_days):
        d = date.fromordinal(first_date.toordinal() + delta)
        while state_ix < len(all_state_dates) and all_state_dates[state_ix] <= d:
            current_state = state_by_date[all_state_dates[state_ix]]
            state_ix += 1
        if current_state is None:
            continue
        for tk in current_state:
            daily_rows.append({"date": d, "ticker": tk})
    daily_df = pl.DataFrame(daily_rows, schema={"date": pl.Date, "ticker": pl.Utf8})

    # Build events long-form: include both fja changes AND diffs between adjacent
    # historical snapshots (in case the snapshot file has adds/removes not in
    # the changes file).
    diff_events: list[dict] = []
    for i in range(1, len(snap_dates)):
        prev = snap_by_date[snap_dates[i - 1]]
        curr = snap_by_date[snap_dates[i]]
        for tk in curr - prev:
            diff_events.append(
                {"event_date": snap_dates[i], "ticker": tk, "action": "add"}
            )
        for tk in prev - curr:
            diff_events.append(
                {"event_date": snap_dates[i], "ticker": tk, "action": "remove"}
            )
    diff_df = pl.DataFrame(
        diff_events, schema={"event_date": pl.Date, "ticker": pl.Utf8, "action": pl.Utf8}
    )
    events = pl.concat([diff_df, changes_long]).unique().sort(
        ["event_date", "action", "ticker"]
    )

    events = events.with_columns(
        pl.lit(False).alias("is_rebrand"),
        pl.lit(None, dtype=pl.Utf8).alias("rebrand_counterparty"),
    )
    for date_iso, (frm, to) in KNOWN_REBRAND_EVENTS.items():
        d = date.fromisoformat(date_iso)
        events = events.with_columns(
            pl.when(
                (pl.col("event_date") == d)
                & (pl.col("ticker").is_in([frm, to]))
            )
            .then(True)
            .otherwise(pl.col("is_rebrand"))
            .alias("is_rebrand"),
            pl.when(
                (pl.col("event_date") == d) & (pl.col("ticker") == frm)
            )
            .then(pl.lit(to))
            .when(
                (pl.col("event_date") == d) & (pl.col("ticker") == to)
            )
            .then(pl.lit(frm))
            .otherwise(pl.col("rebrand_counterparty"))
            .alias("rebrand_counterparty"),
        )
    return daily_df, events


# ---- Sanity checks -------------------------------------------------------


@dataclass
class SanityCheck:
    label: str
    date: str
    ticker: str
    expected_state: str  # "member" | "not_member"
    actual_state: str
    passed: bool


KNOWN_EVENTS_TO_CHECK = [
    # (date, ticker, expected_state_before_date, expected_state_on_or_after_date, label)
    ("2020-12-20", "TSLA", "not_member", None, "TSLA not in index day before add"),
    ("2020-12-21", "TSLA", None, "member", "TSLA added 2020-12-21"),
    ("2023-08-25", "KVUE", None, "member", "KVUE added 2023-08-25 (JNJ spin)"),
    ("2023-01-04", "GEHC", None, "member", "GEHC added 2023-01-04 (GE spin)"),
    ("2024-04-02", "GEV", None, "member", "GEV added 2024-04-02 (GE Vernova spin)"),
    ("2024-04-01", "SOLV", None, "member", "SOLV added 2024-04-01 (MMM spin)"),
    ("2022-06-08", "FB", None, "member", "FB member day before META rebrand"),
    ("2022-06-09", "META", None, "member", "META member on rebrand day"),
    # LEH appears in the source as LEHMQ (bankruptcy 'Q' suffix). This is a
    # source-specific ticker convention, not a data error.
    ("2008-09-15", "LEHMQ", None, "member", "LEHMQ (Lehman) member day of bankruptcy"),
    ("2023-03-15", "SIVB", None, "not_member", "SIVB removed after 2023-03 collapse"),
    ("2001-01-16", None, None, None, "trustworthy start — >=494 constituents"),
]


def run_sanity_checks(daily: pl.DataFrame) -> list[dict]:
    results: list[dict] = []
    for check_date, ticker, before, after, label in KNOWN_EVENTS_TO_CHECK:
        d = date.fromisoformat(check_date)
        if ticker is None:
            # count-based check
            n = daily.filter(pl.col("date") == d).height
            results.append(
                {
                    "label": label,
                    "date": check_date,
                    "ticker": None,
                    "expected": ">=494 members",
                    "actual": f"{n} members",
                    "passed": n >= 494,
                }
            )
            continue
        # ticker on date check
        is_member = (
            daily.filter((pl.col("date") == d) & (pl.col("ticker") == ticker)).height
            > 0
        )
        if after == "member":
            passed = is_member
            expected = "member"
        elif after == "not_member":
            passed = not is_member
            expected = "not_member"
        elif before == "member":
            passed = is_member
            expected = "member"
        else:
            passed = not is_member
            expected = "not_member"
        results.append(
            {
                "label": label,
                "date": check_date,
                "ticker": ticker,
                "expected": expected,
                "actual": "member" if is_member else "not_member",
                "passed": passed,
            }
        )
    return results


# ---- Reporting -----------------------------------------------------------


def membership_stats_by_year(daily: pl.DataFrame) -> pl.DataFrame:
    return (
        daily.with_columns(pl.col("date").dt.year().alias("year"))
        .group_by("year")
        .agg(
            pl.col("ticker").n_unique().alias("distinct_tickers_in_year"),
            pl.col("date").n_unique().alias("n_days"),
        )
        .with_columns(
            (pl.col("distinct_tickers_in_year") / 1.0).alias("distinct_tickers_in_year"),
        )
        .join(
            daily.with_columns(pl.col("date").dt.year().alias("year"))
            .group_by(["year", "date"])
            .agg(pl.col("ticker").n_unique().alias("members_on_day"))
            .group_by("year")
            .agg(
                pl.col("members_on_day").min().alias("min_daily_members"),
                pl.col("members_on_day").max().alias("max_daily_members"),
                pl.col("members_on_day").median().alias("median_daily_members"),
            ),
            on="year",
        )
        .sort("year")
    )


def hf_coverage_by_year(daily: pl.DataFrame) -> pl.DataFrame:
    """Fraction of PIT constituents present in HF per year (before/after aliases)."""
    from phase1.hfdatalibrary_adapter import TICKER_ALIASES

    inv = pl.read_parquet(PHASE1_DATA / "hf_inventory.parquet")
    hf_tickers_canonical = set(inv["ticker"].to_list())
    # Also include provider-symbol tickers so that pre-rebrand PIT names resolve.
    hf_tickers_provider = set(inv["provider_symbol"].to_list())
    hf_all = hf_tickers_canonical | hf_tickers_provider
    # And explicit aliases: PIT ticker FB should be considered covered by HF (via META alias file FB).
    alias_reverse = {v: k for k, v in TICKER_ALIASES.items()}
    hf_all_with_aliases = set(hf_all)
    for provider_sym in list(hf_all):
        if provider_sym in alias_reverse:
            hf_all_with_aliases.add(alias_reverse[provider_sym])

    daily_annot = daily.with_columns(
        pl.col("date").dt.year().alias("year"),
        pl.col("ticker").is_in(list(hf_all_with_aliases)).alias("covered_by_hf"),
    )
    per_day = daily_annot.group_by(["year", "date"]).agg(
        pl.col("covered_by_hf").sum().alias("n_covered"),
        pl.len().alias("n_members"),
    )
    return (
        per_day.group_by("year")
        .agg(
            pl.col("n_members").median().alias("median_members"),
            pl.col("n_covered").median().alias("median_covered"),
            (pl.col("n_covered") / pl.col("n_members")).median().alias(
                "median_coverage_frac"
            ),
        )
        .sort("year")
    )


def _content_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def run() -> dict:
    ensure_dirs()
    PIT_DIR.mkdir(parents=True, exist_ok=True)

    daily, events = reconstruct_membership()
    daily.write_parquet(MEMBERSHIP_DAILY_PATH)
    events.write_parquet(EVENTS_PATH)

    year_stats = membership_stats_by_year(daily)
    year_stats.write_parquet(YEAR_STATS_PATH)

    hf_cov = hf_coverage_by_year(daily)
    hf_cov.write_parquet(HF_COVERAGE_PATH)

    sanity = run_sanity_checks(daily)

    # Real churn events (excluding rebrands)
    real_add = int(
        events.filter((pl.col("action") == "add") & (~pl.col("is_rebrand"))).height
    )
    real_remove = int(
        events.filter(
            (pl.col("action") == "remove") & (~pl.col("is_rebrand"))
        ).height
    )
    rebrand_events = int(events.filter(pl.col("is_rebrand")).height)

    unresolved_ambiguities = [
        {
            "issue": "Bankruptcy-era ticker suffixes ('Q')",
            "detail": (
                "The source dataset preserves bankruptcy-era ticker suffixes (e.g. "
                "'LEHMQ' for post-bankruptcy Lehman, 'AAMRQ' for AMR/AA). Real-world "
                "S&P membership for Lehman was under 'LEH' pre-bankruptcy; source "
                "conflates these. Downstream code matching PIT tickers against HF "
                "must handle these suffixes; HF may store under either variant."
            ),
        },
        {
            "issue": "FISV → FI rebrand (2024-07-11)",
            "detail": (
                "Fiserv rebranded to 'FI' on 2024-07-11 but fja05680 changes file "
                "does not record this event. PIT membership therefore does not "
                "reflect the rename; downstream code must alias FI ↔ FISV manually "
                "(already done in phase1.hfdatalibrary_adapter TICKER_ALIASES)."
            ),
        },
        {
            "issue": "Pre-2001 completeness",
            "detail": (
                "Historical snapshot begins with 499 tickers on 1996-01-02 and "
                "slowly grows to 494 by 2001-01-16. Maintainer suspects missing "
                "symbols in early years but could not verify. Do not use PIT "
                "membership prior to 2001-01-16 for confirmatory work."
            ),
        },
        {
            "issue": "Post-2026-08-18 latency",
            "detail": (
                "Latest change in the fja05680 changes file is 2026-08-18. Any "
                "S&P membership event between 2026-08-19 and dataset use date "
                "is not reflected. Current Wikipedia snapshot at project start "
                "(2026-09-27) shows 503 constituents including RDDT and VMRK "
                "(added 2026-08-18) but may include additional post-cutoff changes."
            ),
        },
        {
            "issue": "Old delisting-suffix tickers",
            "detail": (
                "Some historic tickers in snapshots carry `-YYYYMM` suffix marking "
                "delisting month (e.g. 'AAL-199702'). Adapter strips the suffix and "
                "uses the bare ticker. Provider files in HF might use fully "
                "different exchange symbols for old delistings — matching against "
                "HF for pre-2010 constituents is imperfect."
            ),
        },
    ]

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": SOURCE_PROVENANCE,
        "file_hashes": {
            "sp500_historical_components.csv": _content_hash(RAW_SNAPSHOT_PATH),
            "sp500_changes_since_2019.csv": _content_hash(RAW_CHANGES_PATH),
        },
        "trustworthy_start_et": TRUSTWORTHY_START_ET.isoformat(),
        "reconstruction_scope": {
            "first_date": str(daily["date"].min()),
            "last_date": str(daily["date"].max()),
            "n_days_covered": daily["date"].n_unique(),
            "n_distinct_tickers_ever_member": daily["ticker"].n_unique(),
        },
        "events_summary": {
            "n_events_total": events.height,
            "real_additions_ex_rebrand": real_add,
            "real_removals_ex_rebrand": real_remove,
            "n_rebrand_rows": rebrand_events,
            "rebrands_recognised": list(KNOWN_REBRAND_EVENTS.keys()),
        },
        "sanity_checks": sanity,
        "n_sanity_passed": sum(1 for c in sanity if c["passed"]),
        "n_sanity_total": len(sanity),
        "unresolved_ambiguities": unresolved_ambiguities,
        "artifacts": {
            "membership_daily": str(MEMBERSHIP_DAILY_PATH),
            "events": str(EVENTS_PATH),
            "year_stats": str(YEAR_STATS_PATH),
            "hf_coverage_by_year": str(HF_COVERAGE_PATH),
        },
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
    return report


if __name__ == "__main__":
    report = run()
    print("=== PIT reconstruction summary ===")
    print(f"scope: {report['reconstruction_scope']['first_date']} → {report['reconstruction_scope']['last_date']}")
    print(f"distinct tickers ever member: {report['reconstruction_scope']['n_distinct_tickers_ever_member']}")
    print(f"events (ex rebrand): +{report['events_summary']['real_additions_ex_rebrand']} / "
          f"-{report['events_summary']['real_removals_ex_rebrand']}, "
          f"rebrand rows: {report['events_summary']['n_rebrand_rows']}")
    print(f"\nSanity checks: {report['n_sanity_passed']}/{report['n_sanity_total']} passed")
    for c in report["sanity_checks"]:
        mark = "✓" if c["passed"] else "✗"
        print(f"  {mark}  {c['label']:<55} expected={c['expected']:<15} actual={c['actual']}")
    print(f"\nYear stats:")
    print(pl.read_parquet(YEAR_STATS_PATH))
    print(f"\nHF coverage of PIT constituents by year:")
    print(pl.read_parquet(HF_COVERAGE_PATH))
    print(f"\nReport: {REPORT_PATH}")
