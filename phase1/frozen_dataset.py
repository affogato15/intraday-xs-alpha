"""Phase 1 FROZEN dataset specification.

This module is the single machine-readable source of truth for what Phase 2
consumes. Every rule, source, hash, and policy that governs the confirmatory
dataset is nailed down here.

DO NOT EDIT after freeze without also creating a new versioned entry in the
Phase 1 trial registry. Any change here means Phase 2 runs against a different
dataset than the one described by the freeze — and per the roadmap that starts
a new experiment.

Provenance note: none of the values in this module were tuned in response to
Phase 2 hypothesis-performance metrics (IC, decile spreads, portfolio returns).
See `PHASE2_ISOLATION_STATEMENT` at the bottom.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

FROZEN_SPEC_ID = "phase1_dataset_v1"
FROZEN_SPEC_VERSION = "1.0.0"

# ---- Source dataset ------------------------------------------------------

BAR_SOURCE_NAME = "hfdatalibrary_clean_5min_1391tickers_2026-09-28"
BAR_SOURCE_DIR = "data/hfdatalibrary_clean_5min_1391tickers_2026-09-28"
BAR_SOURCE_MANIFEST_SHA256 = (
    "3b692dd45cce976780bc63cf11b8d8b2ec8f7fda66d4118e2c723b723f026585"
)
BAR_SOURCE_N_FILES = 1388
BAR_SOURCE_TOTAL_BYTES = 8_283_620_000  # ~8.28 GB
BAR_SOURCE_SCHEMA = (
    "per-ticker parquet with columns: datetime (naive ET wall-clock), "
    "Open/High/Low/Close (float64, split-adjusted), Volume (int64), source (str)"
)

# ---- Usable date range and source regime -----------------------------------

DATASET_MIN_DATE = date(2003, 1, 2)   # first meaningful trading day
DATASET_MAX_DATE = date(2026, 9, 24)  # last bar in snapshot
SOURCE_REGIME_BOUNDARY = date(2022, 3, 7)   # pitrading -> iex
SOURCE_REGIME_PRE = "pitrading"    # 2003-01-02 → 2022-03-04 (consolidated)
SOURCE_REGIME_POST = "iex"         # 2022-03-07 → 2026-09-24 (IEX-only feed)
USABLE_START_DATE = date(2003, 1, 2)
USABLE_END_DATE = date(2026, 9, 24)

# ---- PIT S&P 500 membership -----------------------------------------------

PIT_SOURCE_NAME = "fja05680/sp500 (GitHub)"
PIT_SOURCE_URLS = (
    "https://github.com/fja05680/sp500",
    "https://raw.githubusercontent.com/fja05680/sp500/master/S%26P%20500%20Historical%20Components%20%26%20Changes.csv",
    "https://raw.githubusercontent.com/fja05680/sp500/master/sp500_changes_since_2019.csv",
)
PIT_HISTORICAL_SHA256 = "bd2745939a709f31"  # abbreviated; full in pit_report.json
PIT_CHANGES_SHA256 = "1fb4ec6641324ca7"
PIT_COVERAGE_START = date(1996, 1, 2)
PIT_COVERAGE_END = date(2026, 8, 18)  # last event in current snapshot
PIT_TRUSTWORTHY_START = date(2001, 1, 16)  # source-flagged incompleteness before this

# ---- Universe construction rule -------------------------------------------

UNIVERSE_RULE = (
    "Universe(t) = PIT S&P 500 membership on date t (from fja05680 reconstruction) "
    "MINUS Real Estate constituents (sector_etf == 'XLRE' in the sector_map) "
    "INTERSECT HF-available tickers (via TICKER_ALIASES adapter). "
    "Sector ETFs and SPY are NOT part of the ranking cross-section — they are "
    "benchmarks only."
)

# ---- Ticker aliases and rebrand handling ----------------------------------

TICKER_ALIASES_APPLIED = {
    "META": "FB",
    "FI": "FISV",
    "GEN": "NLOK",
}
REBRAND_EVENTS_TREATED_AS_RENAMES = {
    "2022-06-09": ("FB", "META"),
    "2022-11-08": ("NLOK", "GEN"),
}
REBRAND_HANDLING_RULE = (
    "Rebrand events are NOT treated as real index churn. In PIT events they "
    "carry `is_rebrand=True`; downstream must not count them as add/remove. "
    "Adapter TICKER_ALIASES ensures callers can query the canonical modern "
    "symbol and receive the pre-rebrand history from the provider's on-disk "
    "file."
)
KNOWN_TRUNCATED_SERIES = {
    "META (via FB)": "continuous 2012-05-18 → 2022-06-08; sparse & unreliable thereafter",
    "GEN (via NLOK)": "continuous → 2022-11-07; no continuation series in provider",
}
KNOWN_INTERNAL_GAPS = {
    "O":     "192-session gap 2008-02-12 → 2008-11-14",
    "BF-B":  "111-session gap 2026-03-27 → 2026-09-08 (provider issue)",
    "BRK-B": "111-session gap 2026-03-27 → 2026-09-08 (provider issue)",
}
FISV_RENAME_STATUS = (
    "FISV → FI rebrand on 2024-07-11 is NOT recorded in fja05680 changes file; "
    "adapter aliases FI→FISV manually. PIT membership still lists FISV post-2024."
)

# ---- Sector mapping and Real Estate policy --------------------------------

SECTOR_MAP_SOURCE = "phase0/sector_map.py — Wikipedia S&P 500 constituents table (GICS Sector column)"
SECTOR_MAP_SNAPSHOT = "data/phase0_yahoo/sector_map_v1.parquet"
SECTOR_MAP_SHA256 = "adde3c0a0028d08a"
SECTOR_ETFS_USED = ("XLK", "XLF", "XLV", "XLE", "XLY", "XLP", "XLI", "XLB", "XLU", "XLC")
REAL_ESTATE_POLICY = "B"  # exclude Real Estate constituents; see phase1/real_estate_policy.py
REAL_ESTATE_RULE = (
    "Any ticker with sector_map[ticker] == 'XLRE' is FILTERED OUT of the "
    "cross-section at the eligibility layer. Real Estate stocks do not appear "
    "in Phase 2 rankings. Rationale: XLRE is missing from the HF provider "
    "snapshot; excluding preserves the frozen hypothesis spec for 10 of 11 "
    "sectors without introducing external data or specification deviation. "
    "Coverage cost: ~3-6% of PIT observations."
)

# ---- Trading calendar and session rules -----------------------------------

CALENDAR_SOURCE = "pandas_market_calendars (NYSE)"
SESSION_CANONICAL_HOURS_ET = ("09:30", "16:00")
EARLY_CLOSE_SESSIONS_HANDLED = True
EARLY_CLOSE_RULE = (
    "Session length and expected bar count are read from `common.nyse_calendar` "
    "per session date, not assumed. Full sessions = 78 5-min bars (09:30-15:55 "
    "open); 13:00 ET early closes = 42 bars (Black Friday, July 3, Christmas "
    "Eve, occasional others). Decision times whose forward-horizon bar falls "
    "outside session_close are automatically ineligible on that session."
)
BAR_INTERVAL_MIN = 5
TIMEZONE_OF_RECORD = "America/New_York"
STORAGE_TIMEZONE = "UTC"

# ---- Corporate actions -----------------------------------------------------

CORPORATE_ACTIONS_TREATMENT = (
    "Provider ships split-adjusted OHLC. Verified on TSLA 2020-08-31 (5-for-1) "
    "and 2022-08-25 (3-for-1) — bar closes continuous through split boundaries. "
    "Dividend adjustment status unspecified by provider; treated as-is (Phase 4 "
    "will assess whether this is material)."
)

# ---- Missing-bar policy ---------------------------------------------------

MISSING_BAR_POLICY = (
    "Missing 5-min bars are NEVER forward-filled. The reindexer "
    "(phase0.features.reindex_bars_to_grid) generates a full 78-bar grid per "
    "(ticker, session) and fills missing bars with NULL OHLCV. Downstream "
    "return computations propagate the null; eligibility rejects the row."
)

# ---- (ticker, session, decision_time) eligibility rules --------------------

DECISION_TIME_BAR_SEQS = {
    "13:00 ET": 42,  # frozen hypothesis decision time
}
LOOKBACK_BARS_PRIMARY = 3   # 15 minutes
HORIZON_BARS_PRIMARY = 12   # 60 minutes
MIN_COVERAGE_LOOKBACK = 0.85

ELIGIBILITY_RULES = (
    "A (ticker, session, decision_bar) row is eligible iff ALL of: "
    "(a) ticker is in Universe(session) per the PIT + sector filter rule; "
    "(b) ticker's sector ETF has a non-null close on that (session, decision_bar); "
    "(c) trailing-lookback close coverage on the reindexed grid is >= 0.85; "
    "(d) decision-bar close for the ticker is not null; "
    "(e) target-horizon-bar close (decision_bar + 12 bars) is not null — "
    "which automatically drops early-close sessions where 14:00 ET is past close; "
    "(f) ticker is not past its provider truncation date (KNOWN_TRUNCATED_SERIES)."
)

# ---- Coverage requirements ------------------------------------------------

COVERAGE_REQUIREMENTS = (
    "Per-session per-ticker coverage denominator = NYSE calendar's expected_bars "
    "(78 or 42), NOT max-observed heuristic. Per-ticker roll-up separates "
    "mean_coverage_all_sessions (over dataset calendar) from mean_coverage_active "
    "(over ticker's own first-to-last observed session range) — Phase 2 quality "
    "checks use the latter."
)

# ---- Known provider gaps / exceptions -------------------------------------

PROVIDER_GAPS_AND_EXCEPTIONS = (
    "1. XLRE absent — see REAL_ESTATE_POLICY above.\n"
    "2. Recent S&P spinoffs absent from HF: GEHC (2023-01), KVUE (2023-08), "
    "VLTO (2023-09), GEV (2024-04), SOLV (2024-04), WBD (2022-04), KKR, LII, "
    "SMCI, PSKY. Any Phase 2 window containing these dates will not include "
    "these tickers even if PIT lists them. Impact bounded by HF coverage "
    "figures in phase1_pit/hf_coverage_by_year.parquet.\n"
    "3. Pre-2001 PIT is untrustworthy (source note verified).\n"
    "4. Pre-2003 HF data does not exist.\n"
    "5. Bankruptcy-era ticker suffixes (LEHMQ, AAMRQ) may not match HF file "
    "names for pre-2010 delistings.\n"
    "6. Volume series discontinuity at SOURCE_REGIME_BOUNDARY (~20-40x drop as "
    "IEX feed replaces consolidated). Non-issue for the frozen hypothesis "
    "which is price-based; would be an issue for volume-based features.\n"
    "7. Session calendar gap: 3 weekend-spillover sessions (2002-12-07, "
    "2002-12-08, 2009-01-10) in provider files are dropped by the audit."
)

# ---- Phase 2 isolation statement ------------------------------------------

PHASE2_ISOLATION_STATEMENT = (
    "No Phase 2 frozen-hypothesis performance metric (IC, decile spread, "
    "portfolio return, Sharpe, or any downstream signal statistic) was "
    "computed, inspected, or used to inform any value in this Phase 1 "
    "specification. Every parameter here — universe rule, coverage threshold, "
    "eligibility gate, Real Estate policy, source regime handling — was set "
    "on the basis of data integrity and preregistered hypothesis semantics "
    "only. The Phase 0 exploratory window 2026-07-02 → 2026-09-25 remains "
    "embargoed from every Phase 2 partition per the Phase 0F freeze."
)


@dataclass(frozen=True)
class FrozenDatasetSpec:
    spec_id: str = FROZEN_SPEC_ID
    version: str = FROZEN_SPEC_VERSION
    bar_source_name: str = BAR_SOURCE_NAME
    bar_source_manifest_sha256: str = BAR_SOURCE_MANIFEST_SHA256
    usable_start_date: date = USABLE_START_DATE
    usable_end_date: date = USABLE_END_DATE
    source_regime_boundary: date = SOURCE_REGIME_BOUNDARY
    pit_source: str = PIT_SOURCE_NAME
    pit_trustworthy_start: date = PIT_TRUSTWORTHY_START
    universe_rule: str = UNIVERSE_RULE
    real_estate_policy: str = REAL_ESTATE_POLICY
    early_close_handled: bool = EARLY_CLOSE_SESSIONS_HANDLED
    ticker_aliases: tuple = field(
        default_factory=lambda: tuple(sorted(TICKER_ALIASES_APPLIED.items()))
    )
    embargo_start: date = date(2026, 7, 2)
    embargo_end: date = date(2026, 9, 25)


FROZEN = FrozenDatasetSpec()


if __name__ == "__main__":
    import json
    from dataclasses import asdict

    print(f"Phase 1 dataset spec: {FROZEN.spec_id} v{FROZEN.version}")
    print(json.dumps(asdict(FROZEN), indent=2, default=str))
