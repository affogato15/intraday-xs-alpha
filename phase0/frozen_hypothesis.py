"""Phase 0F frozen hypothesis specification.

This is the ONE primary hypothesis that exits Phase 0. Per project rules it
receives exactly one independent confirmatory test in Phase 2 and cannot be
retuned in light of Phase 2 results.

Every value in this module is *frozen*. If you feel the need to change one, you
are starting a new experiment against a new independent dataset — this module
should not be edited.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ---- Hypothesis identity --------------------------------------------------

HYPOTHESIS_ID: str = "phase0f_v1_midday_sector_reversal"

STATEMENT: str = (
    "The sector-relative 15-minute return prior to 13:00 ET predicts the "
    "next 60-minute sector-relative return with a negative cross-sectional "
    "rank IC (mean-reversion). Signal is used long-only: buy the bottom "
    "cross-sectional decile (biggest recent sector-relative losers) at "
    "13:00 ET, hold for 60 minutes, close by 14:00 ET, on liquid U.S. "
    "equities."
)

# ---- Universe -------------------------------------------------------------

UNIVERSE_VERSION: str = "v1"
UNIVERSE_DESCRIPTION: str = (
    "Current-membership S&P 500 (503 tickers as of 2026-09-27 Wikipedia "
    "snapshot) plus SPDR sector ETFs (11) and SPY as benchmarks. Cross-section "
    "for signal ranking = S&P 500 stocks only; SPY and sector ETFs excluded. "
    "This is a survivorship-biased proxy acceptable for Phase 0 hypothesis "
    "discovery. Phase 2 must use a proper PIT universe on the confirmatory "
    "dataset."
)

# ---- Signal ---------------------------------------------------------------

LOOKBACK_MINUTES: int = 15
LOOKBACK_BARS: int = 3  # 15 min / 5 min

DECISION_TIME_ET: str = "13:00"
DECISION_BAR_SEQ: int = 42  # 09:30 ET + 42*5min = 13:00 ET

HORIZON_MINUTES: int = 60
HORIZON_BARS: int = 12  # 60 min / 5 min

# Signal is sector-relative return with beta fixed to 1 (not estimated).
# This is *market/sector-relative*, not a strict idiosyncratic residual.
SIGNAL_DEFINITION: str = (
    "ret_15m_sec_rel = close_t / close_{t-3 bars} - 1 "
    "minus close_{sector_ETF(i), t} / close_{sector_ETF(i), t-3 bars} + 1, "
    "where sector_ETF(i) is the SPDR sector ETF for ticker i's GICS sector "
    "(mapping in phase0/sector_map.py). Both stock and ETF close prices are "
    "on the reindexed 78-bar 09:30-15:55 ET grid, and the L-bar shift is "
    "partitioned by (ticker, session_date_et) so no term crosses overnight."
)

TARGET_DEFINITION: str = (
    "fwd_ret_60m_sec_rel = close_{i, t+12 bars} / close_{i, t} - 1 "
    "minus close_{sector_ETF(i), t+12 bars} / close_{sector_ETF(i), t} + 1, "
    "again partitioned by (ticker, session_date_et)."
)

EXPECTED_DIRECTION: str = "reversal"
EXPECTED_IC_SIGN: str = "negative"

# ---- Eligibility gate -----------------------------------------------------

MIN_COVERAGE: float = 0.85
ELIGIBILITY_RULES: str = (
    "At each (ticker, session, decision_bar) the trailing L/5=3-bar close "
    "coverage must be >= 0.85, the decision-bar close must be present, and "
    "the target-horizon-bar close (12 bars ahead) must be present. Same rule "
    "must hold for the ticker's sector ETF (a null sector-ETF close nulls "
    "the sec_rel columns). PIT membership is enforced implicitly by the "
    "coverage gate — no ticker-level blacklist."
)

# ---- Portfolio v1 ---------------------------------------------------------

PORTFOLIO_SCOPE: str = "long-only"
TOP_N_BUCKET: str = "Q1"  # bottom decile of the (signed) sector-relative signal
TOP_N_NAMES: int = 20
WEIGHTING: str = "equal-weight"
HOLDING_MINUTES: int = 60
PORTFOLIO_DESCRIPTION: str = (
    "At 13:00 ET take the 20 stocks with the lowest ret_15m_sec_rel among "
    "eligible names, equal-weight, hold 60 minutes, close at 14:00 ET. "
    "Long-only. No short leg (out of scope for v1). Q10-Q1 spread is a "
    "diagnostic of ranking quality only and is never realized as a portfolio."
)

# ---- Benchmarks -----------------------------------------------------------

PRIMARY_ECONOMIC_BENCHMARK: str = (
    "Top-20 portfolio's excess return over SPY on the same 13:00 -> 14:00 ET "
    "intraday interval, computed net of realistic execution costs in Phase 4."
)
INVESTMENT_BENCHMARK: str = "SPY buy-and-hold over the same calendar period."

# ---- Confirmatory metrics for Phase 2 -------------------------------------

CONFIRMATORY_PRIMARY_METRIC: str = "Spearman cross-sectional Rank IC of ret_15m_sec_rel vs fwd_ret_60m_sec_rel at 13:00 ET, mean across sessions, block bootstrap CI."
CONFIRMATORY_SUPPORTING_METRICS: tuple[str, ...] = (
    "IC t-statistic with block bootstrap",
    "IC stability across chronological folds",
    "Decile monotonicity (Spearman of decile vs mean forward return)",
    "Q1 mean fwd_ret_60m_sec_rel vs zero",
    "Q1 mean fwd_ret_60m_spy_rel (long-only economic proxy)",
)

# ---- Contamination window (mandatory embargo in Phase 2) -------------------

CONTAMINATION_START_ET: str = "2026-07-02"
CONTAMINATION_END_ET: str = "2026-09-25"
CONTAMINATION_NOTE: str = (
    "These 60 trading sessions are the Yahoo 5-min window used for Phase 0 "
    "exploratory work. They must be embargoed from every Phase 2 partition "
    "(train / validation / WFOOS / locked test). Any metric computed on these "
    "dates is exploratory-contaminated and cannot count as independent "
    "confirmation."
)

# ---- Preregistration guarantees -------------------------------------------

@dataclass(frozen=True)
class FrozenSpec:
    hypothesis_id: str = HYPOTHESIS_ID
    statement: str = STATEMENT
    universe_version: str = UNIVERSE_VERSION
    lookback_minutes: int = LOOKBACK_MINUTES
    decision_time_et: str = DECISION_TIME_ET
    horizon_minutes: int = HORIZON_MINUTES
    signal_definition: str = SIGNAL_DEFINITION
    target_definition: str = TARGET_DEFINITION
    expected_direction: str = EXPECTED_DIRECTION
    expected_ic_sign: str = EXPECTED_IC_SIGN
    min_coverage: float = MIN_COVERAGE
    portfolio_scope: str = PORTFOLIO_SCOPE
    top_n_names: int = TOP_N_NAMES
    holding_minutes: int = HOLDING_MINUTES
    top_n_bucket: str = TOP_N_BUCKET
    primary_metric: str = CONFIRMATORY_PRIMARY_METRIC
    supporting_metrics: tuple[str, ...] = CONFIRMATORY_SUPPORTING_METRICS
    contamination_start_et: str = CONTAMINATION_START_ET
    contamination_end_et: str = CONTAMINATION_END_ET

    def to_dict(self) -> dict:
        return {
            "hypothesis_id": self.hypothesis_id,
            "statement": self.statement,
            "universe_version": self.universe_version,
            "lookback_minutes": self.lookback_minutes,
            "decision_time_et": self.decision_time_et,
            "horizon_minutes": self.horizon_minutes,
            "signal_definition": self.signal_definition,
            "target_definition": self.target_definition,
            "expected_direction": self.expected_direction,
            "expected_ic_sign": self.expected_ic_sign,
            "min_coverage": self.min_coverage,
            "portfolio_scope": self.portfolio_scope,
            "top_n_names": self.top_n_names,
            "holding_minutes": self.holding_minutes,
            "top_n_bucket": self.top_n_bucket,
            "primary_metric": self.primary_metric,
            "supporting_metrics": list(self.supporting_metrics),
            "contamination_window_et": [
                self.contamination_start_et,
                self.contamination_end_et,
            ],
        }


FROZEN = FrozenSpec()

if __name__ == "__main__":
    import json
    print(json.dumps(FROZEN.to_dict(), indent=2))
