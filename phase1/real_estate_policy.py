"""Phase 1 Real Estate benchmark policy — decision module.

Documents the analysis behind the Real Estate sector policy decision for the
Phase 1 dataset. Produces a versioned artifact with the coverage impact of each
option so the decision is auditable independently of Phase 2 signal outcomes.

    IMPORTANT: this module computes coverage / universe impact metrics only.
    It does NOT compute IC, decile returns, or any hypothesis performance
    number. The benchmark choice must be independent of signal outcome.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

from common.paths import PROJECT_ROOT, ensure_dirs

PIT_DIR = PROJECT_ROOT / "data" / "phase1_pit"
PHASE1_DIR = PROJECT_ROOT / "data" / "phase1"
POLICY_REPORT_PATH = PHASE1_DIR / "real_estate_policy_report.json"

SECTOR_MAP_PATH = PROJECT_ROOT / "data" / "phase0_yahoo" / "sector_map_v1.parquet"
MEMBERSHIP_DAILY_PATH = PIT_DIR / "membership_daily.parquet"
HF_INVENTORY_PATH = PHASE1_DIR / "hf_inventory.parquet"
HF_DATASET_DIR = PROJECT_ROOT / "data" / "hfdatalibrary_clean_5min_1391tickers_2026-09-28"

TRUSTWORTHY_START = date(2001, 1, 16)
BEST_HF_WINDOW = (date(2011, 1, 1), date(2022, 12, 31))
XLRE_INCEPTION = date(2015, 10, 8)


def _coverage_impact_stats() -> dict:
    sector_map = pl.read_parquet(SECTOR_MAP_PATH)
    re_tickers = set(
        sector_map.filter(pl.col("sector_etf") == "XLRE")["ticker"].to_list()
    )

    daily = pl.read_parquet(MEMBERSHIP_DAILY_PATH).with_columns(
        pl.col("ticker").is_in(list(re_tickers)).alias("is_real_estate")
    )

    def _window_stats(start: date, end: date | None, label: str) -> dict:
        window = daily.filter(pl.col("date") >= start)
        if end is not None:
            window = window.filter(pl.col("date") <= end)
        total = window.height
        re = int(window["is_real_estate"].sum())
        return {
            "label": label,
            "start": start.isoformat(),
            "end": end.isoformat() if end else None,
            "total_observations": total,
            "real_estate_observations": re,
            "real_estate_fraction": (re / total) if total else 0.0,
        }

    by_year = (
        daily.with_columns(pl.col("date").dt.year().alias("year"))
        .group_by(["year", "date"])
        .agg(
            pl.col("is_real_estate").sum().alias("re_members"),
            pl.len().alias("total_members"),
        )
        .group_by("year")
        .agg(
            pl.col("re_members").median().cast(pl.Int64).alias("median_re_members"),
            pl.col("total_members").median().cast(pl.Int64).alias("median_total_members"),
            (pl.col("re_members") / pl.col("total_members"))
            .median()
            .alias("median_re_frac"),
        )
        .sort("year")
    )

    return {
        "real_estate_tickers_current_sp500": sorted(re_tickers),
        "n_re_tickers_current_sp500": len(re_tickers),
        "trustworthy_window": _window_stats(TRUSTWORTHY_START, None, "trustworthy (2001-01-16+)"),
        "best_hf_window": _window_stats(BEST_HF_WINDOW[0], BEST_HF_WINDOW[1], "best HF coverage 2011-2022"),
        "post_gics_re_carveout": _window_stats(date(2016, 9, 1), None, "post GICS RE carve-out (2016-09-01+)"),
        "median_re_members_by_year": by_year.to_dicts(),
    }


def _hf_benchmark_availability() -> dict:
    """What real-estate ETF benchmarks are actually available in the HF dataset."""
    candidates = ["XLRE", "IYR", "VNQ", "RWR", "REM", "USRT", "REZ", "SCHH", "FREL"]
    out = {}
    for etf in candidates:
        path = HF_DATASET_DIR / f"{etf}_clean_5min.parquet"
        if path.exists():
            df = pl.read_parquet(path, columns=["datetime"])
            out[etf] = {
                "in_hf": True,
                "first_date": str(df["datetime"].min().date()),
                "last_date": str(df["datetime"].max().date()),
                "n_bars": df.height,
            }
        else:
            out[etf] = {"in_hf": False}
    return out


def _hf_re_ticker_coverage() -> dict:
    sector_map = pl.read_parquet(SECTOR_MAP_PATH)
    re_tickers = set(
        sector_map.filter(pl.col("sector_etf") == "XLRE")["ticker"].to_list()
    )
    inv = pl.read_parquet(HF_INVENTORY_PATH)
    hf_tickers = set(inv["ticker"].to_list())
    return {
        "re_tickers_total": len(re_tickers),
        "re_tickers_in_hf": sorted(re_tickers & hf_tickers),
        "re_tickers_missing_from_hf": sorted(re_tickers - hf_tickers),
    }


def _option_analyses() -> list[dict]:
    return [
        {
            "option": "A",
            "label": "use XLRE from a compatible external source",
            "external_data_required": True,
            "candidate_sources_reviewed": [
                {
                    "source": "Yahoo Finance",
                    "cost": "free",
                    "5m_history": "rolling 60 days only",
                    "verdict": "unusable for 2015-2022 backfill",
                },
                {
                    "source": "Alpaca free tier",
                    "cost": "free",
                    "5m_history": "2016+ IEX feed only",
                    "verdict": (
                        "same IEX-only regime as HF post-2022; would still leave "
                        "pre-2016 gap; introduces third feed source with unclear "
                        "adjustment/timestamp semantics"
                    ),
                },
                {
                    "source": "FirstRateData",
                    "cost": "one-shot ~$99 (subscription for updates)",
                    "5m_history": "10 years back",
                    "verdict": (
                        "would work but is paid; unknown alignment with HF's "
                        "pitrading/iex composite; corporate-action policy differs; "
                        "no way to verify bar-boundary convention matches HF"
                    ),
                },
                {
                    "source": "Barchart Premier",
                    "cost": "monthly subscription",
                    "5m_history": "back to 2009",
                    "verdict": "subscription, adjustment/timestamp alignment risk",
                },
                {
                    "source": "Polygon / Databento",
                    "cost": "paid",
                    "5m_history": "deep",
                    "verdict": "high quality but paid, and cross-source risk remains",
                },
            ],
            "data_integrity_concerns": [
                "Splices a 3rd source into the dataset — timestamp convention (bar start vs bar end), holiday handling, corporate-action treatment, and last-print policy may differ from HF.",
                "Pre-XLRE-inception (before 2015-10-08) still leaves Real Estate without a sector benchmark; Option A does NOT solve pre-2015 windows.",
                "Verification of alignment requires per-bar comparison against HF on overlapping periods — non-trivial and cannot fully rule out subtle discrepancies.",
                "Free reliable source for 5m XLRE 2015-2022 does not exist per web search (2026-09-28).",
            ],
            "spec_impact": "preserves frozen hypothesis spec exactly (post-2015-10-08 only)",
        },
        {
            "option": "B",
            "label": "exclude Real Estate constituents",
            "external_data_required": False,
            "coverage_impact": {
                "trustworthy_2001plus_fraction_lost": "3.07% of PIT observations",
                "best_hf_window_2011_2022_fraction_lost": "3.84%",
                "post_gics_2016plus_fraction_lost": "~5.2%",
                "post_2022_fraction_lost": "~5.8% (26-29 of 502 daily members)",
            },
            "data_integrity_concerns": [
                "None — 100% internal to HF dataset; no external splice; no timestamp/adjustment ambiguity.",
            ],
            "spec_impact": (
                "hypothesis spec applies to 10 of 11 sectors (all sectors except "
                "Real Estate); explicitly documented scope narrowing rather than "
                "spec change; cross-sectional signal at 13:00 ET evaluated on "
                "~476 tickers per session instead of ~502"
            ),
            "additional_notes": [
                "XLRE launched 2015-10-08 — pre-2015 RE stocks classified as Financials in GICS anyway; Option B removes an ambiguity across the GICS 2016 carve-out.",
                "Reduced cross-section increases per-timestamp IC standard error by roughly sqrt(502/476) ≈ 1.027, i.e. ~2.7% larger — negligible for statistical power.",
            ],
        },
        {
            "option": "C",
            "label": "use IYR as an explicitly documented specification deviation",
            "external_data_required": False,
            "coverage_impact": {
                "note": "Preserves all Real Estate stocks in cross-section (0% loss)",
            },
            "spec_impact": (
                "SIGNIFICANT — sector-relative representation for Real Estate stocks "
                "becomes 'stock minus IYR' instead of 'stock minus XLRE'. IYR tracks "
                "the Dow Jones U.S. Real Estate Capped Index (~85 constituents "
                "including mid/small-cap REITs, mortgage REITs, and real estate "
                "services), a different universe from XLRE's ~30 S&P 500 REITs. "
                "The typical XLRE/IYR daily-return correlation is ~0.95-0.98 (high "
                "but not identical); residuals from the two benchmarks are not "
                "interchangeable at the 5-minute cross-sectional level."
            ),
            "data_integrity_concerns": [
                "None on the data-integrity axis (same HF source, deep history 2003+).",
                "BUT: this is a *specification deviation*. The frozen Phase 0F hypothesis names 'SPDR sector ETF' as the β=1 sector benchmark. Substituting IYR is a rewrite of that clause, not a mere technical fill-in.",
            ],
            "additional_notes": [
                "If C is chosen, Phase 2 confirmation is testing a DIFFERENT frozen spec than what was preregistered.",
                "Recording IYR-based Phase 2 results as confirmation of the Phase 0F hypothesis would be misleading.",
            ],
        },
    ]


def run() -> dict:
    ensure_dirs()
    PHASE1_DIR.mkdir(parents=True, exist_ok=True)

    coverage = _coverage_impact_stats()
    benchmarks = _hf_benchmark_availability()
    re_hf_cov = _hf_re_ticker_coverage()
    options = _option_analyses()

    recommendation = {
        "chosen": "B",
        "label": "exclude Real Estate constituents",
        "rationale": (
            "Purely on data-integrity grounds: Option B is the only choice that "
            "introduces no external data (Option A) and no specification deviation "
            "(Option C). The coverage impact is small — 3.07% of the PIT universe "
            "in the trustworthy window (2001+), 3.84% in the best HF-coverage "
            "window (2011-2022), and ~5.8% in the modern post-2022 regime. "
            "Statistical power loss is negligible (~2.7% larger per-timestamp IC "
            "standard error). The frozen Phase 0F hypothesis remains applied "
            "exactly as written to 10 of 11 sectors; the exclusion is a documented "
            "scope narrowing, not a spec change."
        ),
        "operational_rule": (
            "Filter out any (ticker, session) row where sector_map[ticker] == "
            "'XLRE'. Applied at the eligibility layer inside phase1.eligibility "
            "before the cross-sectional ranking step. Never derive Real Estate "
            "sector-relative returns; those stocks are simply absent from Phase 2 "
            "cross-sections."
        ),
        "consequences_documented_for_phase_2": [
            "Cross-section size at each decision time drops by ~26-29 names post-2016.",
            "Real Estate-specific reversal (if any) will not be evaluated by Phase 2. This is a scope limitation acknowledged upfront, not a survivorship issue.",
            "If a future Phase 10 research cycle wants to test Real Estate reversal, that requires an independent XLRE data source and a NEW preregistered hypothesis.",
        ],
    }

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": (
            "Resolve Phase 1 Real Estate sector benchmark policy on data-integrity "
            "grounds before freezing the Phase 1 dataset spec. No IC, decile, or "
            "signal-performance metric was computed."
        ),
        "phase_0F_frozen_spec_reference": (
            "frozen hypothesis names 'SPDR sector ETF' as the β=1 sector benchmark; "
            "XLRE was the intended Real Estate benchmark"
        ),
        "xlre_status_in_hf": "MISSING from provider snapshot 2026-09-28",
        "xlre_inception": XLRE_INCEPTION.isoformat(),
        "trustworthy_start_pit": TRUSTWORTHY_START.isoformat(),
        "coverage_impact": coverage,
        "hf_benchmark_availability": benchmarks,
        "hf_real_estate_ticker_coverage": re_hf_cov,
        "option_analyses": options,
        "recommendation": recommendation,
    }
    POLICY_REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
    return report


if __name__ == "__main__":
    report = run()
    print("=== Real Estate policy analysis ===")
    print(f"XLRE in HF: {report['xlre_status_in_hf']}")
    print(f"IYR in HF:  {report['hf_benchmark_availability']['IYR']}")
    print(f"\nCoverage impact of excluding Real Estate:")
    for w in ("trustworthy_window", "best_hf_window", "post_gics_re_carveout"):
        w_data = report["coverage_impact"][w]
        print(f"  {w_data['label']:<45} → {100*w_data['real_estate_fraction']:.2f}%")

    print(f"\nHF Real Estate ticker coverage:")
    print(f"  in HF:  {len(report['hf_real_estate_ticker_coverage']['re_tickers_in_hf'])}")
    print(f"  missing: {report['hf_real_estate_ticker_coverage']['re_tickers_missing_from_hf']}")

    print(f"\n=== Recommendation: Option {report['recommendation']['chosen']} — "
          f"{report['recommendation']['label']} ===")
    print(f"\n{report['recommendation']['rationale']}")
    print(f"\nOperational rule: {report['recommendation']['operational_rule']}")
    print(f"\nReport: {POLICY_REPORT_PATH}")
