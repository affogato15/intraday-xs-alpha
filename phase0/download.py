"""Phase 0 Yahoo 5-min bar downloader.

Fetches 5-minute OHLCV bars for the Phase 0 seed universe and writes one Parquet
file per ticker plus a JSON manifest that records the exact fetched window.

The manifest is the *contamination record* for Phase 2: the min/max timestamps in
each ticker file must be embargoed from any independent confirmatory test.

No forward-filling. Missing bars stay missing. Timezone-of-record is UTC on disk;
America/New_York is materialized as a helper column for downstream session logic.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from common.paths import PHASE0_DATA, ensure_dirs
from phase0.universe import load_snapshot

BARS_DIR = PHASE0_DATA / "bars_5m"
MANIFEST_PATH = PHASE0_DATA / "download_manifest_5m.json"

BAR_SCHEMA = {
    "ticker": pl.Utf8,
    "ts_utc": pl.Datetime(time_unit="us", time_zone="UTC"),
    "ts_et": pl.Datetime(time_unit="us", time_zone="America/New_York"),
    "session_date_et": pl.Date,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Int64,
}


@dataclass
class TickerFetchResult:
    ticker: str
    ok: bool
    n_bars: int = 0
    min_ts_utc: str | None = None
    max_ts_utc: str | None = None
    file: str | None = None
    error: str | None = None


@dataclass
class DownloadManifest:
    universe_version: str
    source: str
    interval: str
    period: str
    fetched_at_utc: str
    n_tickers_requested: int
    n_tickers_ok: int
    n_tickers_failed: int
    data_start_utc: str | None
    data_end_utc: str | None
    per_ticker: dict[str, dict] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def _yahoo_df_to_polars(df, ticker: str) -> pl.DataFrame | None:
    """Convert a yfinance pandas DataFrame (single ticker) to the canonical schema.

    Drops rows with any NaN OHLCV values. Never forward-fills.
    """
    import pandas as pd

    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns={c: c.lower() for c in df.columns})
    needed = {"open", "high", "low", "close", "volume"}
    if not needed.issubset(df.columns):
        return None

    df = df[["open", "high", "low", "close", "volume"]].copy()
    df = df.dropna(how="any")
    if df.empty:
        return None

    idx = df.index
    if getattr(idx, "tz", None) is None:
        idx = idx.tz_localize("UTC")
    ts_utc = idx.tz_convert("UTC")
    ts_et = idx.tz_convert("America/New_York")

    out = pl.DataFrame(
        {
            "ticker": [ticker] * len(df),
            "ts_utc": ts_utc.to_pydatetime().tolist(),
            "ts_et": ts_et.to_pydatetime().tolist(),
            "session_date_et": [d.date() for d in ts_et.to_pydatetime()],
            "open": df["open"].astype(float).tolist(),
            "high": df["high"].astype(float).tolist(),
            "low": df["low"].astype(float).tolist(),
            "close": df["close"].astype(float).tolist(),
            "volume": df["volume"].astype("int64").tolist(),
        },
        schema=BAR_SCHEMA,
    )
    return out.sort("ts_utc").unique(subset=["ts_utc"], keep="last", maintain_order=True)


def _fetch_batch(tickers: list[str], period: str, interval: str, max_retries: int = 3):
    """Call yfinance.download with retries. Returns pandas DataFrame or None."""
    import yfinance as yf

    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            df = yf.download(
                tickers=tickers,
                period=period,
                interval=interval,
                group_by="ticker",
                auto_adjust=False,
                prepost=False,
                progress=False,
                threads=True,
            )
            return df
        except Exception as exc:
            last_err = exc
            time.sleep(2 * attempt)
    raise RuntimeError(f"yfinance download failed after {max_retries} retries: {last_err}")


def _split_batch(batch_df, requested_tickers: list[str]) -> dict[str, "pd.DataFrame"]:
    """Given a yfinance batch DataFrame (group_by='ticker'), split into per-ticker frames."""
    import pandas as pd

    out: dict[str, pd.DataFrame] = {}
    if batch_df is None or batch_df.empty:
        return out
    if isinstance(batch_df.columns, pd.MultiIndex):
        level0 = batch_df.columns.get_level_values(0).unique().tolist()
        for tk in requested_tickers:
            if tk in level0:
                out[tk] = batch_df[tk]
    else:
        if len(requested_tickers) == 1:
            out[requested_tickers[0]] = batch_df
    return out


def _write_bars(df: pl.DataFrame, ticker: str) -> Path:
    BARS_DIR.mkdir(parents=True, exist_ok=True)
    path = BARS_DIR / f"{ticker}_5m.parquet"
    df.write_parquet(path)
    return path


def download_5m(
    universe_version: str = "v1",
    period: str = "60d",
    interval: str = "5m",
    batch_size: int = 25,
    inter_batch_sleep_s: float = 1.0,
    tickers: list[str] | None = None,
) -> DownloadManifest:
    ensure_dirs()
    if tickers is None:
        snap = load_snapshot(universe_version)
        tickers = list(snap.all_tickers)

    per_ticker: dict[str, TickerFetchResult] = {}
    global_min: datetime | None = None
    global_max: datetime | None = None

    for start in range(0, len(tickers), batch_size):
        chunk = tickers[start : start + batch_size]
        print(f"[{start + 1:>4}/{len(tickers)}] fetching {len(chunk)} tickers...")
        try:
            batch_df = _fetch_batch(chunk, period=period, interval=interval)
        except Exception as exc:
            for tk in chunk:
                per_ticker[tk] = TickerFetchResult(ticker=tk, ok=False, error=str(exc))
            time.sleep(inter_batch_sleep_s)
            continue

        splits = _split_batch(batch_df, chunk)
        for tk in chunk:
            raw = splits.get(tk)
            df = _yahoo_df_to_polars(raw, tk) if raw is not None else None
            if df is None or df.is_empty():
                per_ticker[tk] = TickerFetchResult(ticker=tk, ok=False, error="no_data")
                continue
            path = _write_bars(df, tk)
            tmin = df["ts_utc"].min()
            tmax = df["ts_utc"].max()
            if global_min is None or tmin < global_min:
                global_min = tmin
            if global_max is None or tmax > global_max:
                global_max = tmax
            per_ticker[tk] = TickerFetchResult(
                ticker=tk,
                ok=True,
                n_bars=df.height,
                min_ts_utc=tmin.isoformat(),
                max_ts_utc=tmax.isoformat(),
                file=str(path.relative_to(PHASE0_DATA.parent)),
            )
        time.sleep(inter_batch_sleep_s)

    manifest = DownloadManifest(
        universe_version=universe_version,
        source="yahoo",
        interval=interval,
        period=period,
        fetched_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        n_tickers_requested=len(tickers),
        n_tickers_ok=sum(1 for r in per_ticker.values() if r.ok),
        n_tickers_failed=sum(1 for r in per_ticker.values() if not r.ok),
        data_start_utc=global_min.isoformat() if global_min else None,
        data_end_utc=global_max.isoformat() if global_max else None,
        per_ticker={tk: asdict(r) for tk, r in per_ticker.items()},
    )
    MANIFEST_PATH.write_text(manifest.to_json())
    return manifest


def _parse_args():
    p = argparse.ArgumentParser(description="Phase 0 Yahoo 5-min downloader")
    p.add_argument("--universe-version", default="v1")
    p.add_argument("--period", default="60d", help="yfinance period, e.g. '60d', '30d'")
    p.add_argument("--interval", default="5m")
    p.add_argument("--batch-size", type=int, default=25)
    p.add_argument("--sleep", type=float, default=1.0, dest="inter_batch_sleep_s")
    p.add_argument("--limit", type=int, default=None,
                   help="Debug: cap to first N tickers of the universe.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    tickers_override: list[str] | None = None
    if args.limit is not None:
        snap = load_snapshot(args.universe_version)
        tickers_override = list(snap.all_tickers)[: args.limit]
        print(f"[DEBUG] limiting to first {len(tickers_override)} of "
              f"{len(snap.all_tickers)} tickers")
    manifest = download_5m(
        universe_version=args.universe_version,
        period=args.period,
        interval=args.interval,
        batch_size=args.batch_size,
        inter_batch_sleep_s=args.inter_batch_sleep_s,
        tickers=tickers_override,
    )
    print("---")
    print(f"OK: {manifest.n_tickers_ok}/{manifest.n_tickers_requested}")
    print(f"Failed: {manifest.n_tickers_failed}")
    print(f"Range UTC: {manifest.data_start_utc}  →  {manifest.data_end_utc}")
    print(f"Manifest:  {MANIFEST_PATH}")
