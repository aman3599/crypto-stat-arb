"""
data.py — yfinance OHLC data fetcher with parquet caching.

Uses yfinance with daily ("1d") bars — the only granularity that covers the full
2023-present history on Yahoo Finance.  Hourly data is hard-limited to ~730 days,
which makes it unusable for the 2023-2024 training window.

Assets: BTC, ETH, SOL, ADA, XRP, DOGE, DOT, AVAX, LINK
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

# ── Config ─────────────────────────────────────────────────────────────────

CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

TICKER_MAP = {
    "BTC":  "BTC-USD",
    "ETH":  "ETH-USD",
    "SOL":  "SOL-USD",
    "ADA":  "ADA-USD",
    "XRP":  "XRP-USD",
    "DOGE": "DOGE-USD",
    "DOT":  "DOT-USD",
    "AVAX": "AVAX-USD",
    "LINK": "LINK-USD",
    "LTC":  "LTC-USD",
    "BCH":  "BCH-USD",
    "ATOM": "ATOM-USD",
    "NEAR": "NEAR-USD",
    "UNI":  "UNI7083-USD",   # UNI-USD froze 2025-04-17; Uniswap's live Yahoo symbol is UNI7083
    "AAVE": "AAVE-USD",
    "FIL":  "FIL-USD",
}

# Kept as int constants so downstream code that references them still works.
# The pipeline now runs on daily bars (INTERVAL_1D); INTERVAL_4H is an alias.
INTERVAL_1D = 1440   # daily candles (minutes)
INTERVAL_4H = INTERVAL_1D   # alias — Yahoo Finance has no native 4H interval


# ── Core fetch ─────────────────────────────────────────────────────────────

def fetch_ohlc(
    ticker:   str,
    start:    str = "2023-01-01",
    end:      Optional[str] = None,
    interval: int = INTERVAL_1D,
    refresh:  bool = False,
) -> pd.DataFrame:
    """
    Fetch daily OHLC data for a single ticker with parquet caching.

    Yahoo Finance provides unlimited-history daily bars for crypto assets.
    (1H bars are limited to the last ~730 days and cannot cover 2023 training data.)
    """
    if ticker not in TICKER_MAP:
        raise ValueError(f"Unknown ticker '{ticker}'. Valid: {list(TICKER_MAP.keys())}")

    yf_symbol  = TICKER_MAP[ticker]
    cache_path = CACHE_DIR / f"{ticker}_{interval}.parquet"

    # ── Cache hit ──────────────────────────────────────────────────────────
    if not refresh and cache_path.exists():
        print(f"[data] Loading {ticker} from cache: {cache_path}")
        df = pd.read_parquet(cache_path)
        return _slice(df, start, end)

    # ── Live fetch ─────────────────────────────────────────────────────────
    import yfinance as yf

    print(f"[data] Fetching {ticker} ({yf_symbol}) from yfinance (daily)...")
    end_str = end if end else pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")

    ticker_obj = yf.Ticker(yf_symbol)
    df = ticker_obj.history(
        start=start,
        end=end_str,
        interval="1d",
        auto_adjust=True,
    )

    if df.empty:
        raise RuntimeError(
            f"No data returned for {ticker} ({yf_symbol}). "
            "Check the symbol and date range."
        )

    # Normalize column names
    df.columns = [c.lower() for c in df.columns]

    # Ensure UTC-aware index
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    df = df[keep]

    df.to_parquet(cache_path)
    print(f"[data] Cached {len(df):,} daily bars for {ticker} → {cache_path}")

    return _slice(df, start, end)


def _slice(df: pd.DataFrame, start: str, end: Optional[str]) -> pd.DataFrame:
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts   = pd.Timestamp(end, tz="UTC") if end else None
    mask = df.index >= start_ts
    if end_ts is not None:
        mask &= df.index <= end_ts
    return df[mask]


# ── Multi-asset loader ──────────────────────────────────────────────────────

def load_prices(
    tickers:   Optional[list] = None,
    start:     str = "2023-01-01",
    end:       Optional[str] = None,
    interval:  int = INTERVAL_1D,
    price_col: str = "close",
    refresh:   bool = False,
) -> pd.DataFrame:
    """
    Load close prices for all tickers into a single aligned DataFrame.
    """
    if tickers is None:
        tickers = list(TICKER_MAP.keys())

    frames = {}
    for ticker in tickers:
        try:
            df = fetch_ohlc(ticker, start=start, end=end, interval=interval, refresh=refresh)
            if df.empty:
                print(f"[data] WARNING: {ticker} returned empty DataFrame — skipping.")
                continue
            frames[ticker] = df[price_col]
        except Exception as e:
            print(f"[data] WARNING: Failed to fetch {ticker}: {e}")

    if not frames:
        raise RuntimeError("Failed to load any price data.")

    prices = pd.DataFrame(frames)

    # Forward-fill small gaps (weekends/holidays), drop rows where ALL tickers are NaN
    prices = prices.ffill().dropna(how="all")

    n_nan = prices.isnull().sum().sum()
    if n_nan > 0:
        bad = prices.columns[prices.isnull().any()].tolist()
        print(f"[data] WARNING: NaNs remain in {bad} after ffill — dropping affected rows.")
        prices = prices.dropna()

    if prices.empty:
        raise RuntimeError(
            "prices DataFrame is empty after alignment. "
            "This usually means the tickers have non-overlapping date ranges."
        )

    print(f"[data] Loaded {len(prices):,} aligned daily bars for {list(prices.columns)}")
    print(f"[data] Date range: {prices.index[0].date()} → {prices.index[-1].date()}")
    return prices


def load_returns(
    tickers:  Optional[list] = None,
    start:    str = "2023-01-01",
    end:      Optional[str] = None,
    interval: int = INTERVAL_1D,
    refresh:  bool = False,
) -> pd.DataFrame:
    """Load log returns for all tickers."""
    prices = load_prices(tickers=tickers, start=start, end=end, interval=interval, refresh=refresh)
    return np.log(prices / prices.shift(1)).dropna()


# ── Sanity check ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running data.py sanity check...\n")
    prices = load_prices(start="2023-01-01", end="2024-12-31", interval=INTERVAL_1D)
    print("\nSample prices (last 5 rows):")
    print(prices.tail())
    print("\nShape:", prices.shape)
    print("Missing values:", prices.isnull().sum().to_dict())
