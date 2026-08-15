from __future__ import annotations

import os
import pandas as pd
import numpy as np

from .universe import all_tickers


CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cache")


def _cache_path(start: str, end: str | None) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    end_part = end or "latest"
    return os.path.join(CACHE_DIR, f"prices_{start}_{end_part}.parquet")


def load_prices(start: str = "2012-01-01", end: str | None = None, use_cache: bool = True) -> pd.DataFrame:
    """Download daily adjusted closes for the full universe via yfinance."""
    cache = _cache_path(start, end)
    if use_cache and os.path.exists(cache):
        return pd.read_parquet(cache)

    import yfinance as yf  # lazy import so module loads without network

    df = yf.download(
        all_tickers(),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    # yfinance returns a multi-index when multiple tickers are requested
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs("Close", axis=1, level=1)
    else:
        df = df[["Close"]].rename(columns={"Close": all_tickers()[0]})

    df = df.dropna(how="all")
    df.index = pd.to_datetime(df.index)

    if use_cache:
        df.to_parquet(cache)

    return df


def align_calendar(prices: pd.DataFrame, reference: str = "SPY") -> pd.DataFrame:
    """Reindex to the reference asset's trading calendar (handles cross-class holidays)."""
    if reference not in prices.columns:
        return prices
    ref_dates = prices[reference].dropna().index
    return prices.reindex(ref_dates).ffill(limit=1)


def filter_stale(prices: pd.DataFrame, max_consecutive: int = 3) -> pd.DataFrame:
    """Mask prices that haven't moved for >= max_consecutive bars (illiquid / stale feed)."""
    changed = prices.diff().abs() > 0
    rolling_changed = changed.rolling(max_consecutive).sum()
    stale = (rolling_changed == 0) & prices.notna()
    return prices.mask(stale)


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1))


def vol_adjusted_returns(returns: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Standardize returns by trailing realized vol so correlations aren't driven by vol regimes."""
    rv = returns.rolling(window, min_periods=window // 2).std()
    return (returns / rv).replace([np.inf, -np.inf], np.nan)
