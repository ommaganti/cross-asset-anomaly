from __future__ import annotations

import pandas as pd


def vix_regime(vix: pd.Series, window: int = 252, min_periods: int = 60) -> pd.Series:
    """Classify each date into low/mid/high vol using trailing VIX terciles (no lookahead)."""
    q33 = vix.shift(1).rolling(window, min_periods=min_periods).quantile(0.33)
    q67 = vix.shift(1).rolling(window, min_periods=min_periods).quantile(0.67)
    regime = pd.Series("mid_vol", index=vix.index, dtype=object)
    regime[vix < q33] = "low_vol"
    regime[vix > q67] = "high_vol"
    regime[q33.isna() | q67.isna()] = pd.NA
    return regime


def realized_vol_regime(returns: pd.Series, vol_window: int = 20, lookback: int = 252) -> pd.Series:
    """Alternative regime: classify by trailing realized vol terciles."""
    rv = returns.rolling(vol_window, min_periods=vol_window // 2).std()
    return vix_regime(rv, window=lookback)


def duration_regime(
    tlt: pd.Series, shy: pd.Series, window: int = 60, threshold: float = 0.02
) -> pd.Series:
    """Long-vs-short duration performance regime — proxy for bull/bear curve moves.

    long_leading  → TLT outperforming SHY (yields falling more at long end OR long end stronger)
    short_leading → SHY outperforming TLT (front-end stronger)
    """
    ratio = tlt / shy
    slope = ratio.pct_change(window)
    r = pd.Series("curve_flat", index=slope.index, dtype=object)
    r[slope > threshold] = "long_leading"
    r[slope < -threshold] = "short_leading"
    r[slope.isna()] = pd.NA
    return r


def usd_trend_regime(
    dxy: pd.Series, window: int = 200, threshold: float = 0.02
) -> pd.Series:
    """USD trend regime: strong / weak / flat based on N-bar return of DXY."""
    slope = dxy / dxy.shift(window) - 1
    r = pd.Series("usd_flat", index=slope.index, dtype=object)
    r[slope > threshold] = "usd_strong"
    r[slope < -threshold] = "usd_weak"
    r[slope.isna()] = pd.NA
    return r


def credit_regime(
    hyg: pd.Series, lqd: pd.Series, window: int = 60, threshold: float = 0.02
) -> pd.Series:
    """Credit risk-appetite regime via HY-vs-IG ratio momentum."""
    ratio = hyg / lqd
    slope = ratio.pct_change(window)
    r = pd.Series("credit_neutral", index=slope.index, dtype=object)
    r[slope > threshold] = "credit_risk_on"
    r[slope < -threshold] = "credit_risk_off"
    r[slope.isna()] = pd.NA
    return r


def build_regime_axes(prices: pd.DataFrame) -> dict[str, pd.Series]:
    """Build the full multi-axis regime dict from a price panel."""
    axes: dict[str, pd.Series] = {}
    if "^VIX" in prices.columns:
        axes["vol"] = vix_regime(prices["^VIX"])
    if "TLT" in prices.columns and "SHY" in prices.columns:
        axes["curve"] = duration_regime(prices["TLT"], prices["SHY"])
    if "DX-Y.NYB" in prices.columns:
        axes["usd"] = usd_trend_regime(prices["DX-Y.NYB"])
    if "HYG" in prices.columns and "LQD" in prices.columns:
        axes["credit"] = credit_regime(prices["HYG"], prices["LQD"])
    return axes
