from __future__ import annotations

import numpy as np
import pandas as pd


def zscore_matrix(metric_df: pd.DataFrame, lookback: int = 756, min_periods: int = 252) -> pd.DataFrame:
    """Z-score each column against its trailing distribution. Shape preserved."""
    past = metric_df.shift(1)
    m = past.rolling(lookback, min_periods=min_periods).mean()
    s = past.rolling(lookback, min_periods=min_periods).std()
    return (metric_df - m) / s


def systemic_stress_index(
    z_matrices: dict[str, pd.DataFrame], threshold: float = 2.5
) -> pd.DataFrame:
    """Count concurrent pair breaks per date across all metrics — the joint-anomaly signal.

    Distinguishes 'one regime-shift event with many pair breaks' from 'independent noise'.
    """
    rows = {}
    total_tests = 0
    for name, z in z_matrices.items():
        if z.empty:
            continue
        cnt = (z.abs() >= threshold).sum(axis=1).astype(float)
        rows[f"{name}_breaks"] = cnt
        total_tests += z.shape[1]
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, axis=1)
    out.columns = list(rows.keys())
    out["total_breaks"] = out.sum(axis=1)
    out["pct_universe_broken"] = 100.0 * out["total_breaks"] / max(total_tests, 1)
    return out


def stress_regime(
    stress_series: pd.Series, calm_q: float = 0.6, extreme_q: float = 0.9,
    min_periods: int = 252,
) -> pd.Series:
    """Classify each date into calm / elevated / extreme based on trailing-quantile thresholds."""
    past = stress_series.shift(1)
    low = past.expanding(min_periods).quantile(calm_q)
    high = past.expanding(min_periods).quantile(extreme_q)
    out = pd.Series("calm", index=stress_series.index, dtype=object)
    out[stress_series > low] = "elevated"
    out[stress_series > high] = "extreme"
    out[low.isna()] = pd.NA
    return out


def stress_zscore(stress_series: pd.Series, lookback: int = 756, min_periods: int = 252) -> pd.Series:
    past = stress_series.shift(1)
    m = past.rolling(lookback, min_periods=min_periods).mean()
    s = past.rolling(lookback, min_periods=min_periods).std()
    return (stress_series - m) / s
