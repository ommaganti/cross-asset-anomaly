from __future__ import annotations

import numpy as np
import pandas as pd


def zscore_vs_history(series: pd.Series, lookback: int = 756, min_periods: int = 252) -> pd.Series:
    """Z-score each observation against its own trailing distribution (excluding the current point)."""
    past = series.shift(1)
    m = past.rolling(lookback, min_periods=min_periods).mean()
    s = past.rolling(lookback, min_periods=min_periods).std()
    return (series - m) / s


def zscore_by_regime(
    series: pd.Series, regime: pd.Series, min_obs: int = 60
) -> pd.Series:
    """Z-score within the matching regime's expanding history (no lookahead)."""
    out = pd.Series(np.nan, index=series.index, dtype=float)
    aligned = pd.concat([series.rename("v"), regime.rename("r")], axis=1)
    for r, sub in aligned.groupby("r"):
        v = sub["v"]
        past = v.shift(1)
        m = past.expanding(min_obs).mean()
        s = past.expanding(min_obs).std()
        z = (v - m) / s
        out.loc[v.index] = z
    return out


def persistence_filter(flag: pd.Series, min_bars: int = 3) -> pd.Series:
    """Keep alert only if the flag has been True for >= min_bars consecutive bars."""
    rolled = flag.fillna(False).astype(int).rolling(min_bars).sum()
    return rolled >= min_bars


def detect_anomalies(
    metric: pd.Series,
    regime_axes: dict[str, pd.Series] | pd.Series | None,
    z_threshold: float = 2.5,
    persistence: int = 3,
    lookback: int = 756,
    regime_threshold_frac: float = 0.7,
) -> pd.DataFrame:
    """Build an anomaly table.

    regime_axes may be a single Series (back-compat) or a dict of name -> Series.
    Multi-axis: require the break to be significant under EVERY regime lens, at a
    fractionally lower threshold per axis (concurrence test).
    """
    z_overall = zscore_vs_history(metric, lookback=lookback)

    # Normalize input
    if regime_axes is None:
        axes: dict[str, pd.Series] = {}
    elif isinstance(regime_axes, pd.Series):
        axes = {"regime": regime_axes}
    else:
        axes = dict(regime_axes)

    per_axis_z: dict[str, pd.Series] = {}
    for name, regime in axes.items():
        per_axis_z[name] = zscore_by_regime(metric, regime)

    flag = z_overall.abs() >= z_threshold
    if per_axis_z:
        per_axis_pass = pd.concat(per_axis_z, axis=1).abs() >= (z_threshold * regime_threshold_frac)
        # Require ALL axes that have a defined regime-z to agree (NaN axes don't block)
        defined = pd.concat(per_axis_z, axis=1).notna()
        agree = (per_axis_pass | ~defined).all(axis=1)
        flag = flag & agree

    persistent = persistence_filter(flag, min_bars=persistence)

    out = pd.DataFrame({
        "metric": metric,
        "z_overall": z_overall,
        "flag_raw": flag,
        "flag_persistent": persistent,
    })
    for name, z in per_axis_z.items():
        out[f"z_{name}"] = z
    # Back-compat column: keep z_regime when single-axis was supplied
    if isinstance(regime_axes, pd.Series):
        out["z_regime"] = per_axis_z["regime"]
    return out
