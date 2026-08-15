from __future__ import annotations

import numpy as np
import pandas as pd


def forward_returns(returns: pd.DataFrame, horizons: tuple[int, ...] = (5, 20, 60)) -> dict[int, pd.DataFrame]:
    """For each horizon h, the cumulative return from t+1 to t+h, aligned to date t."""
    out: dict[int, pd.DataFrame] = {}
    for h in horizons:
        out[h] = returns.shift(-1).rolling(h, min_periods=h).sum().shift(-(h - 1))
    return out


def conditional_forward_stats(
    anomaly_dates: pd.DatetimeIndex,
    returns: pd.DataFrame,
    horizons: tuple[int, ...] = (5, 20, 60),
) -> pd.DataFrame:
    """For each (horizon, asset), report mean / std / hit-rate of forward returns following anomalies vs baseline."""
    fwd = forward_returns(returns, horizons)
    rows = []
    for h, fr in fwd.items():
        baseline = fr
        cond = fr.loc[fr.index.intersection(anomaly_dates)]
        if cond.empty:
            continue
        for asset in fr.columns:
            c = cond[asset].dropna()
            b = baseline[asset].dropna()
            if c.empty or b.empty:
                continue
            rows.append({
                "horizon_bars": h,
                "asset": asset,
                "n_anomaly": int(c.shape[0]),
                "mean_anomaly": float(c.mean()),
                "std_anomaly": float(c.std()),
                "hit_rate_pos": float((c > 0).mean()),
                "mean_baseline": float(b.mean()),
                "std_baseline": float(b.std()),
                "excess_mean": float(c.mean() - b.mean()),
                "t_stat": float((c.mean() - b.mean()) / (c.std() / np.sqrt(max(c.shape[0], 1)))) if c.std() > 0 else np.nan,
            })
    return pd.DataFrame(rows)


def anomaly_followups(
    anomalies: pd.DataFrame,
    returns: pd.DataFrame,
    horizons: tuple[int, ...] = (5, 20, 60),
    pair_assets: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Per-anomaly forward returns of the two pair members (when applicable)."""
    if anomalies.empty:
        return pd.DataFrame()
    fwd = forward_returns(returns, horizons)
    rows = []
    for _, row in anomalies.iterrows():
        date = pd.Timestamp(row["date"])
        assets = (row.get("a"), row.get("b")) if pair_assets is None else pair_assets
        rec = {"date": date, "a": row.get("a"), "b": row.get("b"), "metric": row.get("metric_name")}
        for h, fr in fwd.items():
            if date not in fr.index:
                continue
            for asset in assets:
                if asset in fr.columns:
                    rec[f"{asset}_fwd_{h}d"] = fr.loc[date, asset]
        rows.append(rec)
    return pd.DataFrame(rows)
