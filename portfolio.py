from __future__ import annotations

import numpy as np
import pandas as pd


# Canonical portfolios — weights must sum to 1.0 and reference tickers in the universe.
CANONICAL_PORTFOLIOS: dict[str, dict[str, float]] = {
    "60_40":              {"SPY": 0.60, "IEF": 0.30, "TLT": 0.10},
    "global_60_40":       {"SPY": 0.40, "EFA": 0.15, "EEM": 0.05,
                           "IEF": 0.25, "TLT": 0.10, "LQD": 0.05},
    "risk_parity_lite":   {"SPY": 0.30, "TLT": 0.30, "GLD": 0.20,
                           "USO": 0.10, "EEM": 0.10},
    "all_weather_lite":   {"SPY": 0.30, "TLT": 0.40, "IEF": 0.15,
                           "GLD": 0.075, "USO": 0.075},
}


def _slice_weights(weights: dict[str, float], available: list[str]) -> tuple[list[str], np.ndarray]:
    assets = [a for a in weights if a in available]
    w = np.array([weights[a] for a in assets], dtype=float)
    if w.sum() == 0:
        return assets, w
    return assets, w / w.sum()


def portfolio_vol(weights: dict[str, float], cov: pd.DataFrame) -> float:
    assets, w = _slice_weights(weights, cov.columns.tolist())
    if not assets:
        return float("nan")
    sub = cov.loc[assets, assets].values
    return float(np.sqrt(w @ sub @ w))


def covariance_from_corr_vol(corr: pd.DataFrame, vols: pd.Series) -> pd.DataFrame:
    common = [a for a in corr.index if a in vols.index]
    c = corr.loc[common, common].values
    v = vols.loc[common].values
    return pd.DataFrame(np.outer(v, v) * c, index=common, columns=common)


def risk_lens(
    returns: pd.DataFrame,
    portfolios: dict[str, dict[str, float]] | None = None,
    current_window: int = 60,
    baseline_window: int = 1260,
    ann_factor: int = 252,
) -> pd.DataFrame:
    """For each portfolio, decompose the change in risk into corr-only vs vol-only vs total."""
    if portfolios is None:
        portfolios = CANONICAL_PORTFOLIOS
    rets = returns.dropna(how="all")
    if len(rets) < current_window + 60:
        return pd.DataFrame()

    cur = rets.iloc[-current_window:]
    base = rets.iloc[-(baseline_window + current_window):-current_window] if len(rets) >= baseline_window + current_window else rets.iloc[:-current_window]

    cur_vols = cur.std() * np.sqrt(ann_factor)
    base_vols = base.std() * np.sqrt(ann_factor)
    cur_corr = cur.corr()
    base_corr = base.corr()

    cur_cov = covariance_from_corr_vol(cur_corr, cur_vols)
    base_cov = covariance_from_corr_vol(base_corr, base_vols)
    corr_only_cov = covariance_from_corr_vol(cur_corr, base_vols)
    vol_only_cov = covariance_from_corr_vol(base_corr, cur_vols)

    z = 1.645  # 95% one-sided normal
    rows = []
    for name, weights in portfolios.items():
        v_cur = portfolio_vol(weights, cur_cov)
        v_base = portfolio_vol(weights, base_cov)
        v_corr_only = portfolio_vol(weights, corr_only_cov)
        v_vol_only = portfolio_vol(weights, vol_only_cov)
        if not np.isfinite(v_base) or v_base == 0:
            continue
        rows.append({
            "portfolio": name,
            "vol_baseline_ann":     v_base,
            "vol_current_ann":      v_cur,
            "vol_corr_only_ann":    v_corr_only,
            "vol_vol_only_ann":     v_vol_only,
            "delta_total_pct":      (v_cur - v_base) / v_base * 100,
            "delta_corr_only_pct":  (v_corr_only - v_base) / v_base * 100,
            "delta_vol_only_pct":   (v_vol_only - v_base) / v_base * 100,
            "var95_baseline_ann":   z * v_base,
            "var95_current_ann":    z * v_cur,
            "var95_delta_ann_pct":  (v_cur - v_base) / v_base * 100,
        })
    return pd.DataFrame(rows)


def risk_delta_at_date(
    returns: pd.DataFrame,
    date,
    weights: dict[str, float],
    current_window: int = 60,
    baseline_window: int = 1260,
    ann_factor: int = 252,
) -> dict | None:
    """Portfolio risk decomposition as of a specific date (for time-series / dashboard use)."""
    rets = returns.dropna(how="all")
    if date not in rets.index:
        return None
    loc = rets.index.get_loc(date)
    if loc < current_window + 60:
        return None
    cur = rets.iloc[loc - current_window:loc]
    base_start = max(0, loc - current_window - baseline_window)
    base = rets.iloc[base_start:loc - current_window]
    if len(base) < 60:
        return None

    cur_vols = cur.std() * np.sqrt(ann_factor)
    base_vols = base.std() * np.sqrt(ann_factor)
    cur_corr = cur.corr()
    base_corr = base.corr()

    cur_cov = covariance_from_corr_vol(cur_corr, cur_vols)
    base_cov = covariance_from_corr_vol(base_corr, base_vols)
    corr_only_cov = covariance_from_corr_vol(cur_corr, base_vols)

    v_cur = portfolio_vol(weights, cur_cov)
    v_base = portfolio_vol(weights, base_cov)
    v_corr_only = portfolio_vol(weights, corr_only_cov)
    if not np.isfinite(v_base) or v_base == 0:
        return None
    return {
        "vol_baseline_ann": float(v_base),
        "vol_current_ann": float(v_cur),
        "delta_total_pct": float((v_cur - v_base) / v_base * 100),
        "delta_corr_only_pct": float((v_corr_only - v_base) / v_base * 100),
    }


def pair_var_attribution(
    returns: pd.DataFrame,
    weights: dict[str, float],
    current_window: int = 60,
    baseline_window: int = 1260,
    ann_factor: int = 252,
    top_n: int = 10,
) -> pd.DataFrame:
    """For each pair (i,j) in the portfolio, the contribution of Δρ to portfolio variance change.

    Decomposition: ΔVar ≈ Σ_{i<j} 2 w_i w_j σ_i σ_j Δρ_{ij}
    (holding vols at current-window levels for a corr-only attribution).
    """
    rets = returns[[a for a in weights if a in returns.columns]].dropna(how="all")
    if len(rets) < current_window + 60:
        return pd.DataFrame()
    cur = rets.iloc[-current_window:]
    base = rets.iloc[-(baseline_window + current_window):-current_window]

    cur_corr = cur.corr()
    base_corr = base.corr()
    vols = cur.std() * np.sqrt(ann_factor)

    rows = []
    assets = list(weights.keys())
    for i, ai in enumerate(assets):
        if ai not in cur.columns:
            continue
        for j in range(i + 1, len(assets)):
            aj = assets[j]
            if aj not in cur.columns:
                continue
            d_rho = float(cur_corr.loc[ai, aj] - base_corr.loc[ai, aj])
            contrib = 2 * weights[ai] * weights[aj] * vols[ai] * vols[aj] * d_rho
            rows.append({
                "a": ai, "b": aj,
                "corr_current": float(cur_corr.loc[ai, aj]),
                "corr_baseline": float(base_corr.loc[ai, aj]),
                "delta_corr": d_rho,
                "variance_contribution": contrib,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.reindex(df["variance_contribution"].abs().sort_values(ascending=False).index).head(top_n).reset_index(drop=True)
