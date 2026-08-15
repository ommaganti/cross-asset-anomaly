from __future__ import annotations

import numpy as np
import pandas as pd


def _eigvals(window_matrix: np.ndarray) -> np.ndarray | None:
    sub = window_matrix
    sub = sub[~np.isnan(sub).any(axis=1)]
    if len(sub) < sub.shape[1] + 5:
        return None
    std = sub.std(0)
    if (std == 0).any():
        return None
    sub = (sub - sub.mean(0)) / std
    cov = np.cov(sub, rowvar=False)
    try:
        ev = np.linalg.eigvalsh(cov)
    except np.linalg.LinAlgError:
        return None
    return np.sort(ev)[::-1]


def rolling_pc1_share(returns: pd.DataFrame, window: int) -> pd.Series:
    """Fraction of variance explained by the top eigenvalue — proxy for systemic co-movement."""
    out = pd.Series(np.nan, index=returns.index, dtype=float)
    vals = returns.values
    for i in range(window, len(returns)):
        ev = _eigvals(vals[i - window : i])
        if ev is None or ev.sum() <= 0:
            continue
        out.iloc[i] = ev[0] / ev.sum()
    return out


def rolling_eigenvalue_dispersion(returns: pd.DataFrame, window: int) -> pd.Series:
    """Std(eigenvalues)/mean(eigenvalues): high values = concentrated risk in few factors."""
    out = pd.Series(np.nan, index=returns.index, dtype=float)
    vals = returns.values
    for i in range(window, len(returns)):
        ev = _eigvals(vals[i - window : i])
        if ev is None or ev.mean() == 0:
            continue
        out.iloc[i] = ev.std() / ev.mean()
    return out


def rolling_effective_rank(returns: pd.DataFrame, window: int) -> pd.Series:
    """Participation ratio: exp(entropy of normalized eigenvalues). Inverse-related to PC1 share."""
    out = pd.Series(np.nan, index=returns.index, dtype=float)
    vals = returns.values
    for i in range(window, len(returns)):
        ev = _eigvals(vals[i - window : i])
        if ev is None:
            continue
        p = ev / ev.sum()
        p = p[p > 0]
        out.iloc[i] = float(np.exp(-(p * np.log(p)).sum()))
    return out
