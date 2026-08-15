from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_pearson(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    return x.rolling(window, min_periods=window // 2).corr(y)


def rolling_spearman(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    """Spearman = Pearson on ranks computed within each window."""
    rx = x.rolling(window, min_periods=window // 2).rank()
    ry = y.rolling(window, min_periods=window // 2).rank()
    return rx.rolling(window, min_periods=window // 2).corr(ry)


def rolling_tail_dep(
    x: pd.Series, y: pd.Series, window: int, q: float = 0.1, side: str = "lower"
) -> pd.Series:
    """Pearson correlation conditional on both series being in their joint lower/upper q-tail of the window."""
    out = pd.Series(np.nan, index=x.index, dtype=float)
    xv, yv = x.values, y.values
    n = len(x)
    min_tail = 5
    for i in range(window, n):
        xw = xv[i - window : i]
        yw = yv[i - window : i]
        mask = ~(np.isnan(xw) | np.isnan(yw))
        if mask.sum() < window // 2:
            continue
        xw, yw = xw[mask], yw[mask]
        if side == "lower":
            tx, ty = np.quantile(xw, q), np.quantile(yw, q)
            sel = (xw <= tx) & (yw <= ty)
        else:
            tx, ty = np.quantile(xw, 1 - q), np.quantile(yw, 1 - q)
            sel = (xw >= tx) & (yw >= ty)
        if sel.sum() < min_tail:
            continue
        out.iloc[i] = np.corrcoef(xw[sel], yw[sel])[0, 1]
    return out


def rolling_partial_corr(
    x: pd.Series, y: pd.Series, factors: pd.DataFrame, window: int
) -> pd.Series:
    """Partial correlation of x and y after residualizing both on the factors, over rolling windows."""
    out = pd.Series(np.nan, index=x.index, dtype=float)
    xv = x.values
    yv = y.values
    Fv = factors.values
    n = len(x)
    for i in range(window, n):
        sl = slice(i - window, i)
        xw, yw, Fw = xv[sl], yv[sl], Fv[sl]
        mask = ~(np.isnan(xw) | np.isnan(yw) | np.isnan(Fw).any(axis=1))
        if mask.sum() < max(window // 2, Fw.shape[1] + 5):
            continue
        xw, yw, Fw = xw[mask], yw[mask], Fw[mask]
        Fa = np.column_stack([np.ones(len(Fw)), Fw])
        try:
            bx, *_ = np.linalg.lstsq(Fa, xw, rcond=None)
            by, *_ = np.linalg.lstsq(Fa, yw, rcond=None)
            rx = xw - Fa @ bx
            ry = yw - Fa @ by
            sx, sy = rx.std(), ry.std()
            if sx == 0 or sy == 0:
                continue
            out.iloc[i] = np.corrcoef(rx, ry)[0, 1]
        except np.linalg.LinAlgError:
            continue
    return out


def best_lag_correlation(
    x: pd.Series, y: pd.Series, window: int, max_lag: int = 5
) -> tuple[pd.Series, pd.Series]:
    """At each date, find the lag in [-max_lag, +max_lag] that maximizes |corr(x_t, y_{t+lag})|."""
    best_corr = pd.Series(np.nan, index=x.index, dtype=float)
    best_lag = pd.Series(np.nan, index=x.index, dtype=float)
    for lag in range(-max_lag, max_lag + 1):
        c = x.rolling(window, min_periods=window // 2).corr(y.shift(-lag))
        if lag == -max_lag:
            best_corr = c.copy()
            best_lag = pd.Series(lag, index=x.index, dtype=float)
            best_lag[c.isna()] = np.nan
        else:
            update = c.abs() > best_corr.abs()
            update = update.fillna(False)
            best_corr[update] = c[update]
            best_lag[update] = lag
    return best_corr, best_lag
