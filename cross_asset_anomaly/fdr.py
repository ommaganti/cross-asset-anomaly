from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def two_sided_p_from_z(z: np.ndarray) -> np.ndarray:
    return 2.0 * (1.0 - stats.norm.cdf(np.abs(z)))


def _bh_reject_row(pvals: np.ndarray, alpha: float) -> np.ndarray:
    """Benjamini-Hochberg: return boolean mask of rejected p-values."""
    m = len(pvals)
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(pvals)
    sorted_p = pvals[order]
    thresholds = np.arange(1, m + 1) / m * alpha
    below = sorted_p <= thresholds
    if not below.any():
        return np.zeros(m, dtype=bool)
    k_max = np.where(below)[0].max()
    cutoff = sorted_p[k_max]
    return pvals <= cutoff


def bh_filter_per_date(
    z_matrices: dict[str, pd.DataFrame], alpha: float = 0.05
) -> tuple[dict[str, pd.DataFrame], pd.Series]:
    """Apply Benjamini-Hochberg per date across the joint pair×metric grid.

    Returns:
        - dict mapping metric name -> boolean DataFrame (same shape) marking surviving cells
        - per-date series of how many tests survived
    """
    metric_names = list(z_matrices.keys())
    if not metric_names:
        return {}, pd.Series(dtype=int)

    dates = z_matrices[metric_names[0]].index
    masks = {m: pd.DataFrame(False, index=dates, columns=z_matrices[m].columns) for m in metric_names}
    survivors = pd.Series(0, index=dates, dtype=int)

    # Pre-compute column slices for fast row-stack
    cols_per_metric = {m: z_matrices[m].columns.tolist() for m in metric_names}
    width_per_metric = {m: len(cols_per_metric[m]) for m in metric_names}
    offsets = {}
    cum = 0
    for m in metric_names:
        offsets[m] = cum
        cum += width_per_metric[m]
    total_width = cum

    # Build a stacked (T, total_width) z matrix
    stacked = np.full((len(dates), total_width), np.nan)
    for m in metric_names:
        zs = z_matrices[m].reindex(dates).values
        stacked[:, offsets[m]:offsets[m] + width_per_metric[m]] = zs

    pvals_full = two_sided_p_from_z(stacked)

    for i in range(len(dates)):
        row_p = pvals_full[i]
        valid = ~np.isnan(row_p)
        if not valid.any():
            continue
        idx_valid = np.where(valid)[0]
        rejected = _bh_reject_row(row_p[valid], alpha)
        survivors.iloc[i] = int(rejected.sum())
        if not rejected.any():
            continue
        idx_rejected_global = idx_valid[rejected]
        for m in metric_names:
            lo, hi = offsets[m], offsets[m] + width_per_metric[m]
            local = idx_rejected_global[(idx_rejected_global >= lo) & (idx_rejected_global < hi)] - lo
            if len(local):
                masks[m].iloc[i, local] = True

    return masks, survivors


def apply_mask_to_alerts(alerts: pd.DataFrame, masks: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Filter an alerts DataFrame down to only rows where the (date, pair, metric) survived FDR."""
    if alerts.empty:
        return alerts
    keep = []
    for _, row in alerts.iterrows():
        m = row["metric_name"]
        d = pd.Timestamp(row["date"])
        a, b = row["a"], row["b"]
        mask_df = masks.get(m)
        if mask_df is None or d not in mask_df.index:
            keep.append(False)
            continue
        col = (a, b) if (a, b) in mask_df.columns else (b, a) if (b, a) in mask_df.columns else None
        keep.append(bool(mask_df.at[d, col]) if col is not None else False)
    return alerts.loc[keep].reset_index(drop=True)
