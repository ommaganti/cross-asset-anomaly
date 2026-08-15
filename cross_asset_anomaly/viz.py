from __future__ import annotations

import os
import numpy as np
import pandas as pd


def _import_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def delta_heatmap(current: pd.DataFrame, baseline: pd.DataFrame, out_path: str, title: str) -> str:
    plt = _import_plt()
    delta = (current - baseline).loc[current.index, current.columns]
    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(delta.columns)), max(6, 0.5 * len(delta.index))))
    im = ax.imshow(delta.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(delta.columns)))
    ax.set_xticklabels(delta.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(delta.index)))
    ax.set_yticklabels(delta.index, fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Δ correlation")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def zscore_heatmap(zmatrix: pd.DataFrame, out_path: str, title: str, vmax: float = 4.0) -> str:
    plt = _import_plt()
    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(zmatrix.columns)), max(6, 0.5 * len(zmatrix.index))))
    im = ax.imshow(zmatrix.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(zmatrix.columns)))
    ax.set_xticklabels(zmatrix.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(zmatrix.index)))
    ax.set_yticklabels(zmatrix.index, fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.7, label="z-score vs history")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def stress_timeseries(total_breaks: pd.Series, pct_universe, out_path: str) -> str:
    plt = _import_plt()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(total_breaks.index, total_breaks.values, color="C3", alpha=0.4)
    ax.plot(total_breaks.index, total_breaks.values, color="C3", lw=0.8)
    ax.set_ylabel("Concurrent pair breaks", color="C3")
    ax.set_title("Systemic stress index — count of |z|>threshold pair breaks per day")
    ax.grid(alpha=0.3)
    if pct_universe is not None:
        ax2 = ax.twinx()
        ax2.plot(pct_universe.index, pct_universe.values, color="C0", lw=0.6, alpha=0.5)
        ax2.set_ylabel("% of universe broken", color="C0")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def pca_timeseries(pc1_share: pd.Series, dispersion: pd.Series, out_path: str) -> str:
    plt = _import_plt()
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax[0].plot(pc1_share.index, pc1_share.values, color="C0")
    ax[0].set_ylabel("PC1 share")
    ax[0].set_title("Cross-asset factor structure")
    ax[0].grid(alpha=0.3)
    ax[1].plot(dispersion.index, dispersion.values, color="C1")
    ax[1].set_ylabel("Eigenvalue dispersion")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
