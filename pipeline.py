from __future__ import annotations

import os
from itertools import combinations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import data as data_mod
from .universe import UNIVERSE, REGIME_TICKER, COMMON_FACTORS, tickers, asset_class_map
from .regimes import vix_regime, build_regime_axes
from .correlations import (
    rolling_pearson, rolling_spearman, rolling_tail_dep,
    rolling_partial_corr, best_lag_correlation,
)
from .pca_factors import rolling_pc1_share, rolling_eigenvalue_dispersion
from .pairs import STRUCTURAL_PAIRS, expected_sign
from .detector import detect_anomalies
from .events import tag_alerts
from .backtest import conditional_forward_stats, anomaly_followups
from .systemic import zscore_matrix, systemic_stress_index, stress_regime, stress_zscore
from .fdr import bh_filter_per_date, apply_mask_to_alerts
from .portfolio import risk_lens, pair_var_attribution, CANONICAL_PORTFOLIOS
from .dashboard import build_dashboard_data, write_dashboard
from .viz import delta_heatmap, zscore_heatmap, pca_timeseries, stress_timeseries


@dataclass
class PipelineConfig:
    start: str = "2014-01-01"
    end: str | None = None
    windows: tuple[int, ...] = (20, 60, 252)
    primary_window: int = 60
    zscore_lookback: int = 756
    z_threshold: float = 2.5
    persistence: int = 3
    tail_q: float = 0.10
    lead_lag_max: int = 5
    pca_window: int = 60
    backtest_horizons: tuple[int, ...] = (5, 20, 60)
    fdr_alpha: float = 0.05
    risk_current_window: int = 60
    risk_baseline_window: int = 1260
    output_dir: str = "./output"


def _pair_universe() -> list[tuple[str, str]]:
    """All structural pairs plus all unique cross-class pairs (capped to avoid blowup)."""
    pairs = {(p.a, p.b) for p in STRUCTURAL_PAIRS}
    ac = asset_class_map()
    for a, b in combinations(tickers(), 2):
        if ac.get(a) != ac.get(b):
            pairs.add((a, b))
    return sorted(pairs)


def _ranked_alerts_for_metric(
    metric_df: pd.DataFrame,   # columns: ("a","b") -> metric series
    regime_axes: dict[str, pd.Series],
    metric_name: str,
    cfg: PipelineConfig,
) -> pd.DataFrame:
    rows = []
    axis_cols = [f"z_{name}" for name in regime_axes]
    for (a, b), series in metric_df.items():
        if series.dropna().empty:
            continue
        d = detect_anomalies(
            series, regime_axes,
            z_threshold=cfg.z_threshold,
            persistence=cfg.persistence,
            lookback=cfg.zscore_lookback,
        )
        active = d[d["flag_persistent"]]
        if active.empty:
            continue
        for date, row in active.iterrows():
            rec = {
                "date": date,
                "a": a,
                "b": b,
                "metric_name": metric_name,
                "value": row["metric"],
                "z_overall": row["z_overall"],
                "expected_sign": expected_sign(a, b) or "",
            }
            for col in axis_cols:
                rec[col] = row.get(col, float("nan"))
            # Back-compat: surface the volatility-axis z as z_regime when present
            rec["z_regime"] = row.get("z_vol", float("nan"))
            rows.append(rec)
    if not rows:
        base_cols = ["date", "a", "b", "metric_name", "value", "z_overall", "expected_sign", "z_regime"]
        return pd.DataFrame(columns=base_cols + axis_cols)
    out = pd.DataFrame(rows).sort_values(["date", "z_overall"], key=lambda s: s.abs() if s.name == "z_overall" else s, ascending=[False, False])
    return out.reset_index(drop=True)


def run(cfg: PipelineConfig) -> dict:
    os.makedirs(cfg.output_dir, exist_ok=True)
    print(f"[1/9] Loading prices  start={cfg.start} end={cfg.end or 'latest'}")
    prices = data_mod.load_prices(start=cfg.start, end=cfg.end)
    prices = data_mod.align_calendar(prices, reference="SPY")
    prices = data_mod.filter_stale(prices)

    print("[2/9] Computing returns + vol-adjusted returns")
    rets = data_mod.log_returns(prices).dropna(how="all")
    rets_adj = data_mod.vol_adjusted_returns(rets, window=20)

    print("[3/9] Classifying multi-axis regimes (vol / curve / USD / credit)")
    if REGIME_TICKER not in prices.columns:
        raise RuntimeError(f"Regime ticker {REGIME_TICKER} not in price data")
    regime_axes = build_regime_axes(prices)
    regime = regime_axes.get("vol", vix_regime(prices[REGIME_TICKER]))
    print(f"       regime axes active: {', '.join(regime_axes.keys())}")

    uni = tickers()
    rets_u = rets_adj[uni].copy()
    factors = rets_adj[COMMON_FACTORS].copy()

    pairs = _pair_universe()
    print(f"[4/9] Computing rolling correlations  pairs={len(pairs)} windows={cfg.windows}")

    pearson: dict[int, dict[tuple[str, str], pd.Series]] = {w: {} for w in cfg.windows}
    spearman: dict[tuple[str, str], pd.Series] = {}
    tail: dict[tuple[str, str], pd.Series] = {}
    partial: dict[tuple[str, str], pd.Series] = {}
    leadlag_corr: dict[tuple[str, str], pd.Series] = {}
    leadlag_lag: dict[tuple[str, str], pd.Series] = {}

    pw = cfg.primary_window
    structural_set = {(p.a, p.b) for p in STRUCTURAL_PAIRS}

    for a, b in pairs:
        if a not in rets_u.columns or b not in rets_u.columns:
            continue
        x, y = rets_u[a], rets_u[b]
        for w in cfg.windows:
            pearson[w][(a, b)] = rolling_pearson(x, y, w)
        spearman[(a, b)] = rolling_spearman(x, y, pw)
        # Heavier metrics: only on structural pairs to keep runtime reasonable
        if (a, b) in structural_set:
            tail[(a, b)] = rolling_tail_dep(x, y, pw, q=cfg.tail_q)
            remaining = [c for c in COMMON_FACTORS if c not in (a, b)]
            if remaining:
                partial[(a, b)] = rolling_partial_corr(x, y, factors[remaining], pw)
            bc, bl = best_lag_correlation(x, y, pw, max_lag=cfg.lead_lag_max)
            leadlag_corr[(a, b)] = bc
            leadlag_lag[(a, b)] = bl

    print("[5/9] Computing factor structure (PCA)")
    pc1 = rolling_pc1_share(rets_u, cfg.pca_window)
    disp = rolling_eigenvalue_dispersion(rets_u, cfg.pca_window)

    pearson_pw_df = pd.DataFrame(pearson[pw])
    pearson_short_df = pd.DataFrame(pearson[cfg.windows[0]])
    pearson_long_df = pd.DataFrame(pearson[cfg.windows[-1]])
    spearman_df = pd.DataFrame(spearman)
    tail_df = pd.DataFrame(tail) if tail else pd.DataFrame()
    partial_df = pd.DataFrame(partial) if partial else pd.DataFrame()
    leadlag_corr_df = pd.DataFrame(leadlag_corr) if leadlag_corr else pd.DataFrame()
    leadlag_lag_df = pd.DataFrame(leadlag_lag) if leadlag_lag else pd.DataFrame()

    short_vs_long = pearson_short_df - pearson_long_df

    print("[6/9] Detecting anomalies (z-score + regime + persistence)")
    metrics = {
        f"pearson_{pw}d": pearson_pw_df,
        "short_minus_long": short_vs_long,
        f"spearman_{pw}d": spearman_df,
    }
    if not tail_df.empty:
        metrics[f"tail_dep_{pw}d"] = tail_df
    if not partial_df.empty:
        metrics[f"partial_corr_{pw}d"] = partial_df

    alerts_by_metric = {}
    for name, df in metrics.items():
        alerts_by_metric[name] = _ranked_alerts_for_metric(df, regime_axes, name, cfg)

    all_alerts = pd.concat(alerts_by_metric.values(), ignore_index=True) if alerts_by_metric else pd.DataFrame()
    if not all_alerts.empty:
        all_alerts = tag_alerts(all_alerts, date_col="date", window_days=3)
        all_alerts["abs_z"] = all_alerts["z_overall"].abs()
        all_alerts = all_alerts.sort_values(["date", "abs_z"], ascending=[False, False]).reset_index(drop=True)

    print("[7/9] Systemic stress index + FDR (Benjamini-Hochberg) correction")
    # Build z-score matrices for every metric, aligned to a common index
    z_matrices = {name: zscore_matrix(df, lookback=cfg.zscore_lookback) for name, df in metrics.items()}
    z_matrices = {k: v for k, v in z_matrices.items() if not v.empty}

    stress = systemic_stress_index(z_matrices, threshold=cfg.z_threshold)
    if not stress.empty:
        stress["stress_regime"] = stress_regime(stress["total_breaks"])
        stress["stress_z"] = stress_zscore(stress["total_breaks"])

    masks, fdr_survivors = bh_filter_per_date(z_matrices, alpha=cfg.fdr_alpha)
    fdr_alerts = apply_mask_to_alerts(all_alerts, masks) if not all_alerts.empty else pd.DataFrame()

    print("[8/9] Structural sign-violation + portfolio risk lens + backtest")
    sign_violations = _structural_sign_violations(pearson_pw_df, cfg.primary_window)

    # Portfolio risk lens: how does today's correlation regime reprice canonical-portfolio risk
    risk = risk_lens(rets[uni], current_window=cfg.risk_current_window,
                     baseline_window=cfg.risk_baseline_window)
    attrib_6040 = pair_var_attribution(
        rets[uni], CANONICAL_PORTFOLIOS["60_40"],
        current_window=cfg.risk_current_window, baseline_window=cfg.risk_baseline_window,
    )

    bt = pd.DataFrame()
    followups = pd.DataFrame()
    if not all_alerts.empty:
        anomaly_dates = pd.DatetimeIndex(all_alerts["date"].unique())
        bt = conditional_forward_stats(anomaly_dates, rets[uni], cfg.backtest_horizons)
        followups = anomaly_followups(all_alerts.head(200), rets[uni], cfg.backtest_horizons)

    print("[9/9] Writing outputs")
    od = cfg.output_dir
    if not all_alerts.empty:
        all_alerts.to_csv(os.path.join(od, "alerts.csv"), index=False)
        # Biggest-ever anomalies: dedup to one row per (a,b,metric) by keeping the max-|z| occurrence
        ranked = all_alerts.copy()
        ranked["pair_key"] = ranked.apply(lambda r: tuple(sorted([r["a"], r["b"]])) + (r["metric_name"],), axis=1)
        ranked = ranked.sort_values("abs_z", ascending=False).drop_duplicates("pair_key", keep="first")
        ranked = ranked.drop(columns=["pair_key"]).head(200).reset_index(drop=True)
        ranked.to_csv(os.path.join(od, "alerts_top.csv"), index=False)
    sign_violations.to_csv(os.path.join(od, "structural_sign_violations.csv"), index=False)
    if not bt.empty:
        bt.to_csv(os.path.join(od, "backtest_conditional_forward_returns.csv"), index=False)
    if not followups.empty:
        followups.to_csv(os.path.join(od, "anomaly_followups.csv"), index=False)
    if not stress.empty:
        stress.to_csv(os.path.join(od, "systemic_stress.csv"))
    if not fdr_alerts.empty:
        fdr_alerts.to_csv(os.path.join(od, "fdr_alerts.csv"), index=False)
    if not risk.empty:
        risk.to_csv(os.path.join(od, "portfolio_risk.csv"), index=False)
    if not attrib_6040.empty:
        attrib_6040.to_csv(os.path.join(od, "portfolio_pair_attribution.csv"), index=False)

    latest = pearson_pw_df.dropna(how="all").iloc[-1]
    baseline = pearson_pw_df.mean()
    cur_mat = _series_pairs_to_matrix(latest, uni)
    base_mat = _series_pairs_to_matrix(baseline, uni)
    delta_heatmap(cur_mat, base_mat, os.path.join(od, "heatmap_delta_vs_norm.png"),
                  title=f"Pair corr ({pw}d) — current minus all-time mean")

    latest_z = _latest_zscore_matrix(pearson_pw_df, regime, uni, cfg)
    zscore_heatmap(latest_z, os.path.join(od, "heatmap_zscore.png"),
                   title=f"Pair corr ({pw}d) — z-score vs own history")

    pca_timeseries(pc1, disp, os.path.join(od, "pca_factor_structure.png"))
    if not stress.empty:
        stress_timeseries(stress["total_breaks"], stress.get("pct_universe_broken"),
                          os.path.join(od, "systemic_stress.png"))

    # Interactive dashboard (self-contained HTML, data embedded inline)
    if not stress.empty and not pearson_pw_df.empty:
        from .events import event_calendar
        z_pearson = z_matrices.get(f"pearson_{pw}d", pd.DataFrame())
        dash = build_dashboard_data(
            pearson_pw_df=pearson_pw_df,
            z_pearson_df=z_pearson,
            stress=stress,
            pc1=pc1,
            rets_uni=rets[uni],
            asset_list=uni,
            class_map=asset_class_map(),
            events=event_calendar(),
            leadlag_lag_df=leadlag_lag_df,
            primary_window=pw,
            risk_current_window=cfg.risk_current_window,
            risk_baseline_window=cfg.risk_baseline_window,
        )
        write_dashboard(dash, os.path.join(od, "dashboard.html"))
        print(f"       dashboard: {os.path.join(od, 'dashboard.html')} "
              f"({dash['meta']['n_snapshots']} snapshots)")

    print()
    _summary(all_alerts, fdr_alerts, sign_violations, bt, pc1, disp, stress, risk, attrib_6040, fdr_survivors)

    return {
        "alerts": all_alerts,
        "fdr_alerts": fdr_alerts,
        "structural_sign_violations": sign_violations,
        "backtest": bt,
        "followups": followups,
        "pc1_share": pc1,
        "eigenvalue_dispersion": disp,
        "leadlag_corr": leadlag_corr_df,
        "leadlag_lag": leadlag_lag_df,
        "systemic_stress": stress,
        "fdr_survivors": fdr_survivors,
        "portfolio_risk": risk,
        "portfolio_pair_attribution": attrib_6040,
        "output_dir": cfg.output_dir,
    }


def _structural_sign_violations(corr_df: pd.DataFrame, window: int) -> pd.DataFrame:
    rows = []
    last = corr_df.dropna(how="all").iloc[-1] if not corr_df.empty else pd.Series(dtype=float)
    for sp in STRUCTURAL_PAIRS:
        key = (sp.a, sp.b)
        if key not in corr_df.columns:
            continue
        series = corr_df[key].dropna()
        if series.empty:
            continue
        cur = float(series.iloc[-1])
        avg = float(series.mean())
        violated = (sp.expected_sign == "positive" and cur < 0) or (sp.expected_sign == "negative" and cur > 0)
        rows.append({
            "a": sp.a, "b": sp.b,
            "expected_sign": sp.expected_sign,
            f"corr_{window}d_now": cur,
            f"corr_{window}d_avg": avg,
            "violated": violated,
            "rationale": sp.rationale,
        })
    return pd.DataFrame(rows).sort_values("violated", ascending=False).reset_index(drop=True)


def _series_pairs_to_matrix(series: pd.Series, universe: list[str]) -> pd.DataFrame:
    mat = pd.DataFrame(np.nan, index=universe, columns=universe)
    for (a, b), v in series.items():
        if a in mat.index and b in mat.columns:
            mat.loc[a, b] = v
            mat.loc[b, a] = v
    np.fill_diagonal(mat.values, 1.0)
    return mat


def _latest_zscore_matrix(corr_df: pd.DataFrame, regime: pd.Series, universe: list[str], cfg: PipelineConfig) -> pd.DataFrame:
    mat = pd.DataFrame(np.nan, index=universe, columns=universe)
    for (a, b) in corr_df.columns:
        s = corr_df[(a, b)]
        if s.dropna().empty:
            continue
        d = detect_anomalies(s, regime, z_threshold=cfg.z_threshold,
                             persistence=1, lookback=cfg.zscore_lookback)
        latest_z = d["z_overall"].dropna()
        if latest_z.empty:
            continue
        v = float(latest_z.iloc[-1])
        if a in mat.index and b in mat.columns:
            mat.loc[a, b] = v
            mat.loc[b, a] = v
    np.fill_diagonal(mat.values, 0.0)
    return mat


def _summary(alerts: pd.DataFrame, fdr_alerts: pd.DataFrame, sign_violations: pd.DataFrame,
             bt: pd.DataFrame, pc1: pd.Series, disp: pd.Series,
             stress: pd.DataFrame, risk: pd.DataFrame, attrib: pd.DataFrame,
             fdr_survivors: pd.Series) -> None:
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)

    # --- Systemic stress headline (the top-level dial) ---
    if not stress.empty:
        last = stress.iloc[-1]
        reg = last.get("stress_regime", "n/a")
        print(f"SYSTEMIC STRESS: {int(last['total_breaks'])} concurrent pair breaks "
              f"({last['pct_universe_broken']:.1f}% of universe)  "
              f"z={last.get('stress_z', float('nan')):+.2f}  regime={reg}")
        fdr_today = int(fdr_survivors.iloc[-1]) if not fdr_survivors.empty else 0
        raw_today = int((alerts['date'] == alerts['date'].max()).sum()) if not alerts.empty else 0
        print(f"FDR (BH α): {fdr_today} tests survive correction today "
              f"(vs {raw_today} raw persistent alerts) — the rest are multiple-comparison noise")
    print()

    if alerts.empty:
        print("No persistent correlation anomalies detected with current thresholds.")
    else:
        cols = ["date", "a", "b", "metric_name", "value", "z_overall", "z_regime", "nearby_event"]
        cols = [c for c in cols if c in alerts.columns]

        # Prefer FDR-surviving alerts for the headline list when available
        show = fdr_alerts if not fdr_alerts.empty else alerts
        label = "FDR-surviving" if not fdr_alerts.empty else "raw"
        recent = show.sort_values(["date", "abs_z"], ascending=[False, False]).head(15) if "abs_z" in show.columns else show.head(15)
        print(f"Top {len(recent)} most recent {label} anomalies:")
        print(recent[cols].to_string(index=False))
        print()

        ranked = alerts.copy()
        ranked["pair_key"] = ranked.apply(lambda r: tuple(sorted([r["a"], r["b"]])) + (r["metric_name"],), axis=1)
        ranked = ranked.sort_values("abs_z", ascending=False).drop_duplicates("pair_key", keep="first").head(15)
        print(f"Top {len(ranked)} biggest-ever anomalies (one row per pair+metric):")
        print(ranked[cols].to_string(index=False))
    print()

    if not sign_violations.empty:
        v = sign_violations[sign_violations["violated"]]
        if not v.empty:
            print(f"Structural-pair SIGN VIOLATIONS ({len(v)}):")
            print(v[["a", "b", "expected_sign", "rationale"]].to_string(index=False))
        else:
            print("No structural-pair sign violations currently.")
    print()

    if not pc1.dropna().empty:
        z_pc1 = (pc1.iloc[-1] - pc1.mean()) / pc1.std()
        z_disp = (disp.iloc[-1] - disp.mean()) / disp.std()
        print(f"Factor structure  PC1 share: {pc1.iloc[-1]:.2%} (z={z_pc1:+.2f})   "
              f"eigenvalue dispersion: {disp.iloc[-1]:.3f} (z={z_disp:+.2f})")
    print()

    # --- Portfolio risk lens (what a PM actually cares about) ---
    if not risk.empty:
        print("PORTFOLIO RISK LENS — annualized vol, current corr regime vs trailing baseline:")
        show_cols = ["portfolio", "vol_baseline_ann", "vol_current_ann",
                     "delta_total_pct", "delta_corr_only_pct", "delta_vol_only_pct"]
        r = risk[show_cols].copy()
        for c in ["vol_baseline_ann", "vol_current_ann"]:
            r[c] = (r[c] * 100).round(2)
        for c in ["delta_total_pct", "delta_corr_only_pct", "delta_vol_only_pct"]:
            r[c] = r[c].round(1)
        print(r.to_string(index=False))
        if not attrib.empty:
            print()
            print("60/40 risk change — top pair contributions from Δcorrelation:")
            a = attrib[["a", "b", "corr_baseline", "corr_current", "delta_corr", "variance_contribution"]].copy()
            print(a.round(4).to_string(index=False))
    print()

    if not bt.empty:
        print("Conditional forward-return excess (anomaly mean − baseline mean):")
        pivot = bt.pivot_table(index="asset", columns="horizon_bars", values="excess_mean")
        print(pivot.round(4).to_string())
