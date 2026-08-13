from __future__ import annotations

import argparse
import os

from .pipeline import PipelineConfig, run


def main() -> None:
    p = argparse.ArgumentParser(
        prog="cross_asset_anomaly",
        description="Cross-asset rolling-correlation anomaly detector.",
    )
    p.add_argument("--start", default="2014-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--windows", default="20,60,252",
                   help="Comma-separated rolling-correlation windows")
    p.add_argument("--primary-window", type=int, default=60)
    p.add_argument("--zscore-lookback", type=int, default=756,
                   help="Bars of history used to z-score the correlation series")
    p.add_argument("--z-threshold", type=float, default=2.5)
    p.add_argument("--persistence", type=int, default=3,
                   help="Consecutive bars an alert must hold to be flagged")
    p.add_argument("--tail-q", type=float, default=0.10)
    p.add_argument("--lead-lag-max", type=int, default=5)
    p.add_argument("--pca-window", type=int, default=60)
    p.add_argument("--horizons", default="5,20,60",
                   help="Backtest forward-return horizons in bars")
    p.add_argument("--fdr-alpha", type=float, default=0.05,
                   help="Benjamini-Hochberg false-discovery-rate level")
    p.add_argument("--risk-current-window", type=int, default=60,
                   help="Window for current correlation regime in the portfolio risk lens")
    p.add_argument("--risk-baseline-window", type=int, default=1260,
                   help="Trailing baseline window for the portfolio risk lens")
    p.add_argument("--output-dir", default=os.path.abspath("./output"))
    args = p.parse_args()

    cfg = PipelineConfig(
        start=args.start,
        end=args.end,
        windows=tuple(int(x) for x in args.windows.split(",")),
        primary_window=args.primary_window,
        zscore_lookback=args.zscore_lookback,
        z_threshold=args.z_threshold,
        persistence=args.persistence,
        tail_q=args.tail_q,
        lead_lag_max=args.lead_lag_max,
        pca_window=args.pca_window,
        backtest_horizons=tuple(int(x) for x in args.horizons.split(",")),
        fdr_alpha=args.fdr_alpha,
        risk_current_window=args.risk_current_window,
        risk_baseline_window=args.risk_baseline_window,
        output_dir=args.output_dir,
    )
    run(cfg)


if __name__ == "__main__":
    main()
