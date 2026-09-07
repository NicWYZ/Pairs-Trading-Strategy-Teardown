#!/usr/bin/env python
"""
Run the full study from configs/pairs.yaml and write results + figures.

This is the one-command reproduction entry point: everything the study claims
should be regenerable by `make run` from the config alone.

The pipeline logic lives in ``pairs_teardown.study`` — including the
in-sample-only sizing hedge fit. This script is only argument parsing, 
file writing, and a summary print.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pairs_teardown.config import Config, load_config
from pairs_teardown.data.loaders import load_or_download
from pairs_teardown.plotting.charts import (
    plot_drawdown,
    plot_equity_curve,
    plot_spread_zscore,
)
from pairs_teardown.study import PairRun, run_study, to_frame


def write_outputs(runs: list[PairRun], table: pd.DataFrame, cfg: Config) -> None:
    """Write the metrics CSV, the run manifest, and three figures per pair."""
    results_dir = Path(cfg.output.results_dir)
    figures_dir = Path(cfg.output.figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    table.to_csv(results_dir / "metrics.csv", index=False)

    # A run manifest: what was run, with what parameters, when. Committing this
    # (not the data) is what makes a past result auditable.
    manifest = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "window": cfg.signal.window,
        "entry": cfg.signal.entry,
        "exit": cfg.signal.exit,
        "signal_hedge": cfg.signal.signal_hedge,
        "sizing_hedge": cfg.signal.sizing_hedge,
        "commission_bps": cfg.costs.commission_bps,
        "slippage_bps": cfg.costs.slippage_bps,
        "in_sample_end": cfg.split.in_sample_end,
        "data_start": cfg.data.start,
        "data_end": cfg.data.end,
        "hedge_ratios": {r.pair.name: r.sizing_hedge_ratio for r in runs},
    }
    (results_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    for r in runs:
        slug = r.pair.name.replace("/", "_")
        res = r.result
        gross_equity = (1 + res.gross_returns).cumprod()

        fig = plot_spread_zscore(
            r.spread,
            r.zscore,
            cfg.signal.entry,
            cfg.signal.exit,
            title=f"{r.pair.name} spread and z-score",
        )
        fig.savefig(figures_dir / f"{slug}_spread_zscore.png", dpi=150)

        fig = plot_equity_curve(
            res.equity_curve, gross_equity, title=f"{r.pair.name} equity"
        )
        fig.savefig(figures_dir / f"{slug}_equity.png", dpi=150)

        fig = plot_drawdown(res.equity_curve, title=f"{r.pair.name} drawdown")
        fig.savefig(figures_dir / f"{slug}_drawdown.png", dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/pairs.yaml")
    ap.add_argument(
        "--official-only",
        action="store_true",
        help="run only the three pre-specified pairs",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)

    # Always request the full ticker list so the loader's cache key stays
    # stable; --official-only filters which pairs are RUN, not what is cached.
    prices_raw = load_or_download(
        list(cfg.tickers), cfg.data.start, cfg.data.end, cfg.data.cache_dir
    )

    runs = run_study(prices_raw, cfg, official_only=args.official_only)
    table = to_frame(runs)
    write_outputs(runs, table, cfg)

    view = table[(table.basis == "net") & (table.period != "full")]
    pivot = view.pivot_table(index="pair", columns="period", values="total_return")
    print("\nNET total return, in-sample vs out-of-sample (%):\n")
    print((pivot * 100).round(1).to_string())
    print(f"\nwrote {cfg.output.results_dir}/metrics.csv and figures\n")


if __name__ == "__main__":
    main()
