#!/usr/bin/env python
"""
Run the full study from configs/pairs.yaml and write results + figures.

This is the one-command reproduction entry point: everything the study claims
should be regenerable by `make run` from the config alone.

THE CENTRAL DISCIPLINE OF THIS SCRIPT
-------------------------------------
The static SIZING hedge ratio is estimated on IN-SAMPLE DATA ONLY, then frozen
and applied unchanged to the out-of-sample period. Fitting it on the full sample
-- as the exploratory notebooks did -- leaks out-of-sample information into the
position sizing of every trade, which would invalidate the OOS result that this
stage exists to produce. `estimate_hedge_ratio`'s own docstring warns about this;
here it is enforced.

The out-of-sample period is scored exactly once, with all parameters frozen.
Nothing in this script selects a parameter by looking at a return.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from pairs_teardown.backtest.costs import CostModel
from pairs_teardown.backtest.engine import run_backtest
from pairs_teardown.config import Config, Pair, load_config
from pairs_teardown.data.clean import align_prices, handle_missing
from pairs_teardown.data.loaders import load_or_download
from pairs_teardown.metrics.performance import summary
from pairs_teardown.plotting.charts import (
    plot_drawdown,
    plot_equity_curve,
    plot_spread_zscore,
)
from pairs_teardown.signals.rules import target_positions
from pairs_teardown.signals.spread import build_rolling_spread, rolling_zscore
from pairs_teardown.stats.cointegration import estimate_hedge_ratio


def run_pair(pair: Pair, prices_raw: pd.DataFrame, cfg: Config) -> dict:
    """Run one pair end-to-end and return its result bundle."""
    px = align_prices(handle_missing(prices_raw[[pair.a, pair.b]]))
    if px.empty:
        raise ValueError(f"{pair.name}: no overlapping price data in range")

    # Log prices: multiplicative growth makes level spreads heteroskedastic and
    # breaks the stationarity the whole method assumes.
    log_a = pd.Series(np.log(px[pair.a]))
    log_b = pd.Series(np.log(px[pair.b]))

    is_mask = cfg.split.is_mask(px.index)
    if is_mask.sum() < cfg.signal.window:
        raise ValueError(f"{pair.name}: in-sample window shorter than signal window")

    # --- SIZING: static, IN-SAMPLE ONLY, then frozen for the whole run ------
    g_static = estimate_hedge_ratio(log_a[is_mask], log_b[is_mask])
    sizing_hedge = pd.Series(g_static, index=px.index)

    # --- SIGNAL: rolling hedge -> spread -> z-score -> positions ------------
    # Rolling is ADF-justified (static-hedge spreads fail stationarity for
    # several pairs). It must never be used for sizing.
    spread = build_rolling_spread(log_a, log_b, cfg.signal.window)
    z = rolling_zscore(spread, cfg.signal.window)
    positions = target_positions(
        z,
        entry_threshold=cfg.signal.entry,
        exit_threshold=cfg.signal.exit,
    )

    # --- backtest (engine applies the one-bar lag and validates the hedge) --
    cost_model = CostModel(
        commission_bps=cfg.costs.commission_bps,
        slippage_bps=cfg.costs.slippage_bps,
    )
    result = run_backtest(
        price_a=px[pair.a],
        price_b=px[pair.b],
        target_positions=positions,
        hedge_ratio=sizing_hedge,
        cost_model=cost_model,
    )

    ppy = cfg.backtest.periods_per_year
    is_m = cfg.split.is_mask(result.returns.index)
    oos_m = cfg.split.oos_mask(result.returns.index)

    return {
        "pair": pair,
        "result": result,
        "spread": spread,
        "zscore": z,
        "g_static": g_static,
        "full": summary(result, periods_per_year=ppy),
        "in_sample": summary(_slice(result, is_m), periods_per_year=ppy),
        "out_of_sample": summary(_slice(result, oos_m), periods_per_year=ppy),
    }


class _Sliced:
    """
    Minimal stand-in exposing the three attributes ``summary`` duck-types on.

    Slicing one simulation rather than re-running is deliberate: a fresh run for
    the OOS window could accidentally re-fit something.
    """

    def __init__(self, returns, gross_returns, held_positions):
        self.returns = returns
        self.gross_returns = gross_returns
        self.held_positions = held_positions


def _slice(result, mask: pd.Series) -> _Sliced:
    return _Sliced(
        result.returns[mask],
        result.gross_returns[mask],
        result.held_positions[mask],
    )


def to_frame(runs: list[dict]) -> pd.DataFrame:
    """Flatten all runs into one long-form table: pair x period x gross/net."""
    rows = []
    for r in runs:
        for period in ("full", "in_sample", "out_of_sample"):
            for basis in ("gross", "net"):
                row = {
                    "pair": r["pair"].name,
                    "group": r["pair"].group,
                    "period": period,
                    "basis": basis,
                    "hedge_ratio": r["g_static"],
                    "n_trades": r["result"].n_trades,
                }
                row.update(r[period][basis])
                rows.append(row)
    return pd.DataFrame(rows)


def write_outputs(runs: list[dict], table: pd.DataFrame, cfg: Config) -> None:
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
        "hedge_ratios": {r["pair"].name: r["g_static"] for r in runs},
    }
    (results_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    for r in runs:
        slug = r["pair"].name.replace("/", "_")
        res = r["result"]
        gross_equity = (1 + res.gross_returns).cumprod()

        fig = plot_spread_zscore(
            r["spread"],
            r["zscore"],
            cfg.signal.entry,
            cfg.signal.exit,
            title=f"{r['pair'].name} spread and z-score",
        )
        fig.savefig(figures_dir / f"{slug}_spread_zscore.png", dpi=150)

        fig = plot_equity_curve(
            res.equity_curve, gross_equity, title=f"{r['pair'].name} equity"
        )
        fig.savefig(figures_dir / f"{slug}_equity.png", dpi=150)

        fig = plot_drawdown(res.equity_curve, title=f"{r['pair'].name} drawdown")
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
    pairs = cfg.official_pairs if args.official_only else cfg.pairs

    # Always request the full ticker list so the loader's cache key stays
    # stable; --official-only filters which pairs are RUN, not what is cached.
    prices_raw = load_or_download(
        list(cfg.tickers), cfg.data.start, cfg.data.end, cfg.data.cache_dir
    )

    runs = [run_pair(p, prices_raw, cfg) for p in pairs]
    table = to_frame(runs)
    write_outputs(runs, table, cfg)

    view = table[(table.basis == "net") & (table.period != "full")]
    pivot = view.pivot_table(index="pair", columns="period", values="total_return")
    print("\nNET total return, in-sample vs out-of-sample (%):\n")
    print((pivot * 100).round(1).to_string())
    print(f"\nwrote {cfg.output.results_dir}/metrics.csv and figures\n")


if __name__ == "__main__":
    main()