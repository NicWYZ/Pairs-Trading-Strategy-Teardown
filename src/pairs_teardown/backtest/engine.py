"""
Bar-by-bar backtest engine with a strict one-bar execution lag.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from pairs_teardown.backtest.costs import CostModel

@dataclass
class BacktestResult:
    returns: pd.Series          # net daily strategy return (after costs)
    gross_returns: pd.Series    # before costs
    equity_curve: pd.Series     # (1 + net).cumprod()
    held_positions: pd.Series   # spread position actually held each day
    costs: pd.Series            # transaction cost charged each day
    n_trades: int

def run_backtest(
    price_a: pd.Series,
    price_b: pd.Series,
    target_positions: pd.Series,
    hedge_ratio: pd.Series,
    cost_model: CostModel,
) -> BacktestResult:
    """
    Simulate the pairs strategy.

    Causality convention (no look-ahead):
      - target_positions[t] is DECIDED at the close of day t,
      - EXECUTED at t+1, so the position HELD during day t is
        target_positions[t-1]  -> we shift positions by one bar.
      - The hedge ratio sizing that position is the one known at t-1.

    P&L model: holding +1 'spread unit' = long $1 of A and short $g of B,
    so the per-unit daily return is r_A - g * r_B. Costs are charged on the
    traded notional |Δposition| * (1 + |g|) whenever the position changes.
    """

    r_a = price_a.pct_change()
    r_b = price_b.pct_change()

    held = target_positions.shift(1).fillna(0.0) #held during day t
    g = hedge_ratio.shift(1) #hedge known at t-1

    gross = (held * (r_a - g * r_b)).fillna(0.0)

    dpos = held.diff().fillna(0.0).abs() #change in held position
    traded_notional = (dpos * (1 + g.abs())).fillna(0.0)
    cost = cost_model.cost(traded_notional)

    net = gross - cost
    equity = (1 + net).cumprod()

    return BacktestResult(
        returns = net,
        gross_returns=gross,
        equity_curve=equity,
        held_positions=held,
        costs=cost,
        n_trades=int((dpos > 0).sum())
    )