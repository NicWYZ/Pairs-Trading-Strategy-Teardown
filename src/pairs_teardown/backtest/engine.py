"""
Bar-by-bar backtest engine with a strict one-bar execution lag.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from pairs_teardown.backtest.costs import CostModel


class UnstableHedgeRatioError(ValueError):
    """
    Raised when a hedge ratio series is too volatile/sign-unstable for sizing.
    """


def validate_sizing_hedge_ratio(hedge_ratio: pd.Series, max_std: float = 0.05) -> None:
    """
    Guard against passing a noisy rolling hedge ratio into run_backtest for sizing.

    Sizing (unlike signal construction) requires stability: a hedge ratio that
    crosses zero or swings widely can silently convert a market-neutral position
    into a large directional bet on the common factor between the two assets.
    See test_engine.py::test_correct_hedge_sign_neutralizes_common_move_wrong_sign_does_not
    for the underlying mechanism.

    Raises UnstableHedgeRatioError if the series is too volatile or changes sign.
    NaNs (e.g. warmup period) are ignored.
    """
    valid = hedge_ratio.dropna()
    if valid.empty:
        return

    if valid.std() > max_std:
        raise UnstableHedgeRatioError(
            f"Sizing hedge ratio has std={valid.std():.3f} (max allowed {max_std}). "
            "Use a more stable estimate (e.g. static full-sample OLS) for trade sizing; "
            "a rolling estimate is fine for building the SIGNAL, not for SIZING the trade."
        )

    frac_positive = (valid > 0).mean()
    if not (frac_positive > 0.99 or frac_positive < 0.01):
        raise UnstableHedgeRatioError(
            f"Sizing hedge ratio changes sign ({frac_positive:.1%} of values positive). "
            "A sign-unstable hedge ratio breaks the hedge instead of neutralizing it. "
            "Use a stable estimate for trade sizing."
        )


@dataclass
class BacktestResult:
    returns: pd.Series  # net daily strategy return (after costs)
    gross_returns: pd.Series  # before costs
    equity_curve: pd.Series  # (1 + net).cumprod()
    held_positions: pd.Series  # spread position actually held each day
    costs: pd.Series  # transaction cost charged each day
    n_trades: int


def run_backtest(
    price_a: pd.Series,
    price_b: pd.Series,
    target_positions: pd.Series,
    hedge_ratio: pd.Series,
    cost_model: CostModel,
    skip_hedge_validation: bool = False,
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

    By default, validates that `hedge_ratio` is stable enough to safely size
    trades with (see validate_sizing_hedge_ratio). Pass skip_hedge_validation=True
    only for deliberate stress-testing / diagnostic work, never for production runs.
    """

    if not skip_hedge_validation:
        validate_sizing_hedge_ratio(hedge_ratio)

    r_a = price_a.pct_change()
    r_b = price_b.pct_change()

    held = target_positions.shift(1).fillna(0.0)  # held during day t
    g = hedge_ratio.shift(1)  # hedge known at t-1

    gross = (held * (r_a - g * r_b)).fillna(0.0)

    dpos = held.diff().fillna(0.0).abs()  # change in held position
    traded_notional = (dpos * (1 + g.abs())).fillna(0.0)
    cost = cost_model.cost(traded_notional)

    net = gross - cost
    equity = (1 + net).cumprod()

    return BacktestResult(
        returns=net,
        gross_returns=gross,
        equity_curve=equity,
        held_positions=held,
        costs=cost,
        n_trades=int((dpos > 0).sum()),
    )
