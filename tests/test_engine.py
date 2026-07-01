"""
Tests for backtest/engine.py — including the look-ahead guard.
"""

import numpy as np
import pandas as pd

from pairs_teardown.backtest.costs import CostModel
from pairs_teardown.backtest.engine import run_backtest


def test_hand_computed_pnl():
    """
    A 4-day example worked out by hand must match to the cent.
    """
    price_a = pd.Series([100.0, 101.0, 102.0, 101.0])
    price_b = pd.Series([50.0, 50.0, 51.0, 51.0])
    target = pd.Series([1.0, 1.0, 0.0, 0.0])
    hedge = pd.Series([2.0, 2.0, 2.0, 2.0])
    costs = CostModel(commission_bps=1.0, slippage_bps=5.0)  # rate = 0.0006

    res = run_backtest(price_a, price_b, target, hedge, costs)

    # held=[0,1,1,0]; g=[nan,2,2,2]
    # r_a=[nan,.01,.0099010,-.0098039]; r_b=[nan,0,.02,0]
    # gross=[0,.01,.0099010-.04,0]; cost=[0,.0018,0,.0018]
    expected_net = pd.Series([0.0, 0.0082, 0.0099010 - 0.04, -0.0018])
    pd.testing.assert_series_equal(res.returns, expected_net, atol=1e-6, check_names=False)
    assert res.n_trades == 2


def test_no_lookahead_future_price_cannot_change_past_pnl():
    """
    Corrupting a FUTURE price must leave all earlier P&L untouched.
    """
    rng = np.random.default_rng(0)
    n = 100
    price_a = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)))
    price_b = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)))
    target = pd.Series(rng.choice([-1.0, 0.0, 1.0], n))
    hedge = pd.Series(np.full(n, 1.0))

    base = run_backtest(price_a, price_b, target, hedge, CostModel())

    T = 60
    corrupted = price_a.copy()
    corrupted.iloc[T] *= 1.20
    shocked = run_backtest(corrupted, price_b, target, hedge, CostModel())

    pd.testing.assert_series_equal(base.returns.iloc[:T], shocked.returns.iloc[:T])


def test_flat_strategy_has_zero_pnl_and_no_trades():
    price_a = pd.Series([100.0, 101.0, 102.0])
    price_b = pd.Series([50.0, 51.0, 52.0])
    target = pd.Series([0.0, 0.0, 0.0])
    hedge = pd.Series([1.0, 1.0, 1.0])
    res = run_backtest(price_a, price_b, target, hedge, CostModel())
    assert (res.returns == 0).all()
    assert res.n_trades == 0


def test_costs_reduce_returns():
    price_a = pd.Series([100.0, 101.0, 102.0, 103.0])
    price_b = pd.Series([50.0, 50.0, 50.0, 50.0])
    target = pd.Series([1.0, -1.0, 1.0, -1.0])     # churn every day
    hedge = pd.Series([1.0, 1.0, 1.0, 1.0])
    free = run_backtest(price_a, price_b, target, hedge, CostModel(0, 0))
    costly = run_backtest(price_a, price_b, target, hedge, CostModel(10, 20))
    assert costly.returns.sum() < free.returns.sum()

def test_correct_hedge_sign_neutralizes_common_move_wrong_sign_does_not():
    """
    Demonstrates why the sizing hedge ratio must be stable/correctly-signed.

    Two assets move together via a large common shock; a correctly-signed
    hedge ratio should largely cancel that shock out of the spread P&L.
    A sign-flipped hedge ratio (as can happen with a noisy rolling estimate)
    should not cancel it -- it should instead roughly DOUBLE the exposure,
    since both legs then move the same direction.
    """
    price_a = pd.Series([100.0, 110.0])
    price_b = pd.Series([100.0, 110.0])
    target = pd.Series([1.0, 1.0])

    correct_hedge = pd.Series([1.0, 1.0])
    wrong_hedge = pd.Series([-1.0, -1.0])

    res_correct = run_backtest(price_a, price_b, target, correct_hedge, CostModel(0, 0))
    res_wrong = run_backtest(price_a, price_b, target, wrong_hedge, CostModel(0, 0))

    # Correctly hedged: r_a - g*r_b = 0.10 - 1*0.10 = 0 -> ~fully neutralized
    assert abs(res_correct.returns.iloc[1]) < 1e-9

    # Sign-flipped: r_a - g*r_b = 0.10 - (-1)*0.10 = 0.20 -> exposure doubled
    assert abs(res_wrong.returns.iloc[1] - 0.20) < 1e-9

    # Wrong-signed hedge produces dramatically larger, unhedged P&L
    assert abs(res_wrong.returns.iloc[1]) > 100 * abs(res_correct.returns.iloc[1]) + 1e-9