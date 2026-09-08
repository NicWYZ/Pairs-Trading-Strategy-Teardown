"""
Tests for backtest/engine.py — including the look-ahead guard, the
hand-computed P&L check, and the sizing hedge-ratio stability guard.
"""

import numpy as np
import pandas as pd
import pytest

from pairs_teardown.backtest.costs import CostModel
from pairs_teardown.backtest.engine import UnstableHedgeRatioError, run_backtest


def test_hand_computed_pnl():
    """
    A 4-day example worked out by hand must match to the cent.
    """
    price_a = pd.Series([100.0, 101.0, 102.0, 101.0])
    price_b = pd.Series([50.0, 50.0, 51.0, 51.0])
    target = pd.Series([1.0, 1.0, 0.0, 0.0])
    hedge = pd.Series([2.0, 2.0, 2.0, 2.0])
    costs = CostModel(commission_bps=1.0, slippage_bps=5.0)
    res = run_backtest(price_a, price_b, target, hedge, costs)
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
    target = pd.Series([1.0, -1.0, 1.0, -1.0])
    hedge = pd.Series([1.0, 1.0, 1.0, 1.0])
    free = run_backtest(price_a, price_b, target, hedge, CostModel(0, 0))
    costly = run_backtest(price_a, price_b, target, hedge, CostModel(10, 20))
    assert costly.returns.sum() < free.returns.sum()


def test_correct_hedge_sign_neutralizes_common_move_wrong_sign_does_not():
    """
    Demonstrates why the sizing hedge ratio must be stable/correctly-signed.
    """
    price_a = pd.Series([100.0, 110.0])
    price_b = pd.Series([100.0, 110.0])
    target = pd.Series([1.0, 1.0])
    correct_hedge = pd.Series([1.0, 1.0])
    wrong_hedge = pd.Series([-1.0, -1.0])
    res_correct = run_backtest(price_a, price_b, target, correct_hedge, CostModel(0, 0))
    res_wrong = run_backtest(price_a, price_b, target, wrong_hedge, CostModel(0, 0))
    assert abs(res_correct.returns.iloc[1]) < 1e-9
    assert abs(res_wrong.returns.iloc[1] - 0.20) < 1e-9
    assert abs(res_wrong.returns.iloc[1]) > 100 * abs(res_correct.returns.iloc[1]) + 1e-9


# --- hedge ratio validation (threshold calibrated to real evidence) ---


def test_validate_rejects_high_variance_hedge_ratio():
    noisy = pd.Series([0.5, 1.5, 0.2, 1.8, 0.1, 2.0, 0.3])  # std ~0.82
    price_a = pd.Series([100.0] * 7)
    price_b = pd.Series([100.0] * 7)
    target = pd.Series([1.0] * 7)
    with pytest.raises(UnstableHedgeRatioError, match="std"):
        run_backtest(price_a, price_b, target, noisy, CostModel())


def test_validate_rejects_realistic_rolling_hedge_ratio_noise():
    """
    Regression test: a rolling hedge ratio with std in the range this
    project's REAL data actually produced (WM/RSG ~0.31, FOXA/FOX ~0.10)
    must be rejected -- this is the exact mistake the guard exists to catch.
    """
    rng = np.random.default_rng(0)
    realistic_noisy = pd.Series(0.95 + rng.normal(0, 0.15, 100))  # std ~0.15
    price_a = pd.Series([100.0] * 100)
    price_b = pd.Series([100.0] * 100)
    target = pd.Series([1.0] * 100)
    with pytest.raises(UnstableHedgeRatioError, match="std"):
        run_backtest(price_a, price_b, target, realistic_noisy, CostModel())


def test_validate_rejects_sign_changing_hedge_ratio():
    sign_changing = pd.Series([0.005, -0.005, 0.005, -0.005, 0.005])  # std ~0.0055
    price_a = pd.Series([100.0] * 5)
    price_b = pd.Series([100.0] * 5)
    target = pd.Series([1.0] * 5)
    with pytest.raises(UnstableHedgeRatioError, match="sign"):
        run_backtest(price_a, price_b, target, sign_changing, CostModel())


def test_validate_accepts_static_hedge_ratio():
    """
    A true static (constant) hedge ratio must always pass.
    """
    static = pd.Series([0.9336] * 100)  # e.g. WM/RSG's real static value
    price_a = pd.Series([100.0] * 100)
    price_b = pd.Series([100.0] * 100)
    target = pd.Series([1.0] * 100)
    run_backtest(price_a, price_b, target, static, CostModel())  # should not raise


def test_validate_accepts_very_stable_rolling_hedge_ratio():
    """
    SPY/VOO's real rolling hedge ratio (std ~0.0024) should still pass --
    the guard targets NOISE, not the mere fact of being a rolling estimate.
    """
    rng = np.random.default_rng(0)
    very_stable = pd.Series(0.998 + rng.normal(0, 0.002, 100))
    price_a = pd.Series([100.0] * 100)
    price_b = pd.Series([100.0] * 100)
    target = pd.Series([1.0] * 100)
    run_backtest(price_a, price_b, target, very_stable, CostModel())  # should not raise


def test_validate_ignores_nan_warmup():
    with_warmup = pd.Series([np.nan, np.nan, 1.0, 1.0, 1.0])
    price_a = pd.Series([100.0] * 5)
    price_b = pd.Series([100.0] * 5)
    target = pd.Series([1.0] * 5)
    run_backtest(price_a, price_b, target, with_warmup, CostModel())  # should not raise


def test_skip_hedge_validation_flag_bypasses_check():
    noisy = pd.Series([0.5, 1.5, 0.2, 1.8, 0.1, 2.0, 0.3])
    price_a = pd.Series([100.0] * 7)
    price_b = pd.Series([100.0] * 7)
    target = pd.Series([1.0] * 7)
    run_backtest(price_a, price_b, target, noisy, CostModel(), skip_hedge_validation=True)
