"""
Tests for backtest/costs.py.
"""

from pairs_teardown.backtest.costs import CostModel


def test_rate_combines_commission_and_slippage():
    assert CostModel(commission_bps=1.0, slippage_bps=5.0).rate == 0.0006


def test_zero_cost_model():
    assert CostModel(0, 0).rate == 0.0


def test_cost_scales_linearly_with_notional():
    cm = CostModel(commission_bps=2.0, slippage_bps=3.0)  # rate 0.0005
    assert cm.cost(1000) == 0.5
    assert cm.cost(2000) == 1.0
