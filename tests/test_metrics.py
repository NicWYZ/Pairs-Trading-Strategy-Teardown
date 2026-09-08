"""
Closed-form tests for metrics/performance.py.

Every expected value here is derived by hand in the comment above the assertion,
so the tests double as a specification of the exact conventions (ddof=1 sample
std, signed drawdown, active-day hit rate, CAGR annualization).
"""

import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pairs_teardown.metrics.performance import (
    max_drawdown,
    sharpe_ratio,
    summary,
    turnover,
)


# --------------------------------------------------------------------------- #
# sharpe_ratio
# --------------------------------------------------------------------------- #
def test_sharpe_closed_form_unannualized():
    # returns = [0.01, 0.03]; mean = 0.02.
    # sample std (ddof=1) = sqrt(((0.01-0.02)^2 + (0.03-0.02)^2) / (2-1))
    #                     = sqrt((1e-4 + 1e-4)/1) = sqrt(2e-4) = 0.0141421356...
    # sharpe (periods_per_year=1) = 0.02 / 0.0141421356 = sqrt(2).
    r = pd.Series([0.01, 0.03])
    assert sharpe_ratio(r, periods_per_year=1) == pytest.approx(math.sqrt(2))


def test_sharpe_annualization_factor():
    # Same series; annualizing by 252 multiplies the raw ratio by sqrt(252).
    r = pd.Series([0.01, 0.03])
    assert sharpe_ratio(r, periods_per_year=252) == pytest.approx(math.sqrt(2) * math.sqrt(252))


def test_sharpe_zero_mean_is_zero():
    # Symmetric returns -> mean 0 -> Sharpe exactly 0.
    r = pd.Series([0.01, -0.01, 0.01, -0.01])
    assert sharpe_ratio(r, periods_per_year=1) == pytest.approx(0.0)


def test_sharpe_zero_vol_is_nan():
    # Constant series has zero std -> undefined -> nan (not inf, not 0).
    r = pd.Series([0.005, 0.005, 0.005])
    assert math.isnan(sharpe_ratio(r))


def test_sharpe_too_few_points_is_nan():
    assert math.isnan(sharpe_ratio(pd.Series([0.01])))
    assert math.isnan(sharpe_ratio(pd.Series([], dtype=float)))


def test_sharpe_drops_nan_before_computing():
    # A leading NaN (as from pct_change) must not change the answer.
    r = pd.Series([np.nan, 0.01, 0.03])
    assert sharpe_ratio(r, periods_per_year=1) == pytest.approx(math.sqrt(2))


def test_sharpe_risk_free_subtracted():
    # periods_per_year=1 so rf is subtracted whole from each observation.
    # returns [0.02, 0.04], rf=0.02 -> excess [0.00, 0.02], mean 0.01,
    # std(ddof=1) = sqrt(((0-0.01)^2+(0.02-0.01)^2)/1) = sqrt(2e-4) = 0.01414214
    # sharpe = 0.01 / 0.01414214 = sqrt(2)/2 = 1/sqrt(2).
    r = pd.Series([0.02, 0.04])
    assert sharpe_ratio(r, periods_per_year=1, risk_free_rate=0.02) == pytest.approx(
        1 / math.sqrt(2)
    )


# --------------------------------------------------------------------------- #
# max_drawdown
# --------------------------------------------------------------------------- #
def test_max_drawdown_closed_form():
    # equity           = [1.0, 1.2, 0.9, 1.0, 0.6]
    # running max      = [1.0, 1.2, 1.2, 1.2, 1.2]
    # drawdown         = [0,   0,  -0.25, -0.1667, -0.5]
    # max drawdown = -0.5 (the 0.6 vs prior peak 1.2).
    eq = pd.Series([1.0, 1.2, 0.9, 1.0, 0.6])
    assert max_drawdown(eq) == pytest.approx(-0.5)


def test_max_drawdown_monotonic_up_is_zero():
    eq = pd.Series([1.0, 1.1, 1.2, 1.3])
    assert max_drawdown(eq) == pytest.approx(0.0)


def test_max_drawdown_empty_is_nan():
    assert math.isnan(max_drawdown(pd.Series([], dtype=float)))


# --------------------------------------------------------------------------- #
# turnover
# --------------------------------------------------------------------------- #
def test_turnover_closed_form():
    # positions = [0, 1, 1, 0, -1, 0]
    # diff      = [nan, 1, 0, -1, -1, 1]  -> abs (drop nan) = [1, 0, 1, 1, 1]
    # mean = 4/5 = 0.8 ; with periods_per_year=1 -> 0.8.
    pos = pd.Series([0, 1, 1, 0, -1, 0])
    assert turnover(pos, periods_per_year=1) == pytest.approx(0.8)


def test_turnover_annualized():
    pos = pd.Series([0, 1, 1, 0, -1, 0])
    assert turnover(pos, periods_per_year=252) == pytest.approx(0.8 * 252)


def test_turnover_flat_is_zero():
    pos = pd.Series([0, 0, 0, 0])
    assert turnover(pos, periods_per_year=1) == pytest.approx(0.0)


def test_turnover_too_few_points_is_nan():
    assert math.isnan(turnover(pd.Series([1])))


# --------------------------------------------------------------------------- #
# summary
# --------------------------------------------------------------------------- #
def _make_result():
    # net returns   : [0.00, 0.10, -0.05]
    #   equity = [1.00, 1.10, 1.045]  -> total = +0.045
    # gross returns : [0.00, 0.12, -0.03]
    #   equity = [1.00, 1.12, 1.0864] -> total = +0.0864
    # held positions: [0, 1, 1]  -> active days are indices 1 and 2.
    idx = pd.RangeIndex(3)
    return SimpleNamespace(
        returns=pd.Series([0.00, 0.10, -0.05], index=idx),
        gross_returns=pd.Series([0.00, 0.12, -0.03], index=idx),
        held_positions=pd.Series([0, 1, 1], index=idx),
    )


def test_summary_total_returns():
    s = summary(_make_result(), periods_per_year=1)
    # net: 1.10 * 0.95 - 1 = 0.045 ; gross: 1.12 * 0.97 - 1 = 0.0864
    assert s["net"]["total_return"] == pytest.approx(0.045)
    assert s["gross"]["total_return"] == pytest.approx(0.0864)


def test_summary_hit_rate_active_days_only():
    s = summary(_make_result(), periods_per_year=1)
    # Active days: returns 0.10 (>0, hit) and -0.05 (<=0, miss) -> 1/2 = 0.5.
    # The flat day-0 (return 0.00, position 0) is correctly excluded.
    assert s["net"]["hit_rate"] == pytest.approx(0.5)


def test_summary_annualized_return_cagr():
    s = summary(_make_result(), periods_per_year=1)
    # With periods_per_year == n_periods == ... here n=3, ppy=1:
    # CAGR = (1.045)^(1/3) - 1.
    assert s["net"]["annualized_return"] == pytest.approx(1.045 ** (1 / 3) - 1)


def test_summary_reports_n_periods():
    s = summary(_make_result())
    assert s["net"]["n_periods"] == 3
    assert s["gross"]["n_periods"] == 3


def test_summary_gross_net_share_turnover():
    # Turnover depends only on positions, so gross and net blocks must agree.
    s = summary(_make_result())
    assert s["net"]["turnover"] == pytest.approx(s["gross"]["turnover"])
