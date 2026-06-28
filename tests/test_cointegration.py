"""
Tests for stats/cointegration.py.

Uses synthetic series with known answers:
- a cointegrated pair (a = 2*b + stationary noise) where the hedge ratio
  and stationarity are known by construction;
- two independent random walks, which should NOT be cointegrated.
Seeds are fixed so the tests are deterministic, not flaky.
"""

import numpy as np
import pandas as pd
import pytest

from pairs_teardown.stats.cointegration import (
    adf_pvalue,
    analyze_pair,
    build_spread,
    estimate_hedge_ratio,
    engle_granger_pvalue,
)

N = 500

@pytest.fixture
def cointegrated_pair() -> tuple[pd.Series, pd.Series]:
    """
    a = 2*b + white noise; b is a random walk. True hedge ratio = 2.
    """
    rng = np.random.default_rng(0)
    b = pd.Series(np.cumsum(rng.normal(0, 1, N)) + 100.0)
    a = 2.0 * b + rng.normal(0, 1.0, N)
    return a, b


@pytest.fixture
def independent_pair() -> tuple[pd.Series, pd.Series]:
    """
    Two independent random walks: not cointegrated.
    """
    rng = np.random.default_rng(0)
    a = pd.Series(np.cumsum(rng.normal(0, 1, N)) + 100.0)
    b = pd.Series(np.cumsum(rng.normal(0, 1, N)) + 100.0)
    return a, b

def test_hedge_ratio_recovers_known_slope(cointegrated_pair):
    a,b = cointegrated_pair
    assert abs(estimate_hedge_ratio(a,b) - 2.0) < 0.1

def test_cointegrated_pair_has_low_eg_pvalue(cointegrated_pair):
    a, b = cointegrated_pair
    assert engle_granger_pvalue(a, b) < 0.05


def test_cointegrated_pair_spread_is_stationary(cointegrated_pair):
    a, b = cointegrated_pair
    hr = estimate_hedge_ratio(a, b)
    spread = build_spread(a, b, hr)
    assert adf_pvalue(spread) < 0.05


def test_independent_pair_has_high_eg_pvalue(independent_pair):
    a, b = independent_pair
    assert engle_granger_pvalue(a, b) > 0.10

def test_independent_pair_spread_is_nonstationary(independent_pair):
    a, b = independent_pair
    hr = estimate_hedge_ratio(a, b)
    spread = build_spread(a, b, hr)
    assert adf_pvalue(spread) > 0.10

def test_build_spread_matches_formula():
    a = pd.Series([10.0, 20.0, 30.0])
    b = pd.Series([1.0, 2.0, 3.0])
    spread = build_spread(a, b, hedge_ratio=2.0)
    expected = pd.Series([8.0, 16.0, 24.0])
    pd.testing.assert_series_equal(spread, expected)

def test_adf_pvalue_low_for_stationary_series():
    rng = np.random.default_rng(0)
    white_noise = pd.Series(rng.normal(0,1,N))
    assert adf_pvalue(white_noise) < 0.05
