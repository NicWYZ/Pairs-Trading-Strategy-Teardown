"""
Tests for stats/cointegration.py.

Uses synthetic series with known answers:
- a cointegrated pair (a = 2*b + stationary noise) where the hedge ratio
  and stationarity are known by construction;
- two independent random walks, which should NOT be cointegrated.
Seeds are fixed so the tests are deterministic, not flaky.
"""

import math

import numpy as np
import pandas as pd
import pytest

from pairs_teardown.stats.cointegration import (
    adf_pvalue,
    analyze_pair,
    build_spread,
    engle_granger_pvalue,
    estimate_hedge_ratio,
    half_life,
    johansen_trace,
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
    a, b = cointegrated_pair
    assert abs(estimate_hedge_ratio(a, b) - 2.0) < 0.1


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
    white_noise = pd.Series(rng.normal(0, 1, N))
    assert adf_pvalue(white_noise) < 0.05


# --------------------------------------------------------------------------- #
# Johansen
# --------------------------------------------------------------------------- #
def test_johansen_rejects_no_cointegration_for_cointegrated_pair(cointegrated_pair):
    a, b = cointegrated_pair
    res = johansen_trace(a, b)
    assert res.trace_stat > res.crit_99
    assert res.cointegrated_5pct


def test_johansen_does_not_reject_for_independent_walks(independent_pair):
    a, b = independent_pair
    assert not johansen_trace(a, b).cointegrated_5pct


def test_johansen_is_symmetric_in_the_legs(cointegrated_pair):
    """Unlike Engle-Granger, swapping A and B must give the identical statistic."""
    a, b = cointegrated_pair
    assert johansen_trace(a, b).trace_stat == pytest.approx(johansen_trace(b, a).trace_stat)


# --------------------------------------------------------------------------- #
# half-life
# --------------------------------------------------------------------------- #
def _ou(theta: float, n: int, seed: int) -> pd.Series:
    """Discrete OU: s_t = rho * s_{t-1} + eps, rho = exp(-theta)."""
    rng = np.random.default_rng(seed)
    rho = np.exp(-theta)
    s = np.empty(n)
    s[0] = 0.0
    for t in range(1, n):
        s[t] = rho * s[t - 1] + rng.normal()
    return pd.Series(s)


def test_half_life_recovers_the_planted_value():
    true_hl = 20.0
    res = half_life(_ou(theta=np.log(2) / true_hl, n=5000, seed=0))
    assert res.half_life == pytest.approx(true_hl, rel=0.2)
    assert abs(res.half_life - true_hl) < 3 * res.se


def test_half_life_se_shrinks_with_sample_size():
    theta = np.log(2) / 10
    short = half_life(_ou(theta, 500, 1))
    long = half_life(_ou(theta, 5000, 1))
    assert long.se < short.se


def test_half_life_of_random_walk_is_not_distinguishable_from_infinite():
    """
    A random walk does not revert, but OLS on a unit-root series is biased
    toward reversion (phi is pulled below zero -- the Dickey-Fuller bias), so
    the *point* estimate of the half-life is finite. The honest statement is
    about the interval: phi cannot be told apart from zero, so the half-life
    cannot be told apart from infinite. That is why the SE is reported.
    """
    rng = np.random.default_rng(2)
    walk = pd.Series(np.cumsum(rng.normal(size=300)))
    res = half_life(walk)
    assert res.phi + 2 * res.phi_se >= 0.0  # unit root not rejected
    assert math.isinf(res.half_life) or res.se > res.half_life  # SE swamps the estimate


def test_half_life_phi_is_the_adf_coefficient_sign():
    res = half_life(_ou(np.log(2) / 5, 2000, 3))
    assert res.phi < 0
    assert res.phi_se > 0


def test_analyze_pair_bundles_new_diagnostics(cointegrated_pair):
    a, b = cointegrated_pair
    res = analyze_pair(a, b)
    assert res.johansen.cointegrated_5pct
    assert math.isfinite(res.half_life.half_life)
