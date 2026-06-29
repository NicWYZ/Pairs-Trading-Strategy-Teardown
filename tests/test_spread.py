"""
Tests for signals/spread.py.
"""

import numpy as np
import pandas as pd
import pytest

from pairs_teardown.signals.spread import (
    build_rolling_spread,
    rolling_hedge_ratio,
    rolling_zscore,
)
from pairs_teardown.stats.cointegration import estimate_hedge_ratio

N = 300
W = 60


@pytest.fixture
def loggish_pair() -> tuple[pd.Series, pd.Series]:
    """
    A cointegrated-ish pair on a log-price scale. True slope ~1.5.
    """
    rng = np.random.default_rng(0)
    b = pd.Series(np.cumsum(rng.normal(0, 0.01, N)) + 5.0)
    a = 1.5 * b + rng.normal(0, 0.02, N)
    return a, b


def test_rolling_hedge_matches_static_ols_on_a_window(loggish_pair):
    a, b = loggish_pair
    hr = rolling_hedge_ratio(a, b, W)
    t = 200
    static = estimate_hedge_ratio(a.iloc[t - W + 1 : t + 1], b.iloc[t - W + 1 : t + 1])
    assert abs(hr.iloc[t] - static) < 1e-10


def test_rolling_hedge_warmup_is_nan(loggish_pair):
    a, b = loggish_pair
    hr = rolling_hedge_ratio(a, b, W)
    assert hr.iloc[: W - 1].isna().all()
    assert not np.isnan(hr.iloc[W - 1])


def test_zscore_is_causal(loggish_pair):
    """
    Corrupting a FUTURE price must not change any earlier z-score.
    """
    a, b = loggish_pair
    z = rolling_zscore(build_rolling_spread(a, b, W), W)
    a2 = a.copy()
    a2.iloc[250] *= 1.5
    z2 = rolling_zscore(build_rolling_spread(a2, b, W), W)
    pd.testing.assert_series_equal(z.iloc[:250], z2.iloc[:250])


def test_zscore_roughly_standardized(loggish_pair):
    a, b = loggish_pair
    z = rolling_zscore(build_rolling_spread(a, b, W), W).dropna()
    assert abs(z.mean()) < 0.5
    assert 0.5 < z.std() < 1.5