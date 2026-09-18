"""
Tests for stats/inference.py.

Every function has either a closed form (Lo's SE, Holm) or a Monte Carlo
ground truth (bootstrap coverage, expected maximum). Seeds are fixed.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pairs_teardown.stats.inference import (
    bootstrap_ci,
    expected_max_sharpe,
    holm_adjust,
    sharpe_null_sd,
    sharpe_pvalue,
    sharpe_se,
    stationary_bootstrap_indices,
)


# --------------------------------------------------------------------------- #
# Lo (2002) standard error
# --------------------------------------------------------------------------- #
def test_sharpe_se_closed_form():
    # returns = [1, 2, 3]: mean 2, sd 1 -> SR 2, T = 3.
    # SE = sqrt((1 + 2^2/2) / 3) = sqrt(1) = 1 ; no annualization with ppy=1.
    assert sharpe_se(pd.Series([1.0, 2.0, 3.0]), periods_per_year=1) == pytest.approx(1.0)


def test_sharpe_se_annualizes_by_sqrt_ppy():
    r = pd.Series([1.0, 2.0, 3.0])
    assert sharpe_se(r, periods_per_year=252) == pytest.approx(math.sqrt(252))


def test_sharpe_se_at_zero_sharpe_is_sqrt_ppy_over_T():
    # Symmetric returns -> SR = 0 exactly -> SE = sqrt(ppy / T).
    r = pd.Series([-1.0, 1.0, -1.0, 1.0])
    assert sharpe_se(r, periods_per_year=252) == pytest.approx(math.sqrt(252 / 4))
    assert sharpe_null_sd(4, 252) == pytest.approx(math.sqrt(252 / 4))


def test_sharpe_se_matches_monte_carlo_sampling_sd():
    """The formula should reproduce the spread of Sharpe estimates across samples."""
    rng = np.random.default_rng(0)
    T, n_sims = 750, 2000
    draws = rng.normal(0.0005, 0.01, size=(n_sims, T))
    srs = draws.mean(axis=1) / draws.std(axis=1, ddof=1) * math.sqrt(252)
    empirical_sd = srs.std(ddof=1)
    predicted = sharpe_se(pd.Series(draws[0]), 252)
    assert empirical_sd == pytest.approx(predicted, rel=0.1)


def test_sharpe_se_degenerate_cases_are_nan():
    assert math.isnan(sharpe_se(pd.Series([1.0])))
    assert math.isnan(sharpe_se(pd.Series([2.0, 2.0, 2.0])))


def test_sharpe_pvalue_two_sided_normal():
    assert sharpe_pvalue(1.96, 1.0) == pytest.approx(0.05, abs=1e-3)
    assert sharpe_pvalue(0.0, 1.0) == pytest.approx(1.0)
    assert math.isnan(sharpe_pvalue(float("nan"), 1.0))


# --------------------------------------------------------------------------- #
# stationary bootstrap
# --------------------------------------------------------------------------- #
def test_bootstrap_indices_shape_and_range():
    idx = stationary_bootstrap_indices(50, 5.0, 20, np.random.default_rng(0))
    assert idx.shape == (20, 50)
    assert idx.min() >= 0 and idx.max() < 50


def test_bootstrap_indices_continue_blocks_with_probability_one_minus_p():
    """
    Within a block consecutive indices step by +1 (mod n). The fraction of
    consecutive steps that are continuations should be 1 - 1/mean_block.
    """
    n, mean_block = 400, 8.0
    idx = stationary_bootstrap_indices(n, mean_block, 200, np.random.default_rng(1))
    step = (idx[:, 1:] - idx[:, :-1]) % n
    frac_continue = (step == 1).mean()
    assert frac_continue == pytest.approx(1 - 1 / mean_block, abs=0.02)


def test_mean_block_one_is_iid_resampling():
    n = 300
    idx = stationary_bootstrap_indices(n, 1.0, 100, np.random.default_rng(2))
    step = (idx[:, 1:] - idx[:, :-1]) % n
    # An iid draw lands on "previous + 1" with probability 1/n, not ~1.
    assert (step == 1).mean() < 0.02


def test_bootstrap_ci_brackets_estimate_and_narrows_with_n():
    rng = np.random.default_rng(3)
    small = pd.Series(rng.normal(0.001, 0.01, 200))
    large = pd.Series(rng.normal(0.001, 0.01, 2000))
    ci_s = bootstrap_ci(small, np.mean, n_boot=500, mean_block=5, seed=0)
    ci_l = bootstrap_ci(large, np.mean, n_boot=500, mean_block=5, seed=0)
    assert ci_s.lower <= ci_s.estimate <= ci_s.upper
    assert (ci_l.upper - ci_l.lower) < (ci_s.upper - ci_s.lower)


def test_bootstrap_ci_coverage_on_autocorrelated_data():
    """
    On an AR(1) series the block bootstrap interval for the mean should cover
    the true mean at roughly the nominal rate. (An iid bootstrap would be too
    narrow here, which is the reason for the block structure.)
    """
    rng = np.random.default_rng(4)
    n, phi, reps = 400, 0.5, 150
    covered = 0
    for _ in range(reps):
        e = rng.normal(size=n)
        x = np.empty(n)
        x[0] = e[0]
        for t in range(1, n):
            x[t] = phi * x[t - 1] + e[t]
        ci = bootstrap_ci(
            pd.Series(x), np.mean, n_boot=300, mean_block=10, seed=int(rng.integers(1e9))
        )
        covered += ci.lower <= 0.0 <= ci.upper
    coverage = covered / reps
    assert 0.85 <= coverage <= 0.99


def test_bootstrap_ci_reuses_supplied_indices():
    x = pd.Series(np.arange(20, dtype=float))
    idx = stationary_bootstrap_indices(20, 4.0, 50, np.random.default_rng(0))
    a = bootstrap_ci(x, np.mean, indices=idx)
    b = bootstrap_ci(x, np.mean, indices=idx)
    assert (a.lower, a.upper, a.n_boot) == (b.lower, b.upper, 50)


def test_bootstrap_ci_rejects_mismatched_indices():
    x = pd.Series(np.arange(20, dtype=float))
    idx = stationary_bootstrap_indices(19, 4.0, 10, np.random.default_rng(0))
    with pytest.raises(ValueError, match="length"):
        bootstrap_ci(x, np.mean, indices=idx)


# --------------------------------------------------------------------------- #
# Holm
# --------------------------------------------------------------------------- #
def test_holm_closed_form():
    # m = 4 ; sorted p = [0.01, 0.02, 0.03, 0.04]
    # multipliers 4,3,2,1 -> [0.04, 0.06, 0.06, 0.04] -> running max -> [0.04, 0.06, 0.06, 0.06]
    p = [0.03, 0.01, 0.04, 0.02]
    adj = holm_adjust(p)
    assert adj == pytest.approx([0.06, 0.04, 0.06, 0.06])


def test_holm_caps_at_one_and_keeps_nan():
    adj = holm_adjust([0.5, float("nan"), 0.9])
    assert adj[0] == pytest.approx(1.0)
    assert math.isnan(adj[1])
    assert adj[2] == pytest.approx(1.0)


def test_holm_is_never_below_raw_and_never_above_bonferroni():
    rng = np.random.default_rng(5)
    p = rng.uniform(size=10)
    adj = holm_adjust(p)
    assert (adj >= p).all()
    assert (adj <= np.minimum(1.0, p * 10) + 1e-12).all()


# --------------------------------------------------------------------------- #
# expected maximum Sharpe
# --------------------------------------------------------------------------- #
def test_expected_max_sharpe_matches_monte_carlo():
    rng = np.random.default_rng(6)
    n_trials, sd = 10, 0.58
    sims = rng.normal(0.0, sd, size=(20000, n_trials)).max(axis=1).mean()
    assert expected_max_sharpe(n_trials, sd) == pytest.approx(sims, rel=0.05)


def test_expected_max_sharpe_grows_with_trials_and_scales_with_sd():
    assert expected_max_sharpe(100, 1.0) > expected_max_sharpe(10, 1.0)
    assert expected_max_sharpe(10, 2.0) == pytest.approx(2 * expected_max_sharpe(10, 1.0))


def test_expected_max_sharpe_needs_two_trials():
    with pytest.raises(ValueError):
        expected_max_sharpe(1, 1.0)


def test_null_sd_is_lo_se_at_zero_sharpe():
    # Consistency between the two helpers, three years of daily data.
    T = 756
    assert sharpe_null_sd(T) == pytest.approx(math.sqrt(252 / T))
    assert stats.norm.sf(1.0) < 0.2  # scipy import is used; keeps the module honest
