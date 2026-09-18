"""
Statistical inference for backtest results.

The rest of the package produces point estimates; this module attaches
uncertainty to them and corrects for the number of things being looked at.
Everything is a pure function of a return series (or a vector of p-values), so
every function is testable against a closed form or a Monte Carlo experiment.

Three questions, three tools:

1. **How precise is one pair's Sharpe ratio?**
   ``sharpe_se`` — Lo (2002), the delta-method standard error of the Sharpe
   ratio under i.i.d. returns; ``bootstrap_ci`` with ``stationary_bootstrap_indices``
   — Politis & Romano (1994), a nonparametric interval that keeps the serial
   dependence of daily strategy returns, which the i.i.d. formula ignores.
   Where the two disagree, the bootstrap is the one to trust.

2. **Ten pairs were tested; which results survive that?**
   ``holm_adjust`` — Holm (1979) step-down familywise-error correction. It
   makes no independence assumption, which matters because the pairs share a
   market factor.

3. **Is the best pair's Sharpe more than the best of N coin flips?**
   ``expected_max_sharpe`` — Bailey & López de Prado (2014): the expected
   maximum Sharpe ratio among N strategies with no skill, given the sampling
   noise of each. The observed maximum should be judged against this, not
   against zero.

References
----------
Lo, A. W. (2002). The statistics of Sharpe ratios. *Financial Analysts Journal*.
Politis, D. N. & Romano, J. P. (1994). The stationary bootstrap. *JASA*.
Holm, S. (1979). A simple sequentially rejective multiple test procedure.
    *Scandinavian Journal of Statistics*.
Bailey, D. H. & López de Prado, M. (2014). The deflated Sharpe ratio.
    *Journal of Portfolio Management*.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "BootstrapCI",
    "bootstrap_ci",
    "expected_max_sharpe",
    "holm_adjust",
    "sharpe_null_sd",
    "sharpe_pvalue",
    "sharpe_se",
    "stationary_bootstrap_indices",
]

_EULER_GAMMA = 0.5772156649015329


# --------------------------------------------------------------------------- #
# 1a. Analytic Sharpe standard error (Lo 2002)
# --------------------------------------------------------------------------- #
def sharpe_se(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Standard error of the annualized Sharpe ratio under i.i.d. returns.

    Lo (2002), eq. (11): for per-period Sharpe ``SR`` estimated from ``T``
    observations,

        ``SE(SR) = sqrt((1 + SR^2 / 2) / T)``,

    obtained by the delta method from the joint asymptotic normality of the
    sample mean and variance. Annualizing multiplies both ``SR`` and its SE by
    ``sqrt(periods_per_year)``. The ``SR^2/2`` term is the contribution of the
    estimated denominator; for a strategy near zero Sharpe it is negligible and
    the SE collapses to ``sqrt(periods_per_year / T)`` — for three years of
    daily data, about 0.58 on an annual Sharpe, which is the single most useful
    number to keep in mind when reading a backtest table.

    The i.i.d. assumption is the weak point for a mean-reversion strategy whose
    daily returns are serially correlated; ``bootstrap_ci`` is the check.
    Returns ``nan`` with fewer than two observations or zero volatility.
    """
    r = pd.Series(returns).dropna().to_numpy(dtype=float)
    n = len(r)
    if n < 2:
        return math.nan
    sd = r.std(ddof=1)
    if sd == 0 or math.isnan(sd):
        return math.nan
    sr = r.mean() / sd
    return float(math.sqrt((1.0 + 0.5 * sr**2) / n) * math.sqrt(periods_per_year))


def sharpe_null_sd(n_obs: int, periods_per_year: int = 252) -> float:
    """
    Sampling standard deviation of an annualized Sharpe ratio when the true
    Sharpe is zero: ``sqrt(periods_per_year / n_obs)``. This is Lo's SE at
    ``SR = 0`` and is the noise scale that ``expected_max_sharpe`` needs.
    """
    if n_obs < 1:
        return math.nan
    return math.sqrt(periods_per_year / n_obs)


def sharpe_pvalue(sharpe: float, se: float) -> float:
    """
    Two-sided p-value for ``H0: Sharpe = 0`` from the normal approximation
    ``sharpe / se ~ N(0, 1)``. ``nan`` if either input is not finite.
    """
    if not (math.isfinite(sharpe) and math.isfinite(se)) or se <= 0:
        return math.nan
    return float(2.0 * stats.norm.sf(abs(sharpe) / se))


# --------------------------------------------------------------------------- #
# 1b. Stationary bootstrap (Politis & Romano 1994)
# --------------------------------------------------------------------------- #
def stationary_bootstrap_indices(
    n: int, mean_block: float, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Resampling indices for the stationary bootstrap, shape ``(n_boot, n)``.

    Each resample is built from blocks of consecutive original indices. Block
    lengths are geometric with mean ``mean_block`` (so ``p = 1 / mean_block``
    is the per-step probability of starting a new block at a uniformly random
    position); a block that runs off the end wraps around. Random block
    lengths — rather than the fixed length of the circular block bootstrap —
    are what make every resampled series stationary, hence the name.

    Block resampling preserves the serial dependence of the original series
    within a block, which an i.i.d. bootstrap would destroy. ``mean_block`` is
    a tuning parameter: too small reverts to i.i.d., too large leaves few
    effective resamples. ``mean_block = 1`` is exactly the i.i.d. bootstrap.
    """
    if n < 1 or n_boot < 1:
        raise ValueError("n and n_boot must be positive")
    if mean_block < 1:
        raise ValueError("mean_block must be >= 1")
    p = 1.0 / mean_block

    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, size=n_boot)
    new_block = rng.random((n_boot, n)) < p
    fresh_start = rng.integers(0, n, size=(n_boot, n))
    for t in range(1, n):
        idx[:, t] = np.where(new_block[:, t], fresh_start[:, t], (idx[:, t - 1] + 1) % n)
    return idx


@dataclass(frozen=True)
class BootstrapCI:
    """A point estimate with a percentile bootstrap interval."""

    estimate: float
    lower: float
    upper: float
    level: float
    n_boot: int


def bootstrap_ci(
    x: pd.Series,
    statistic: Callable[[np.ndarray], float],
    *,
    n_boot: int = 2000,
    mean_block: float = 10.0,
    level: float = 0.95,
    seed: int = 0,
    indices: np.ndarray | None = None,
) -> BootstrapCI:
    """
    Percentile bootstrap interval for ``statistic(x)`` under the stationary
    bootstrap.

    ``statistic`` maps a 1-D float array to a scalar. Pass ``indices`` (from
    ``stationary_bootstrap_indices``) to reuse one set of resamples across
    several statistics of the same series, which keeps their intervals
    comparable and avoids regenerating the index matrix.

    The percentile interval is the simplest bootstrap interval and is first-
    order correct; it is used here because the study's question is "is zero
    inside it", not the third decimal of the endpoint. NaNs in ``x`` are
    dropped before resampling.
    """
    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0, 1)")
    arr = pd.Series(x).dropna().to_numpy(dtype=float)
    n = len(arr)
    if n < 2:
        return BootstrapCI(math.nan, math.nan, math.nan, level, 0)

    if indices is None:
        indices = stationary_bootstrap_indices(n, mean_block, n_boot, np.random.default_rng(seed))
    elif indices.shape[1] != n:
        raise ValueError(f"indices have length {indices.shape[1]} but series has {n}")

    draws = np.array([statistic(arr[row]) for row in indices], dtype=float)
    draws = draws[np.isfinite(draws)]
    alpha = 1.0 - level
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2]) if len(draws) else (math.nan, math.nan)
    return BootstrapCI(
        estimate=float(statistic(arr)),
        lower=float(lo),
        upper=float(hi),
        level=level,
        n_boot=int(indices.shape[0]),
    )


# --------------------------------------------------------------------------- #
# 2. Multiple testing (Holm 1979)
# --------------------------------------------------------------------------- #
def holm_adjust(pvalues: np.ndarray | pd.Series | list[float]) -> np.ndarray:
    """
    Holm step-down adjusted p-values, controlling the familywise error rate.

    Sort the ``m`` p-values ascending; the ``i``-th smallest is multiplied by
    ``(m - i + 1)``, the running maximum is taken so adjusted values are
    monotone in the raw ones, and everything is capped at 1. Rejecting at
    ``adj_p < alpha`` is then equivalent to Holm's sequential procedure. It is
    uniformly more powerful than Bonferroni and, unlike Benjamini–Hochberg,
    needs no assumption about the dependence between tests — which matters
    here because ten pairs of US large caps are not independent.

    NaN inputs are left NaN and excluded from ``m``.
    """
    p = np.asarray(pvalues, dtype=float)
    out = np.full(p.shape, np.nan)
    valid = np.isfinite(p)
    m = int(valid.sum())
    if m == 0:
        return out
    order = np.argsort(p[valid])
    sorted_p = p[valid][order]
    adj = np.minimum(1.0, np.maximum.accumulate(sorted_p * (m - np.arange(m))))
    adj_unsorted = np.empty(m)
    adj_unsorted[order] = adj
    out[valid] = adj_unsorted
    return out


# --------------------------------------------------------------------------- #
# 3. Selection bias: expected maximum of N null Sharpe ratios
# --------------------------------------------------------------------------- #
def expected_max_sharpe(n_trials: int, sharpe_sd: float) -> float:
    """
    Expected maximum of ``n_trials`` independent Sharpe ratios that are each
    pure noise with standard deviation ``sharpe_sd``.

    Bailey & López de Prado (2014), eq. (3), from the asymptotics of the
    maximum of ``N`` i.i.d. normals:

        ``E[max] ≈ sd * [ (1 − γ) Φ⁻¹(1 − 1/N) + γ Φ⁻¹(1 − 1/(N e)) ]``

    with ``γ`` the Euler–Mascheroni constant. The point of the number: if the
    best of ten strategies has a Sharpe below this, "the best pair did well" is
    what noise looks like, not evidence of edge. ``sharpe_sd`` for a
    strategy with no skill is ``sharpe_null_sd(n_obs)``.

    The approximation assumes independent trials; correlated trials have a
    *smaller* expected maximum, so this is conservative in the sense of being
    generous to the strategy. Requires ``n_trials >= 2``.
    """
    if n_trials < 2:
        raise ValueError("expected_max_sharpe needs at least two trials")
    if not math.isfinite(sharpe_sd) or sharpe_sd < 0:
        return math.nan
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(sharpe_sd * ((1.0 - _EULER_GAMMA) * z1 + _EULER_GAMMA * z2))
