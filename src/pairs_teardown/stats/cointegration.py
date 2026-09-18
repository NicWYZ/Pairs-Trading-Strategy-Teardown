"""
Cointegration testing and spread construction for a price pair.

Two families of test are provided, because they answer the same question with
different assumptions and a reader should be able to see whether they agree:

* **Engle–Granger** (``engle_granger_pvalue``) — residual-based, single
  equation. Regress A on B, test the residual for a unit root. It is the
  textbook test and the one with a p-value, but it is asymmetric (regressing
  B on A can give a different answer) and its residual is, by construction,
  the *most stationary-looking* linear combination of the two series. That
  is why ``adf_pvalue`` on an OLS spread uses the wrong critical values —
  see its docstring.
* **Johansen** (``johansen_trace``) — system-based, symmetric in A and B,
  and tests the rank of the cointegrating space directly. It has no
  normalisation choice to make and so no asymmetry to worry about.

Beyond "is it cointegrated", the quantity a trader actually needs is *how
fast* the spread reverts — ``half_life`` — because that is what determines
whether a given lookback window and cost level can ever be paid for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def estimate_hedge_ratio(a: pd.Series, b: pd.Series) -> float:
    """
    Estimate the hedge ratio by OLS regression of A on B (with intercept).

    IMPORTANT: for the backtest, call this on the IN-SAMPLE window only.
    Fitting it on data the strategy is later evaluated on is look-ahead bias.

    OLS is asymmetric: the slope of A on B is not the reciprocal of the slope
    of B on A unless the fit is perfect. The study fixes the direction (A on B,
    with A the first-named leg of every pair) before any test is run, so the
    asymmetry is a documented convention rather than a degree of freedom.
    """
    model = sm.OLS(a, sm.add_constant(b)).fit()
    return float(model.params.iloc[1])


def build_spread(a: pd.Series, b: pd.Series, hedge_ratio: float) -> pd.Series:
    """
    Construct the spread A - hedge_ratio * B.

    Its (non-zero) mean is handled downstream by the rolling z-score, which
    subtracts a trailing mean, so we do not subtract the intercept here.
    """
    return a - hedge_ratio * b


def adf_pvalue(series: pd.Series) -> float:
    """
    Augmented Dickey-Fuller p-value for stationarity of a series.

    Low p-value (< 0.05) => reject the unit-root null => the series is
    stationary.

    CAVEAT — this is the *standard* ADF test, with critical values derived for
    an observed series. Applied to a spread whose hedge ratio was itself fitted
    by OLS on the same data, it is **too liberal**: OLS chooses the linear
    combination with the smallest residual variance, which is also the one that
    looks most stationary, so the null is rejected too often (Engle & Granger
    1987; Phillips & Ouliaris 1990). ``engle_granger_pvalue`` uses the
    MacKinnon critical values that correct for this and is the one to cite for a
    cointegration claim. ``adf_pvalue`` on a fitted spread is a descriptive
    diagnostic, and on a *rolling*-hedge spread — which refits the level it then
    measures deviations from — it is close to meaningless as a test.
    """
    return float(adfuller(series.dropna(), autolag="AIC")[1])


def engle_granger_pvalue(a: pd.Series, b: pd.Series) -> float:
    """
    Engle-Granger cointegration test p-value (statsmodels `coint`).

    Low p-value (< 0.05) => the two price series are cointegrated. The
    p-value uses MacKinnon's cointegration critical values, which account for
    the hedge ratio having been estimated (see ``adf_pvalue``).
    """
    return float(coint(a, b)[1])


@dataclass(frozen=True)
class JohansenResult:
    """
    Johansen trace test of ``rank = 0`` (no cointegration) for a pair.

    ``trace_stat`` is compared against the tabulated critical values; rejecting
    rank 0 means at least one cointegrating vector exists. For two series that
    is the only relation possible, so "rank >= 1" is "cointegrated". statsmodels
    tabulates critical values rather than p-values, so the result is reported at
    the three conventional levels.
    """

    trace_stat: float
    crit_90: float
    crit_95: float
    crit_99: float

    @property
    def cointegrated_5pct(self) -> bool:
        return self.trace_stat > self.crit_95


def johansen_trace(a: pd.Series, b: pd.Series, lags: int = 1) -> JohansenResult:
    """
    Johansen trace test for the pair, with a constant in the cointegrating
    relation (matching the intercept in ``estimate_hedge_ratio``) and ``lags``
    lagged differences in the VECM.

    Unlike Engle–Granger this is symmetric in (a, b): swapping the legs gives
    the identical statistic.
    """
    df = pd.concat([a, b], axis=1).dropna()
    res = coint_johansen(df.to_numpy(), det_order=0, k_ar_diff=lags)
    return JohansenResult(
        trace_stat=float(res.lr1[0]),
        crit_90=float(res.cvt[0, 0]),
        crit_95=float(res.cvt[0, 1]),
        crit_99=float(res.cvt[0, 2]),
    )


@dataclass(frozen=True)
class HalfLifeResult:
    """
    Speed of mean reversion of a spread, from an AR(1) fit.

    ``half_life`` is in the sampling units of the series (trading days here);
    ``inf`` means the fitted process does not mean-revert. ``se`` is a
    delta-method standard error, ``nan`` when the half-life is infinite.
    ``phi`` is the coefficient on the lagged level in the regression of the
    change on the level; ``phi < 0`` is mean reversion.
    """

    half_life: float
    se: float
    phi: float
    phi_se: float


def half_life(spread: pd.Series) -> HalfLifeResult:
    """
    Half-life of mean reversion via the discrete-time Ornstein–Uhlenbeck fit.

    Model: ``Δs_t = α + φ s_{t-1} + ε_t``. This is the AR(1) ``s_t = c + ρ s_{t-1}
    + ε_t`` with ``ρ = 1 + φ``, and it is the Euler discretisation of the OU
    process ``ds = θ(μ − s)dt + σ dW`` with ``ρ = e^{−θ}``. A deviation decays to
    half its size after

        ``half_life = ln 2 / θ = −ln 2 / ln(1 + φ)``.

    The standard error comes from the delta method applied to the OLS SE of φ:
    ``d(half_life)/dφ = ln 2 / ((1 + φ) ln(1 + φ)^2)``. It is an asymptotic,
    homoskedastic SE — good enough to say whether "20 days" means 15–25 or
    5–100, which is the question that matters here.

    Note the regression is the ADF regression with no augmentation lags, so φ
    is also the ADF coefficient; the two tools look at the same number from
    different angles (is it zero? / how big is it?).
    """
    s = spread.dropna()
    lagged = s.shift(1).iloc[1:]
    delta = s.diff().iloc[1:]
    # has_constant="add": the default ("skip") drops the intercept when the
    # regressor is itself constant, leaving one column and no phi to read. An
    # exactly-constant spread then raises instead of reporting "no reversion".
    design = sm.add_constant(lagged.to_numpy(), has_constant="add")
    fit = sm.OLS(delta.to_numpy(), design).fit()
    phi = float(fit.params[1])
    phi_se = float(fit.bse[1])

    rho = 1.0 + phi
    if rho <= 0 or rho >= 1:
        # rho >= 1: no reversion (unit root or explosive) -> infinite half-life.
        # rho <= 0: oscillatory over-correction; the OU half-life is not defined.
        return HalfLifeResult(half_life=math.inf, se=math.nan, phi=phi, phi_se=phi_se)

    hl = -math.log(2.0) / math.log(rho)
    d_hl = math.log(2.0) / (rho * math.log(rho) ** 2)
    return HalfLifeResult(half_life=hl, se=abs(d_hl) * phi_se, phi=phi, phi_se=phi_se)


@dataclass
class CointegrationResult:
    """
    Bundle of cointegration diagnostics for one pair.
    """

    hedge_ratio: float
    adf_pvalue: float
    eg_pvalue: float
    spread: pd.Series
    johansen: JohansenResult
    half_life: HalfLifeResult


def analyze_pair(a: pd.Series, b: pd.Series) -> CointegrationResult:
    """
    Run the full diagnostic suite on a price pair.
    """
    hr = estimate_hedge_ratio(a, b)
    spread = build_spread(a, b, hr)
    return CointegrationResult(
        hedge_ratio=hr,
        adf_pvalue=adf_pvalue(spread),
        eg_pvalue=engle_granger_pvalue(a, b),
        spread=spread,
        johansen=johansen_trace(a, b),
        half_life=half_life(spread),
    )


__all__ = [
    "CointegrationResult",
    "HalfLifeResult",
    "JohansenResult",
    "adf_pvalue",
    "analyze_pair",
    "build_spread",
    "engle_granger_pvalue",
    "estimate_hedge_ratio",
    "half_life",
    "johansen_trace",
]
