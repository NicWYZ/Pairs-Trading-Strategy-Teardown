"""
Cointegration testing and spread construction for a price pair.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint

def estimate_hedge_ratio(a: pd.Series, b: pd.Series) -> float:
    """
    Estimate the hedge ratio by OLS regression of A on B (with intercept).

    IMPORTANT: for the backtest, call this on the IN-SAMPLE window only.
    Fitting it on data the strategy is later evaluated on is look-ahead bias.
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
    stationary. Applied to the spread, a low value supports cointegration.
    """
    return float(adfuller(series.dropna(), autolag = "AIC")[1])

def engle_granger_pvalue(a: pd.Series, b: pd.Series) -> float:
    """
    Engle-Granger cointegration test p-value (statsmodels `coint`).

    Low p-value (< 0.05) => the two price series are cointegrated.
    """
    return float(coint(a,b)[1])

@dataclass
class CointegrationResult:
    """
    Bundle of cointegration diagnostics for one pair.
    """
    hedge_ratio: float
    adf_pvalue: float
    eg_pvalue: float
    spread: pd.Series

def analyze_pair(a: pd.Series, b: pd.Series) -> CointegrationResult:
    """
    Run the full diagnostic suite on a price pair.
    """
    hr = estimate_hedge_ratio(a,b)
    spread = build_spread(a,b,hr)
    return CointegrationResult(
        hedge_ratio=hr,
        adf_pvalue=adf_pvalue(spread),
        eg_pvalue=engle_granger_pvalue(a,b),
        spread=spread
    )

