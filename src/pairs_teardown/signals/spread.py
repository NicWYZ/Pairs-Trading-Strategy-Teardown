"""
Construct the trading spread and its standardized signal.

All inputs are expected to be LOG prices (see the cointegration finding:
logs stabilize the spread's variance over multi-year price growth).

Causality convention: every value at day t uses data only through day t,
so it is known at the close of t. Execution is delayed to t+1 by the
backtest engine, which is where look-ahead is actually prevented.
"""

from __future__ import annotations

import pandas as pd

def rolling_hedge_ratio(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
    """
    Causal rolling OLS slope over a trailing window.

    The OLS slope of A on B equals Cov(A, B) / Var(B). Computing it from
    pandas rolling Cov/Var is mathematically identical to fitting a fresh OLS
    on each trailing window, but vectorized. NaN during the warmup window.
    """
    cov = a.rolling(window).cov(b)
    var = b.rolling(window).var()
    return cov / var

def build_rolling_spread(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
    """
    Spread A - hedge_t * B using the trailing rolling hedge ratio.

    The rolling hedge ratio lets the spread adapt to a drifting relationship
    instead of assuming one fixed ratio for the whole sample.
    """
    hr = rolling_hedge_ratio(a, b, window)
    return a - hr * b

def rolling_zscore(spread: pd.Series, window: int) -> pd.Series:
    """
    Trailing-window z-score: (spread - rolling_mean) / rolling_std.

    Standardizes the spread against its own recent history, so a z of +2 means
    'two standard deviations above where this spread has recently sat'.
    """
    mean = spread.rolling(window).mean()
    std = spread.rolling(window).std()
    return (spread - mean) / std