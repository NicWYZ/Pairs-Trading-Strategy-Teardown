"""Clean and align a raw price panel."""

from __future__ import annotations

import numpy as np
import pandas as pd

def handle_missing(df: pd.DataFrame, max_gap: int = 3) -> pd.DataFrame:
    """Forward-fill gaps of up to max_gap consecutive NaNs, then leave the rest.

    Small gaps (one-day data-vendor glitches, holidays on one exchange but
    not another) are filled using the last known price. Larger gaps are
    intentionally left as NaN so align_prices can remove them.

    Call this BEFORE align_prices.

    Parameters
    ----------
    df      : raw price DataFrame
    max_gap : maximum run of consecutive NaNs to forward-fill (default 3)
    """
    return df.ffill(limit=max_gap)

def align_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Drop any row where at least one ticker has no price (inner join on dates).

    After this call, every row in the output has a valid price for every
    column. 
    """
    return df.dropna(how="any")

def to_log_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Return natural log of prices."""
    return np.log(df)
