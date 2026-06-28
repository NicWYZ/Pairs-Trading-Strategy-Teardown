"""Tests for data/clean.py."""

import numpy as np
import pandas as pd
import pytest

from pairs_teardown.data.clean import align_prices, handle_missing, to_log_prices

@pytest.fixture
def prices_with_gap() -> pd.DataFrame:
    """5-day panel: ticker A has one NaN on day 3, ticker B is clean."""
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    return pd.DataFrame(
        {
            "A": [100.0, 101.0, np.nan, 103.0, 104.0],
            "B": [50.0, 51.0, 52.0, 53.0, 54.0],
        },
        index=dates,
    )

@pytest.fixture
def clean_prices() -> pd.DataFrame:
    """5-day panel with no missing values."""
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    return pd.DataFrame(
        {"A": [100.0, 101.0, 102.0, 103.0, 104.0],
         "B": [50.0, 51.0, 52.0, 53.0, 54.0]},
        index=dates,
    )

# --- align_prices ---

def test_align_drops_rows_with_any_nan(prices_with_gap: pd.DataFrame) -> None:
    result = align_prices(prices_with_gap)
    assert result.isna().sum().sum() == 0
    assert len(result) == 4  # day 3 dropped


def test_align_preserves_fully_clean_data(clean_prices: pd.DataFrame) -> None:
    result = align_prices(clean_prices)
    assert len(result) == len(clean_prices)

# --- handle_missing ---

def test_handle_missing_fills_within_max_gap(prices_with_gap: pd.DataFrame) -> None:
    filled = handle_missing(prices_with_gap, max_gap=3)
    # Day 3 of column A should be forward-filled from day 2 (101.0)
    assert filled["A"].iloc[2] == 101.0


def test_handle_missing_does_not_fill_beyond_max_gap() -> None:
    dates = pd.date_range("2020-01-01", periods=6, freq="B")
    df = pd.DataFrame(
        {"A": [100.0, np.nan, np.nan, np.nan, np.nan, 105.0]},
        index=dates,
    )
    filled = handle_missing(df, max_gap=2)
    # Positions 1 and 2 filled; positions 3 and 4 still NaN
    assert filled["A"].iloc[1] == 100.0
    assert filled["A"].iloc[2] == 100.0
    assert np.isnan(filled["A"].iloc[3])
    assert np.isnan(filled["A"].iloc[4])


def test_handle_before_align_salvages_small_gaps(prices_with_gap: pd.DataFrame) -> None:
    """Correct order: handle_missing first, then align, keeps all 5 rows."""
    result = align_prices(handle_missing(prices_with_gap, max_gap=3))
    assert len(result) == 5  # gap filled, so nothing dropped

# --- to_log_prices ---

def test_log_prices_match_numpy(clean_prices: pd.DataFrame) -> None:
    result = to_log_prices(clean_prices)
    pd.testing.assert_frame_equal(result, pd.DataFrame(np.log(clean_prices), index = clean_prices.index, columns = clean_prices.columns))