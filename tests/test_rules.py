"""
Tests for signals/rules.py.
"""

import numpy as np
import pandas as pd

from pairs_teardown.signals.rules import target_positions


def test_rules_entry_exit_and_hysteresis():
    # z path: flat -> short entry -> hold in band -> exit -> long entry
    z = pd.Series([0.0, 2.5, 1.2, 0.2, -2.5])
    pos = target_positions(z, entry_threshold=2.0, exit_threshold=0.5)
    assert list(pos) == [0.0, -1.0, -1.0, 0.0, 1.0]


def test_rules_nan_is_flat():
    z = pd.Series([np.nan, np.nan, 2.5])
    pos = target_positions(z, 2.0, 0.5)
    assert list(pos) == [0.0, 0.0, -1.0]
