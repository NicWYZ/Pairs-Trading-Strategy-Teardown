"""Convert the z-score signal into target positions in the spread."""

from __future__ import annotations

import pandas as pd


def target_positions(zscore: pd.Series, entry_threshold: float, exit_threshold: float) -> pd.Series:
    """
    Map z-score to a target spread position in {-1, 0, +1} with hysteresis.

      +1 (long spread)  when z < -entry_threshold   (spread unusually low -> bet it rises)
      -1 (short spread) when z >  entry_threshold    (spread unusually high -> bet it falls)
       0 (flat)         when |z| < exit_threshold    (spread back near normal -> close)
      hold previous     when exit <= |z| <= entry    (in between -> do nothing new)
      warmup / NaN z    -> flat (0)

    The 'hold previous' band is hysteresis: once in a trade you stay in it until
    the spread either reverts past the exit band or flips to the opposite entry,
    which avoids churning in and out around a single threshold.

    Position is in units of 'the spread'; the engine (Stage 4) translates it
    into leg-level trades using the hedge ratio.
    """
    pos = pd.Series(index=zscore.index, dtype="float64")
    pos[zscore > entry_threshold] = -1.0
    pos[zscore < -entry_threshold] = 1.0
    pos[zscore.abs() < exit_threshold] = 0.0
    return pos.ffill().fillna(0.0)
