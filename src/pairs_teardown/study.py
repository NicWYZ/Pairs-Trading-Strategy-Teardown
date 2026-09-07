"""
Study orchestration: run a configured pair end-to-end and tabulate the result.

This module holds the logic that ties the pipeline together — the chain from raw
prices to a metrics row. It lives in the package rather than in
``scripts/run_backtest.py`` for two reasons: notebooks import it instead of
reimplementing the chain inline (which is how ``03_backtest_explore.ipynb``
ended up with a full-sample hedge fit), and the in-sample-only discipline below
is covered by tests rather than by a docstring alone.

``scripts/run_backtest.py`` is a thin CLI and file-writing wrapper around this.

THE CENTRAL DISCIPLINE
----------------------
The static SIZING hedge ratio is estimated on IN-SAMPLE DATA ONLY, then frozen
and applied unchanged to the out-of-sample period. Fitting it on the full sample
leaks out-of-sample information into the position sizing of *every* trade,
including in-sample ones, which would invalidate the OOS result the study exists
to produce. ``test_study.py`` enforces this by mutating out-of-sample prices and
asserting the fitted ratio does not move.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from pairs_teardown.backtest.costs import CostModel
from pairs_teardown.backtest.engine import BacktestResult, run_backtest
from pairs_teardown.config import Config, Pair
from pairs_teardown.data.clean import align_prices, handle_missing
from pairs_teardown.metrics.performance import summary
from pairs_teardown.signals.rules import target_positions
from pairs_teardown.signals.spread import build_rolling_spread, rolling_zscore
from pairs_teardown.stats.cointegration import estimate_hedge_ratio

__all__ = ["PairRun", "PERIODS", "run_pair", "run_study", "slice_result", "to_frame"]

#: The three reporting periods. ``full`` is a convenience view; the study's
#: claims are always stated as in_sample vs out_of_sample (principle 5).
PERIODS = ("full", "in_sample", "out_of_sample")


class _ResultLike(Protocol):
    """The three series ``summary`` duck-types on."""

    returns: pd.Series
    gross_returns: pd.Series
    held_positions: pd.Series


@dataclass(frozen=True)
class _Sliced:
    """
    A period-restricted view of one backtest, for feeding to ``summary``.

    Slicing one simulation rather than re-running it on the OOS window is
    deliberate: a fresh run could accidentally re-fit something on OOS data, and
    it would also drop the position state carried across the split date.
    """

    returns: pd.Series
    gross_returns: pd.Series
    held_positions: pd.Series


def slice_result(result: _ResultLike, mask: pd.Series) -> _Sliced:
    """Restrict a backtest result to the rows selected by ``mask``."""
    return _Sliced(
        returns=result.returns[mask],
        gross_returns=result.gross_returns[mask],
        held_positions=result.held_positions[mask],
    )


@dataclass(frozen=True)
class PairRun:
    """
    Everything one pair produced: the simulation, its inputs, and its metrics.

    ``metrics`` is keyed by period name (see ``PERIODS``), each value being the
    ``{"net": {...}, "gross": {...}}`` block that ``summary`` returns — so the
    mandatory gross-vs-net and IS-vs-OOS tables are both a lookup away.
    """

    pair: Pair
    result: BacktestResult
    spread: pd.Series
    zscore: pd.Series
    sizing_hedge_ratio: float
    metrics: dict[str, dict[str, dict]]


def run_pair(pair: Pair, prices_raw: pd.DataFrame, cfg: Config) -> PairRun:
    """
    Run one configured pair end-to-end.

    Raises ValueError if the pair has no overlapping price data in the
    configured range, or if its in-sample window is shorter than the signal
    window (which would leave the hedge ratio fit on too little data to mean
    anything).
    """
    px = align_prices(handle_missing(prices_raw[[pair.a, pair.b]]))
    if px.empty:
        raise ValueError(f"{pair.name}: no overlapping price data in range")

    # Log prices: multiplicative growth makes level spreads heteroskedastic and
    # breaks the stationarity the whole method assumes.
    log_a = pd.Series(np.log(px[pair.a]))
    log_b = pd.Series(np.log(px[pair.b]))

    is_mask = cfg.split.is_mask(px.index)
    if is_mask.sum() < cfg.signal.window:
        raise ValueError(
            f"{pair.name}: in-sample window ({int(is_mask.sum())} days) is shorter "
            f"than signal.window ({cfg.signal.window})"
        )

    # --- SIZING: static, IN-SAMPLE ONLY, then frozen for the whole run ------
    g_static = estimate_hedge_ratio(log_a[is_mask], log_b[is_mask])
    sizing_hedge = pd.Series(g_static, index=px.index)

    # --- SIGNAL: rolling hedge -> spread -> z-score -> positions ------------
    # Rolling is ADF-justified (static-hedge spreads fail stationarity for
    # several pairs). It must never be used for sizing.
    spread = build_rolling_spread(log_a, log_b, cfg.signal.window)
    z = rolling_zscore(spread, cfg.signal.window)
    positions = target_positions(
        z,
        entry_threshold=cfg.signal.entry,
        exit_threshold=cfg.signal.exit,
    )

    # --- backtest (engine applies the one-bar lag and validates the hedge) --
    result = run_backtest(
        price_a=px[pair.a],
        price_b=px[pair.b],
        target_positions=positions,
        hedge_ratio=sizing_hedge,
        cost_model=CostModel(
            commission_bps=cfg.costs.commission_bps,
            slippage_bps=cfg.costs.slippage_bps,
        ),
    )

    ppy = cfg.backtest.periods_per_year
    masks = {
        "full": pd.Series(True, index=result.returns.index),
        "in_sample": cfg.split.is_mask(result.returns.index),
        "out_of_sample": cfg.split.oos_mask(result.returns.index),
    }
    metrics = {
        period: summary(slice_result(result, masks[period]), periods_per_year=ppy)
        for period in PERIODS
    }

    return PairRun(
        pair=pair,
        result=result,
        spread=spread,
        zscore=z,
        sizing_hedge_ratio=g_static,
        metrics=metrics,
    )


def run_study(prices_raw: pd.DataFrame, cfg: Config, official_only: bool = False) -> list[PairRun]:
    """Run every configured pair (or only the pre-specified ones)."""
    pairs = cfg.official_pairs if official_only else cfg.pairs
    return [run_pair(p, prices_raw, cfg) for p in pairs]


def to_frame(runs: list[PairRun]) -> pd.DataFrame:
    """
    Flatten runs into one long-form table: pair x period x gross/net.

    Long form rather than wide because the two mandatory comparisons — gross vs
    net, and in-sample vs out-of-sample — are then both a ``pivot_table`` away,
    with neither privileged by the storage layout.
    """
    rows = []
    for r in runs:
        for period in PERIODS:
            for basis in ("gross", "net"):
                rows.append(
                    {
                        "pair": r.pair.name,
                        "group": r.pair.group,
                        "period": period,
                        "basis": basis,
                        "hedge_ratio": r.sizing_hedge_ratio,
                        "n_trades": r.result.n_trades,
                        **r.metrics[period][basis],
                    }
                )
    return pd.DataFrame(rows)
