"""
Study orchestration: run a configured pair end-to-end and tabulate the result.

This module holds the logic that ties the pipeline together — the chain from raw
prices to a metrics row.

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

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from pairs_teardown.backtest.costs import CostModel
from pairs_teardown.backtest.engine import (
    BacktestResult,
    run_backtest,
    validate_step_hedge_ratio,
)
from pairs_teardown.config import Config, Pair, WalkForwardConfig
from pairs_teardown.data.clean import align_prices, handle_missing
from pairs_teardown.metrics.performance import sharpe_ratio, summary
from pairs_teardown.signals.rules import target_positions
from pairs_teardown.signals.spread import build_rolling_spread, rolling_zscore
from pairs_teardown.stats.cointegration import build_spread, estimate_hedge_ratio, half_life
from pairs_teardown.stats.inference import (
    bootstrap_ci,
    holm_adjust,
    sharpe_pvalue,
    stationary_bootstrap_indices,
)

__all__ = [
    "PairRun",
    "PERIODS",
    "Segment",
    "inference_frame",
    "refit_dates",
    "run_pair",
    "run_pair_walk_forward",
    "run_study",
    "run_study_walk_forward",
    "segments_frame",
    "slice_result",
    "to_frame",
    "walk_forward_window",
]

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
    n_trades: int  # position changes executed inside the slice


def slice_result(result: _ResultLike, mask: pd.Series) -> _Sliced:
    """
    Restrict a backtest result to the rows selected by ``mask``.

    The trade count is taken from the *full* position series before slicing,
    so a position change on the first day of a period — against the last day
    of the previous one — is counted where it happened rather than lost at the
    boundary.
    """
    changed = result.held_positions.diff().fillna(0.0).abs() > 0
    return _Sliced(
        returns=result.returns[mask],
        gross_returns=result.gross_returns[mask],
        held_positions=result.held_positions[mask],
        n_trades=int(changed[mask].sum()),
    )


@dataclass(frozen=True)
class Segment:
    """
    One walk-forward evaluation segment: what was fitted, on what, and how it
    was traded. ``fit_end`` is the last date used in the fit, strictly before
    ``start``. ``traded`` is False when the refit produced a non-positive hedge
    ratio and the pair sat flat for the segment.
    """

    start: pd.Timestamp
    end: pd.Timestamp
    fit_end: pd.Timestamp
    hedge_ratio: float
    half_life: float
    half_life_se: float
    window: int
    traded: bool


@dataclass(frozen=True)
class PairRun:
    """
    Everything one pair produced: the simulation, its inputs, and its metrics.

    ``metrics`` is keyed by period name (see ``PERIODS``), each value being the
    ``{"net": {...}, "gross": {...}}`` block that ``summary`` returns — so the
    mandatory gross-vs-net and IS-vs-OOS tables are both a lookup away.

    ``segments`` is empty for the frozen-split run and holds one entry per
    evaluation year for the walk-forward run.
    """

    pair: Pair
    result: BacktestResult
    spread: pd.Series
    zscore: pd.Series
    sizing_hedge_ratio: float
    metrics: dict[str, dict[str, dict]]
    segments: tuple[Segment, ...] = ()


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

    # --- SIGNAL: hedge -> spread -> z-score -> positions --------------------
    # Rolling is the configured default and is ADF-justified (static-hedge
    # spreads fail stationarity for several pairs). A rolling hedge must never
    # be used for SIZING, which is why the two are separate settings.
    #
    # The static branch reuses the in-sample-only ratio fitted above rather than
    # re-fitting on everything: a "static signal" that peeked at the full sample
    # would leak, and the sensitivity analysis comparing the two branches would
    # then be comparing a clean estimator against a cheating one.
    if cfg.signal.signal_hedge == "rolling":
        spread = build_rolling_spread(log_a, log_b, cfg.signal.window)
    else:
        spread = build_spread(log_a, log_b, g_static)
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

    return PairRun(
        pair=pair,
        result=result,
        spread=spread,
        zscore=z,
        sizing_hedge_ratio=g_static,
        metrics=_period_metrics(result, cfg),
    )


def _period_masks(index: pd.Index, cfg: Config) -> dict[str, pd.Series]:
    return {
        "full": pd.Series(True, index=index),
        "in_sample": cfg.split.is_mask(index),
        "out_of_sample": cfg.split.oos_mask(index),
    }


def _period_metrics(result: BacktestResult, cfg: Config) -> dict[str, dict[str, dict]]:
    ppy = cfg.backtest.periods_per_year
    masks = _period_masks(result.returns.index, cfg)
    metrics = {}
    for period in PERIODS:
        sliced = slice_result(result, masks[period])
        metrics[period] = summary(sliced, periods_per_year=ppy)
        # A period's trade count is the trades executed *in* it. The engine's
        # full-sample count was once copied into every period row, so the
        # in-sample and out-of-sample rows reported the same number.
        for basis in ("gross", "net"):
            metrics[period][basis]["n_trades"] = sliced.n_trades
    return metrics


def run_study(prices_raw: pd.DataFrame, cfg: Config) -> list[PairRun]:
    """
    Run every configured pair.

    There is no way to run a subset. Every pair in the config is specified before
    it is run and reported whatever it does, so an API that could quietly omit
    one would be a way to launder a bad result out of the study.
    """
    return [run_pair(p, prices_raw, cfg) for p in cfg.pairs]


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
                        "period": period,
                        "basis": basis,
                        "hedge_ratio": r.sizing_hedge_ratio,
                        **r.metrics[period][basis],
                    }
                )
    return pd.DataFrame(rows)


def _total_return(x: np.ndarray) -> float:
    return float(np.prod(1.0 + x) - 1.0)


def inference_frame(runs: list[PairRun], cfg: Config) -> pd.DataFrame:
    """
    Uncertainty for every (pair, period, basis) cell of ``to_frame``.

    Per cell: the Sharpe ratio with Lo's analytic SE and normal-approximation
    p-value; stationary-bootstrap percentile intervals for the Sharpe ratio and
    the total return (block resampling of the daily return series, so the
    serial dependence of a mean-reversion strategy's P&L is kept); and the
    Holm-adjusted p-value, adjusted across the pairs *within* each period and
    basis — the family a reader scans when asking "which pair worked?".

    One set of bootstrap indices is drawn per (pair, period) and shared by the
    gross and net series, so their intervals differ only through costs, never
    through resampling noise. The scheme (``n_boot``, ``mean_block``, ``seed``)
    comes from the config, not from an argument, for the same reason the
    signal parameters do: it is part of the record of what was run.

    Notebooks read the CSV this produces rather than recomputing it, exactly as
    they do for ``metrics.csv``.
    """
    inf = cfg.inference
    ppy = cfg.backtest.periods_per_year
    rows = []
    for r in runs:
        masks = _period_masks(r.result.returns.index, cfg)
        for period in PERIODS:
            sliced = slice_result(r.result, masks[period])
            n = int(sliced.returns.dropna().shape[0])
            rng = np.random.default_rng(inf.seed)
            idx = (
                stationary_bootstrap_indices(n, inf.mean_block, inf.n_boot, rng) if n >= 2 else None
            )
            for basis, series in (("gross", sliced.gross_returns), ("net", sliced.returns)):
                block = r.metrics[period][basis]
                sr_ci = bootstrap_ci(
                    series,
                    lambda x: sharpe_ratio(pd.Series(x), ppy),
                    n_boot=inf.n_boot,
                    mean_block=inf.mean_block,
                    level=inf.ci_level,
                    seed=inf.seed,
                    indices=idx,
                )
                tr_ci = bootstrap_ci(
                    series,
                    _total_return,
                    n_boot=inf.n_boot,
                    mean_block=inf.mean_block,
                    level=inf.ci_level,
                    seed=inf.seed,
                    indices=idx,
                )
                rows.append(
                    {
                        "pair": r.pair.name,
                        "period": period,
                        "basis": basis,
                        "n_periods": n,
                        "sharpe": block["sharpe"],
                        "sharpe_se": block["sharpe_se"],
                        "sharpe_p": sharpe_pvalue(block["sharpe"], block["sharpe_se"]),
                        "sharpe_ci_lo": sr_ci.lower,
                        "sharpe_ci_hi": sr_ci.upper,
                        "total_return": block["total_return"],
                        "total_return_ci_lo": tr_ci.lower,
                        "total_return_ci_hi": tr_ci.upper,
                    }
                )

    table = pd.DataFrame(rows)
    table["sharpe_p_holm"] = np.nan
    for _, grp in table.groupby(["period", "basis"]):
        table.loc[grp.index, "sharpe_p_holm"] = holm_adjust(grp["sharpe_p"].to_numpy())
    table["ci_level"] = inf.ci_level
    table["n_boot"] = inf.n_boot
    table["mean_block"] = inf.mean_block
    return table


# --------------------------------------------------------------------------- #
# Arm B: walk-forward re-estimation (PREREGISTRATION.md)
# --------------------------------------------------------------------------- #
def refit_dates(index: pd.DatetimeIndex, in_sample_end: str) -> list[pd.Timestamp]:
    """
    The first trading day of the evaluation window, then the first trading day
    of every later calendar year inside it. Each is the start of one segment.
    """
    oos = index[index > pd.Timestamp(in_sample_end)]
    if len(oos) == 0:
        return []
    starts = [oos[0]]
    for year in range(oos[0].year + 1, oos[-1].year + 1):
        in_year = oos[oos.year == year]
        if len(in_year):
            starts.append(in_year[0])
    return [pd.Timestamp(d) for d in starts]


def walk_forward_window(half_life_days: float, wf: WalkForwardConfig) -> int:
    """
    The pre-registered window rule: ``clip(round(m * HL), lo, hi)``, with an
    infinite or undetectable half-life mapped to the cap.
    """
    if not math.isfinite(half_life_days):
        return wf.window_max
    raw = round(wf.half_life_multiple * half_life_days)
    return int(min(max(raw, wf.window_min), wf.window_max))


def run_pair_walk_forward(pair: Pair, prices_raw: pd.DataFrame, cfg: Config) -> PairRun:
    """
    Arm B: the frozen-split strategy with nothing left stale.

    Before the split the run is identical to ``run_pair`` (same in-sample hedge
    fit, same window), so in-sample metrics match Arm A exactly. From the split
    onward the evaluation window is cut into calendar-year segments. At the
    start of each, using only data strictly before it:

      * the sizing hedge ratio is re-fit by OLS on the anchored expanding window;
      * the half-life of that fitted spread sets the signal window through
        ``walk_forward_window``;
      * if the refit hedge ratio is not positive, no hedge exists and the pair
        is flat for the segment (position 0, z forced to 0 so no stale position
        is carried through by hysteresis).

    The z-score for a segment is computed on the full series with that
    segment's window (trailing data only) and spliced in; positions come from a
    single ``target_positions`` pass over the spliced z so hysteresis behaves
    across boundaries. The re-hedge itself is executed through the engine's
    one-bar lag and charged as a trade in the B leg.
    """
    px = align_prices(handle_missing(prices_raw[[pair.a, pair.b]]))
    if px.empty:
        raise ValueError(f"{pair.name}: no overlapping price data in range")
    log_a = pd.Series(np.log(px[pair.a]))
    log_b = pd.Series(np.log(px[pair.b]))
    is_mask = cfg.split.is_mask(px.index)
    if is_mask.sum() < cfg.signal.window:
        raise ValueError(
            f"{pair.name}: in-sample window ({int(is_mask.sum())} days) is shorter "
            f"than signal.window ({cfg.signal.window})"
        )

    def signal_z(g: float, window: int) -> tuple[pd.Series, pd.Series]:
        if cfg.signal.signal_hedge == "rolling":
            sp = build_rolling_spread(log_a, log_b, window)
        else:
            sp = build_spread(log_a, log_b, g)
        return sp, rolling_zscore(sp, window)

    # In-sample segment: identical to Arm A.
    g0 = estimate_hedge_ratio(log_a[is_mask], log_b[is_mask])
    spread, z = signal_z(g0, cfg.signal.window)
    z_combined = z.copy()
    spread_combined = spread.copy()
    sizing = pd.Series(g0, index=px.index)
    flat = pd.Series(False, index=px.index)

    starts = refit_dates(pd.DatetimeIndex(px.index), cfg.split.in_sample_end)
    segments: list[Segment] = []
    for k, start in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else None
        seg = (px.index >= start) & ((px.index < end) if end is not None else True)
        fit = px.index < start
        g = estimate_hedge_ratio(log_a[fit], log_b[fit])
        hl = half_life(build_spread(log_a, log_b, g)[fit])
        window = walk_forward_window(hl.half_life, cfg.walk_forward)
        traded = bool(g > 0)

        if traded:
            sp_k, z_k = signal_z(g, window)
            z_combined[seg] = z_k[seg]
            spread_combined[seg] = sp_k[seg]
        else:
            z_combined[seg] = 0.0
            flat[seg] = True
        sizing[seg] = g

        segments.append(
            Segment(
                start=start,
                end=pd.Timestamp(px.index[seg][-1]),
                fit_end=pd.Timestamp(px.index[fit][-1]),
                hedge_ratio=g,
                half_life=hl.half_life,
                half_life_se=hl.se,
                window=window,
                traded=traded,
            )
        )

    positions = target_positions(
        z_combined, entry_threshold=cfg.signal.entry, exit_threshold=cfg.signal.exit
    )
    positions[flat] = 0.0

    validate_step_hedge_ratio(sizing, starts)
    result = run_backtest(
        price_a=px[pair.a],
        price_b=px[pair.b],
        target_positions=positions,
        hedge_ratio=sizing,
        cost_model=CostModel(
            commission_bps=cfg.costs.commission_bps,
            slippage_bps=cfg.costs.slippage_bps,
        ),
        skip_hedge_validation=True,  # validated above as a step function
    )

    return PairRun(
        pair=pair,
        result=result,
        spread=spread_combined,
        zscore=z_combined,
        sizing_hedge_ratio=g0,
        metrics=_period_metrics(result, cfg),
        segments=tuple(segments),
    )


def run_study_walk_forward(prices_raw: pd.DataFrame, cfg: Config) -> list[PairRun]:
    """Arm B for every configured pair. No subset argument, as with ``run_study``."""
    return [run_pair_walk_forward(p, prices_raw, cfg) for p in cfg.pairs]


def segments_frame(runs: list[PairRun]) -> pd.DataFrame:
    """One row per (pair, segment): the record of what each refit decided."""
    rows = []
    for r in runs:
        for i, s in enumerate(r.segments):
            rows.append(
                {
                    "pair": r.pair.name,
                    "segment": i,
                    "start": s.start.date().isoformat(),
                    "end": s.end.date().isoformat(),
                    "fit_end": s.fit_end.date().isoformat(),
                    "hedge_ratio": s.hedge_ratio,
                    "half_life": s.half_life,
                    "half_life_se": s.half_life_se,
                    "window": s.window,
                    "traded": s.traded,
                }
            )
    return pd.DataFrame(rows)
