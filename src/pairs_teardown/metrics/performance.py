"""
Performance metrics for the pairs-trading teardown.

Everything here is a *pure function of a return or position series* — no data
downloads, no state — so the whole module is closed-form testable. The functions
are deliberately explicit about their conventions, because a Sharpe ratio or a
drawdown that silently uses a different std convention or sign is worse than
useless in a teardown whose entire point is honest measurement.

Conventions used throughout (and why):

* **Returns are simple (arithmetic) daily returns of a self-financing spread.**
  Holding +1 "spread unit" is long $1 of A and short $g of B, so the position is
  dollar-neutral and there is no natural capital base to earn a risk-free rate on.
  The Sharpe therefore defaults to a risk-free rate of 0.

* **Sample standard deviation (ddof=1).**

* **NaNs are dropped, not filled.** The first day of a ``pct_change`` series is
  NaN; filling it with 0 would inject a fake flat day into vol and hit-rate
  calculations. Dropping is the honest choice.

* **Drawdown is returned as a signed, non-positive fraction** (e.g. ``-0.23`` for
  a 23% peak-to-trough decline). Returning a signed number removes any ambiguity
  about direction when the value is tabulated next to returns.
"""

from __future__ import annotations  # type: ignore

import numpy as np
import pandas as pd

__all__ = ["sharpe_ratio", "max_drawdown", "turnover", "summary"]


def sharpe_ratio(
    returns: pd.Series,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> float:
    """
    Annualized Sharpe ratio of a simple-return series.

    Sharpe = mean(excess) / std(excess) * sqrt(periods_per_year), where the
    per-period excess return subtracts ``risk_free_rate / periods_per_year``
    from each observation. ``std`` is the sample std (ddof=1).

    Returns ``nan`` when it cannot be defined: fewer than two observations, or
    zero volatility.

    Parameters
    ----------
    returns
        Per-period simple returns. NaNs are dropped.
    periods_per_year
        Annualization factor. Use 252 for daily data, 1 to get the raw
        (un-annualized) ratio.
    risk_free_rate
        Annualized risk-free rate; converted to per-period internally.
    """
    r = pd.Series(returns).dropna()
    if len(r) < 2:
        return float("nan")
    excess = r - risk_free_rate / periods_per_year
    sd = excess.std()  # ddof=1
    if sd == 0 or np.isnan(sd):
        return float("nan")
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: pd.Series) -> float:
    """
    Maximum peak-to-trough drawdown of an equity curve, as a signed fraction.

    Computed as ``min_t (equity_t / running_max_t - 1)``. The result is <= 0;
    ``0.0`` means the curve never dropped below a prior peak. Expects an equity
    *level* series (e.g. ``(1 + net_returns).cumprod()``), not a return series.

    Returns ``nan`` for an empty curve.
    """
    eq = pd.Series(equity_curve).dropna()
    if eq.empty:
        return float("nan")
    running_max = eq.cummax()
    drawdown = eq / running_max - 1.0
    return float(drawdown.min())


def turnover(positions: pd.Series, periods_per_year: int = 252) -> float:
    """
    Annualized one-way turnover of a position series.

    Defined as ``mean(|Δposition|) * periods_per_year`` — the average per-period
    change in the held position, scaled to a yearly rate. For a discrete
    {-1, 0, +1} spread position a full round trip (0 -> 1 -> 0) contributes two
    units of absolute change, so this is a coarse but honest activity measure.

    Note this counts *changes within the sample only. The engine's cost model, not this function,
    is what applies notional-weighted (1 + |g|) costs; turnover here is a
    reporting statistic, not the thing costs are charged on.

    Returns ``nan`` when fewer than two observations are present.
    """
    pos = pd.Series(positions).dropna()
    if len(pos) < 2:
        return float("nan")
    return float(pos.diff().abs().dropna().mean() * periods_per_year)


def _perf_block(
    returns: pd.Series,
    held_positions: pd.Series,
    periods_per_year: int,
    risk_free_rate: float,
) -> dict:
    """
    Compute one bundle of scalar metrics from a return + position series.

    Factored out so the net and gross blocks in :func:`summary` share identical
    logic and can never drift apart.
    """
    r = pd.Series(returns).dropna()
    n = len(r)
    if n == 0:
        return {
            "total_return": float("nan"),
            "annualized_return": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "turnover": float("nan"),
            "hit_rate": float("nan"),
            "n_periods": 0,
        }

    equity = (1.0 + r).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)

    # CAGR from the equity curve. Guard the (1 + total) <= 0 case (a levered
    # long/short book can in principle be wiped out): a fractional power of a
    # non-positive base is not real, so report nan rather than a complex number.
    growth = 1.0 + total_return
    if growth > 0:
        annualized_return = float(growth ** (periods_per_year / n) - 1.0)
    else:
        annualized_return = float("nan")

    # Hit rate over *active* days only (days actually holding a position). A
    # hit rate over all calendar days would be dominated by flat, zero-return
    # days and would say more about how often you trade than about edge.
    active = pd.Series(held_positions).reindex(r.index).fillna(0.0) != 0
    hit_rate = float((r[active] > 0).mean()) if bool(active.any()) else float("nan")

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "sharpe": sharpe_ratio(r, periods_per_year, risk_free_rate),
        "max_drawdown": max_drawdown(equity),
        "turnover": turnover(held_positions, periods_per_year),
        "hit_rate": hit_rate,
        "n_periods": n,
    }


def summary(
    result,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> dict:
    """
    Bundle all headline metrics for a backtest, gross and net.

    ``result`` is duck-typed: any object exposing ``returns`` (net),
    ``gross_returns``, and ``held_positions`` as pandas Series works. Metrics
    are duck-typed rather than importing ``BacktestResult`` so this module has no
    dependency on ``backtest`` (avoids a circular import and lets tests pass a
    lightweight stand-in).

    Returns a nested dict ``{"net": {...}, "gross": {...}}`` where each block
    contains: total_return, annualized_return, sharpe, max_drawdown, turnover,
    hit_rate, n_periods. The nesting is deliberately shaped for direct tabulation
    into a gross-vs-net table, and can be called separately on in-sample and
    out-of-sample slices for the IS/OOS comparison.
    """
    return {
        "net": _perf_block(result.returns, result.held_positions, periods_per_year, risk_free_rate),
        "gross": _perf_block(
            result.gross_returns, result.held_positions, periods_per_year, risk_free_rate
        ),
    }
