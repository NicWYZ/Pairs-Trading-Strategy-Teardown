"""
Chart builders for notebooks and the writeup.

Each function *returns* a ``matplotlib.figure.Figure`` and does not call
``plt.show`` or ``fig.savefig`` itself. That keeps them side-effect-free: a
notebook can display the returned figure, a script can save it to
``reports/figures/``, and a test can assert on it — without any function here
deciding for the caller. No styling beyond matplotlib defaults is imposed, so
the charts inherit whatever rcParams the caller has set.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

def plot_spread_zscore(
    spread: pd.Series,
    zscore: pd.Series,
    entry: float = 2.0,
    exit: float = 0.5,
    title: str | None = None,
):
    """
    Two stacked panels: the spread on top, its rolling z-score below.

    The z-score panel draws the entry (+/- ``entry``) and exit (+/- ``exit``)
    thresholds so the reader can see which excursions would have triggered
    trades. Shared x-axis.
    """

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, figsize=(11, 6), gridspec_kw={"height_ratios": [2, 1]}
    )

    ax_top.plot(spread.index, spread.to_numpy(), linewidth=1.0)
    ax_top.set_ylabel("spread")
    if title:
        ax_top.set_title(title)

    ax_bot.plot(zscore.index, zscore.to_numpy(), linewidth=1.0, color="black")
    for level in (entry, -entry):
        ax_bot.axhline(level, linestyle="--", linewidth=0.8, color="tab:red")
    for level in (exit, -exit):
        ax_bot.axhline(level, linestyle=":", linewidth=0.8, color="tab:green")
    ax_bot.axhline(0.0, linewidth=0.6, color="grey")
    ax_bot.set_ylabel("z-score")
    ax_bot.set_xlabel("date")

    fig.tight_layout()
    return fig


def plot_equity_curve(
    equity_curve: pd.Series,
    gross_equity: pd.Series | None = None,
    title: str | None = None,
):
    """
    Net equity curve, with an optional gross curve overlaid for cost drag.

    Plotting gross and net together makes the transaction-cost wedge visible as
    the gap between the two lines — the central quantity of the teardown.
    """

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(equity_curve.index, equity_curve.to_numpy(), linewidth=1.2, label="net")
    if gross_equity is not None:
        ax.plot(
            gross_equity.index,
            gross_equity.to_numpy(),
            linewidth=1.0,
            linestyle="--",
            color="grey",
            label="gross",
        )
        ax.legend()
    ax.axhline(1.0, linewidth=0.6, color="grey")
    ax.set_ylabel("growth of $1")
    ax.set_xlabel("date")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_drawdown(equity_curve: pd.Series, title: str | None = None):
    """
    Underwater plot: the running drawdown of an equity curve over time.

    Drawdown at each date is ``equity / running_max - 1`` (<= 0), shaded down
    from zero so the depth and duration of losing stretches are legible.
    """
    
    eq = pd.Series(equity_curve).dropna()
    drawdown = eq / eq.cummax() - 1.0

    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.fill_between(drawdown.index, drawdown.to_numpy(), 0.0, color="tab:red", alpha=0.4)
    ax.plot(drawdown.index, drawdown.to_numpy(), linewidth=0.8, color="tab:red")
    ax.set_ylabel("drawdown")
    ax.set_xlabel("date")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig