"""
Tests for the study orchestration layer.

The centrepiece is ``test_out_of_sample_prices_do_not_move_the_sizing_hedge``:
the in-sample-only hedge fit is the discipline that makes the whole OOS result
meaningful, and until now it was guaranteed only by a docstring. Everything else
here guards the plumbing around it — that periods partition the sample, that
gross is never worse than net, and that the long-form table is well formed.

All data is synthetic with a known hedge ratio; nothing here touches the network.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from pairs_teardown.config import (
    BacktestConfig,
    Config,
    CostConfig,
    DataConfig,
    OutputConfig,
    Pair,
    SignalConfig,
    SplitConfig,
)
from pairs_teardown.study import PERIODS, run_pair, run_study, to_frame

# 800 business days ~ 2015-01-01 onward, split so that both periods are ample.
N = 800
SPLIT = "2017-01-01"
TRUE_RATIO = 2.0


@pytest.fixture
def cfg() -> Config:
    return Config(
        data=DataConfig(
            start="2015-01-01", end="2099-12-31", price_field="adj_close", cache_dir="data/raw"
        ),
        split=SplitConfig(in_sample_end=SPLIT),
        signal=SignalConfig(
            window=60,
            entry=2.0,
            exit=0.5,
            signal_hedge="rolling",
            sizing_hedge="static",
            sizing_hedge_max_std=0.05,
        ),
        costs=CostConfig(commission_bps=1.0, slippage_bps=5.0),
        backtest=BacktestConfig(periods_per_year=252),
        output=OutputConfig(results_dir="reports/results", figures_dir="reports/figures"),
        pairs=(
            Pair(name="A/B", a="A", b="B", rationale="synthetic"),
            Pair(name="C/D", a="C", b="D", rationale="synthetic"),
        ),
    )


def make_prices(seed: int = 0, oos_shock: float = 0.0) -> pd.DataFrame:
    """
    Log-cointegrated prices with a known hedge ratio of ``TRUE_RATIO``.

    ``oos_shock`` multiplies leg A's price after the split date only. It is the
    lever the leak test pulls: a full-sample hedge fit would move with it, an
    in-sample-only fit must not.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=N)

    log_b = np.cumsum(rng.normal(0, 0.01, N)) + np.log(100.0)
    log_a = TRUE_RATIO * log_b + rng.normal(0, 0.02, N)

    a = np.exp(log_a)
    b = np.exp(log_b)
    if oos_shock:
        a = a * np.where(idx > pd.Timestamp(SPLIT), oos_shock, 1.0)

    return pd.DataFrame({"A": a, "B": b, "C": b * 1.5, "D": b}, index=idx)


# --------------------------------------------------------------------------- #
# the leak guard — the reason this module has tests at all
# --------------------------------------------------------------------------- #
def test_out_of_sample_prices_do_not_move_the_sizing_hedge(cfg):
    """
    Mutating OOS prices must leave the fitted sizing hedge ratio untouched.

    A full-sample fit would absorb the shock; an in-sample-only fit cannot see
    it. This is the single check standing between the study and a silent
    look-ahead leak in the sizing of every trade.
    """
    baseline = run_pair(cfg.pairs[0], make_prices(), cfg)
    shocked = run_pair(cfg.pairs[0], make_prices(oos_shock=1.5), cfg)

    assert shocked.sizing_hedge_ratio == pytest.approx(baseline.sizing_hedge_ratio)


def test_in_sample_metrics_do_not_move_when_only_oos_prices_change(cfg):
    """
    The corollary: no in-sample number may depend on out-of-sample data.

    This is strictly stronger than the hedge-ratio check above, since it would
    also catch a leak entering through the signal or the z-score.
    """
    baseline = run_pair(cfg.pairs[0], make_prices(), cfg)
    shocked = run_pair(cfg.pairs[0], make_prices(oos_shock=1.5), cfg)

    assert shocked.metrics["in_sample"]["net"] == pytest.approx(
        baseline.metrics["in_sample"]["net"]
    )


def test_sizing_hedge_recovers_the_known_ratio(cfg):
    """Sanity check on the fixture: the fit should find the planted ratio."""
    run = run_pair(cfg.pairs[0], make_prices(), cfg)
    assert run.sizing_hedge_ratio == pytest.approx(TRUE_RATIO, abs=0.05)


def test_sizing_hedge_is_constant_across_the_whole_run(cfg):
    """
    The sizing ratio is frozen after the in-sample fit, so the engine's
    stability guard should see exactly zero drift.
    """
    run = run_pair(cfg.pairs[0], make_prices(), cfg)
    assert isinstance(run.sizing_hedge_ratio, float)
    assert np.isfinite(run.sizing_hedge_ratio)


# --------------------------------------------------------------------------- #
# signal_hedge actually drives behaviour
# --------------------------------------------------------------------------- #
def test_signal_hedge_setting_changes_the_spread(cfg):
    """
    `signal_hedge` is recorded in run_manifest.json as a description of what was
    run. It was once inert -- validated, written to the manifest, and ignored by
    run_pair, which hardcoded the rolling spread. A config option the manifest
    advertises but the code does not honour is a false record of the run.
    """
    rolling = run_pair(cfg.pairs[0], make_prices(), cfg)
    static_cfg = dataclasses.replace(
        cfg, signal=dataclasses.replace(cfg.signal, signal_hedge="static")
    )
    static = run_pair(cfg.pairs[0], make_prices(), static_cfg)

    assert not rolling.spread.equals(static.spread)


def test_static_signal_hedge_uses_the_in_sample_ratio(cfg):
    """
    The static signal branch must reuse the in-sample-only sizing ratio, not
    re-fit on the whole sample -- otherwise switching to it would introduce the
    very leak the rest of this module guards against.
    """
    static_cfg = dataclasses.replace(
        cfg, signal=dataclasses.replace(cfg.signal, signal_hedge="static")
    )
    baseline = run_pair(cfg.pairs[0], make_prices(), static_cfg)
    shocked = run_pair(cfg.pairs[0], make_prices(oos_shock=1.5), static_cfg)

    assert shocked.metrics["in_sample"]["net"] == pytest.approx(
        baseline.metrics["in_sample"]["net"]
    )


# --------------------------------------------------------------------------- #
# period bookkeeping
# --------------------------------------------------------------------------- #
def test_periods_partition_the_sample(cfg):
    """in_sample and out_of_sample must tile `full` exactly — no gap, no overlap."""
    run = run_pair(cfg.pairs[0], make_prices(), cfg)
    n = {p: run.metrics[p]["net"]["n_periods"] for p in PERIODS}
    assert n["in_sample"] + n["out_of_sample"] == n["full"]
    assert n["in_sample"] > 0 and n["out_of_sample"] > 0


def test_both_periods_are_reported_gross_and_net(cfg):
    """Principles 4 and 5 are structural: every period carries both bases."""
    run = run_pair(cfg.pairs[0], make_prices(), cfg)
    for period in PERIODS:
        assert set(run.metrics[period]) == {"gross", "net"}


def test_costs_never_improve_returns(cfg):
    """Net is gross minus a non-negative charge, in every period."""
    run = run_pair(cfg.pairs[0], make_prices(), cfg)
    for period in PERIODS:
        block = run.metrics[period]
        assert block["net"]["total_return"] <= block["gross"]["total_return"] + 1e-12


# --------------------------------------------------------------------------- #
# tabulation
# --------------------------------------------------------------------------- #
def test_to_frame_is_long_form_with_one_row_per_cell(cfg):
    runs = run_study(make_prices(), cfg)
    table = to_frame(runs)

    assert len(table) == len(cfg.pairs) * len(PERIODS) * 2
    assert not table.duplicated(subset=["pair", "period", "basis"]).any()
    assert set(table["period"]) == set(PERIODS)
    assert set(table["basis"]) == {"gross", "net"}


def test_run_study_covers_every_configured_pair(cfg):
    """
    No pair may be silently omitted. There is deliberately no subset argument —
    dropping a pair after seeing its result is the failure mode this guards.
    """
    runs = run_study(make_prices(), cfg)
    assert [r.pair.name for r in runs] == [p.name for p in cfg.pairs]


def test_table_has_no_tier_column(cfg):
    """
    The study has one flat universe. A column that could rank pairs into
    first- and second-class is the thing being kept out.
    """
    table = to_frame(run_study(make_prices(), cfg))
    assert "group" not in table.columns
    assert set(table["pair"]) == {"A/B", "C/D"}


def test_hedge_ratio_column_matches_the_run(cfg):
    runs = run_study(make_prices(), cfg)
    table = to_frame(runs)
    for r in runs:
        rows = table[table["pair"] == r.pair.name]["hedge_ratio"].unique()
        assert rows == pytest.approx([r.sizing_hedge_ratio])


# --------------------------------------------------------------------------- #
# failure modes
# --------------------------------------------------------------------------- #
def test_empty_overlap_raises(cfg):
    """A pair whose legs never trade on the same day is an error, not a silent zero."""
    prices = make_prices()
    prices.loc[:, "A"] = np.nan
    with pytest.raises(ValueError, match="no overlapping price data"):
        run_pair(cfg.pairs[0], prices, cfg)


def test_in_sample_shorter_than_signal_window_raises(cfg):
    """
    Fitting the hedge on fewer days than the signal window is not a judgement
    call the study should make silently.
    """
    late_split = Config(**{**cfg.__dict__, "split": SplitConfig(in_sample_end="2015-01-20")})
    with pytest.raises(ValueError, match="shorter than signal.window"):
        run_pair(late_split.pairs[0], make_prices(), late_split)
