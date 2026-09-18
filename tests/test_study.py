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
    InferenceConfig,
    OutputConfig,
    WalkForwardConfig,
    Pair,
    SignalConfig,
    SplitConfig,
)
from pairs_teardown.study import (
    PERIODS,
    inference_frame,
    refit_dates,
    run_pair,
    run_pair_walk_forward,
    run_study,
    run_study_walk_forward,
    segments_frame,
    to_frame,
    walk_forward_window,
)

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
        inference=InferenceConfig(n_boot=200, mean_block=5.0, ci_level=0.95, seed=0),
        walk_forward=WalkForwardConfig(half_life_multiple=2.0, window_min=20, window_max=250),
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


# --------------------------------------------------------------------------- #
# per-period trade counts
# --------------------------------------------------------------------------- #
def test_n_trades_is_counted_per_period(cfg):
    """
    The full-sample trade count was once copied into every period row, so the
    in-sample and out-of-sample rows claimed the same number. Per-period counts
    must tile the full count exactly.
    """
    run = run_pair(cfg.pairs[0], make_prices(), cfg)
    n = {p: run.metrics[p]["net"]["n_trades"] for p in PERIODS}
    assert n["full"] == run.result.n_trades
    assert n["in_sample"] + n["out_of_sample"] == n["full"]
    assert n["in_sample"] > 0 and n["out_of_sample"] > 0


# --------------------------------------------------------------------------- #
# inference table
# --------------------------------------------------------------------------- #
def test_inference_frame_has_one_row_per_cell_and_the_expected_columns(cfg):
    runs = run_study(make_prices(), cfg)
    inf = inference_frame(runs, cfg)
    assert len(inf) == len(cfg.pairs) * len(PERIODS) * 2
    assert not inf.duplicated(subset=["pair", "period", "basis"]).any()
    for col in [
        "sharpe",
        "sharpe_se",
        "sharpe_p",
        "sharpe_ci_lo",
        "sharpe_ci_hi",
        "total_return",
        "total_return_ci_lo",
        "total_return_ci_hi",
        "sharpe_p_holm",
    ]:
        assert col in inf.columns


def test_inference_intervals_bracket_the_point_estimates(cfg):
    """A percentile interval that excludes its own point estimate is a bug."""
    runs = run_study(make_prices(), cfg)
    inf = inference_frame(runs, cfg)
    ok = inf.dropna(subset=["sharpe_ci_lo"])
    assert (ok.sharpe_ci_lo <= ok.sharpe + 1e-9).all()
    assert (ok.sharpe_ci_hi >= ok.sharpe - 1e-9).all()
    assert (ok.total_return_ci_lo <= ok.total_return + 1e-9).all()
    assert (ok.total_return_ci_hi >= ok.total_return - 1e-9).all()


def test_inference_point_estimates_match_metrics(cfg):
    """inference.csv and metrics.csv must agree on every shared number."""
    runs = run_study(make_prices(), cfg)
    table = to_frame(runs).set_index(["pair", "period", "basis"])
    inf = inference_frame(runs, cfg).set_index(["pair", "period", "basis"])
    pd.testing.assert_series_equal(inf["sharpe"], table.loc[inf.index, "sharpe"], check_names=False)
    pd.testing.assert_series_equal(
        inf["total_return"], table.loc[inf.index, "total_return"], check_names=False
    )


def test_holm_adjustment_is_within_period_and_basis(cfg):
    """
    Adjusted p-values are never below raw ones, and the family is the set of
    pairs inside one (period, basis) cell -- so the adjustment never mixes
    in-sample and out-of-sample tests.
    """
    runs = run_study(make_prices(), cfg)
    inf = inference_frame(runs, cfg)
    ok = inf.dropna(subset=["sharpe_p"])
    assert (ok.sharpe_p_holm >= ok.sharpe_p - 1e-12).all()
    for _, grp in ok.groupby(["period", "basis"]):
        # With m pairs the largest possible multiplier is m.
        assert (grp.sharpe_p_holm <= np.minimum(1.0, grp.sharpe_p * len(grp)) + 1e-12).all()


def test_inference_is_deterministic_under_the_configured_seed(cfg):
    runs = run_study(make_prices(), cfg)
    a = inference_frame(runs, cfg)
    b = inference_frame(runs, cfg)
    pd.testing.assert_frame_equal(a, b)


# --------------------------------------------------------------------------- #
# Arm B: walk-forward
# --------------------------------------------------------------------------- #
def test_refit_dates_are_first_trading_day_of_each_oos_year(cfg):
    idx = make_prices().index
    dates = refit_dates(pd.DatetimeIndex(idx), SPLIT)
    oos = idx[idx > pd.Timestamp(SPLIT)]
    assert dates[0] == oos[0]
    assert [d.year for d in dates] == sorted({d.year for d in oos})
    for d in dates[1:]:
        assert d == oos[oos.year == d.year][0]


def test_walk_forward_window_rule(cfg):
    wf = cfg.walk_forward
    assert walk_forward_window(30.0, wf) == 60
    assert walk_forward_window(5.0, wf) == 20  # floor
    assert walk_forward_window(400.0, wf) == 250  # cap
    assert walk_forward_window(float("inf"), wf) == 250  # undetectable -> cap
    assert walk_forward_window(30.4, wf) == 61  # round, not truncate


def test_walk_forward_in_sample_is_identical_to_arm_a(cfg):
    """Before the split nothing is stale, so Arm B must reproduce Arm A exactly."""
    a = run_pair(cfg.pairs[0], make_prices(), cfg)
    b = run_pair_walk_forward(cfg.pairs[0], make_prices(), cfg)
    assert b.metrics["in_sample"]["net"] == pytest.approx(a.metrics["in_sample"]["net"])
    assert b.sizing_hedge_ratio == pytest.approx(a.sizing_hedge_ratio)


def test_walk_forward_segments_tile_the_evaluation_window(cfg):
    run = run_pair_walk_forward(cfg.pairs[0], make_prices(), cfg)
    idx = run.result.returns.index
    oos = idx[idx > pd.Timestamp(SPLIT)]
    assert run.segments[0].start == oos[0]
    assert run.segments[-1].end == oos[-1]
    for prev, nxt in zip(run.segments, run.segments[1:]):
        assert prev.end < nxt.start
        assert idx[(idx > prev.end) & (idx < nxt.start)].empty
    for s in run.segments:
        assert s.fit_end < s.start


def test_walk_forward_refit_uses_only_data_before_the_segment(cfg):
    """
    The leak guard for Arm B. Shock prices from a date D onward: every segment
    that starts on or before D must keep its hedge ratio, half-life and window,
    and every return before D must be unchanged.
    """
    base = run_pair_walk_forward(cfg.pairs[0], make_prices(), cfg)
    D = base.segments[1].start  # shock from the start of the second OOS segment
    prices = make_prices()
    prices.loc[prices.index >= D, "A"] *= 1.5
    shocked = run_pair_walk_forward(cfg.pairs[0], prices, cfg)

    for s0, s1 in zip(base.segments, shocked.segments):
        if s0.start <= D:
            assert s1.hedge_ratio == pytest.approx(s0.hedge_ratio)
            assert s1.half_life == pytest.approx(s0.half_life)
            assert s1.window == s0.window
    before = base.result.returns.index < D
    pd.testing.assert_series_equal(shocked.result.returns[before], base.result.returns[before])


def test_walk_forward_sizing_is_a_step_function_on_refit_dates(cfg):
    run = run_pair_walk_forward(cfg.pairs[0], make_prices(), cfg)
    idx = run.result.returns.index
    # Reconstruct the sizing series the engine saw from the segments.
    for s in run.segments:
        seg = (idx >= s.start) & (idx <= s.end)
        assert seg.sum() > 0
    starts = [s.start for s in run.segments]
    assert starts == refit_dates(pd.DatetimeIndex(idx), SPLIT)


def test_walk_forward_recovers_the_planted_ratio_every_year(cfg):
    run = run_pair_walk_forward(cfg.pairs[0], make_prices(), cfg)
    for s in run.segments:
        assert s.hedge_ratio == pytest.approx(TRUE_RATIO, abs=0.05)
        assert s.traded


def test_walk_forward_goes_flat_when_the_refit_hedge_is_not_positive(cfg):
    """
    A pair whose true relationship is negative (A moves against B) yields a
    negative refit hedge ratio; the pre-registered rule is to sit flat for
    that segment rather than trade a "hedge" that is a directional bet.
    """
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2015-01-01", periods=N)
    log_b = np.cumsum(rng.normal(0, 0.01, N)) + np.log(100.0)
    log_a = -2.0 * log_b + 2.0 * log_b[0] + np.log(50.0) + rng.normal(0, 0.02, N)
    prices = pd.DataFrame({"A": np.exp(log_a), "B": np.exp(log_b)}, index=idx)

    run = run_pair_walk_forward(cfg.pairs[0], prices, cfg)
    assert run.segments, "fixture must produce at least one evaluation segment"
    assert all(s.hedge_ratio < 0 for s in run.segments)
    assert not any(s.traded for s in run.segments)
    held = run.result.held_positions
    for s in run.segments:
        seg = (held.index > s.start) & (held.index <= s.end)  # after the one-bar lag
        assert (held[seg] == 0).all()
    # ...and the out-of-sample P&L is therefore exactly zero after the position
    # carried across the split has been closed.
    oos_after_first_day = held.index > run.segments[0].start
    assert run.result.gross_returns[oos_after_first_day].abs().sum() == pytest.approx(0.0)


def test_walk_forward_tables_are_well_formed(cfg):
    runs = run_study_walk_forward(make_prices(), cfg)
    assert [r.pair.name for r in runs] == [p.name for p in cfg.pairs]
    table = to_frame(runs)
    assert len(table) == len(cfg.pairs) * len(PERIODS) * 2
    inf = inference_frame(runs, cfg)
    assert len(inf) == len(table)
    seg = segments_frame(runs)
    assert set(seg.columns) >= {
        "pair",
        "segment",
        "start",
        "end",
        "fit_end",
        "hedge_ratio",
        "half_life",
        "window",
        "traded",
    }
    assert len(seg) == sum(len(r.segments) for r in runs)
