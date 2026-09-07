"""
Tests for config loading and validation.
"""

import copy

import pandas as pd
import pytest
import yaml

from pairs_teardown.config import load_config

BASE = {
    "data": {
        "start": "2015-01-01",
        "end": "2024-12-31",
        "price_field": "adj_close",
        "cache_dir": "data/raw",
    },
    "split": {"in_sample_end": "2021-12-31"},
    "signal": {
        "window": 60,
        "entry": 2.0,
        "exit": 0.5,
        "signal_hedge": "rolling",
        "sizing_hedge": "static",
        "sizing_hedge_max_std": 0.05,
    },
    "costs": {"commission_bps": 1.0, "slippage_bps": 5.0},
    "backtest": {"periods_per_year": 252},
    "output": {"results_dir": "reports/results", "figures_dir": "reports/figures"},
    "pairs": {
        "official": [
            {"name": "WM/RSG", "a": "WM", "b": "RSG", "rationale": "duopoly"},
            {"name": "FOXA/FOX", "a": "FOXA", "b": "FOX", "rationale": "share classes"},
        ],
        "sanity_check": [
            {"name": "KO/PEP", "a": "KO", "b": "PEP", "rationale": "beverages"},
        ],
    },
}


def write(tmp_path, mutate=None):
    """Write a config file, optionally applying a mutation to a deep copy."""
    cfg = copy.deepcopy(BASE)
    if mutate:
        mutate(cfg)
    p = tmp_path / "pairs.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


# --------------------------------------------------------------------------- #
# happy path
# --------------------------------------------------------------------------- #
def test_loads_values(tmp_path):
    cfg = load_config(write(tmp_path))
    assert cfg.signal.window == 60
    assert cfg.signal.entry == 2.0
    assert cfg.costs.slippage_bps == 5.0
    assert cfg.backtest.periods_per_year == 252


def test_pair_groups_separated(tmp_path):
    cfg = load_config(write(tmp_path))
    assert [p.name for p in cfg.official_pairs] == ["WM/RSG", "FOXA/FOX"]
    assert [p.name for p in cfg.sanity_pairs] == ["KO/PEP"]
    assert len(cfg.pairs) == 3


def test_tickers_deduplicated_and_ordered(tmp_path):
    cfg = load_config(write(tmp_path))
    assert cfg.tickers == ("WM", "RSG", "FOXA", "FOX", "KO", "PEP")


def test_config_is_frozen(tmp_path):
    """A run must not be able to mutate its own parameters mid-flight."""
    cfg = load_config(write(tmp_path))
    with pytest.raises(Exception):
        cfg.signal.window = 120  # type: ignore[misc]


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


# --------------------------------------------------------------------------- #
# IS/OOS split masks
# --------------------------------------------------------------------------- #
def test_split_masks_partition_the_index(tmp_path):
    cfg = load_config(write(tmp_path))
    idx = pd.date_range("2021-12-29", "2022-01-04", freq="D")
    is_mask = cfg.split.is_mask(idx)
    oos_mask = cfg.split.oos_mask(idx)

    # in-sample is inclusive of the split date; out-of-sample is strictly after
    assert is_mask.sum() == 3  # 12-29, 12-30, 12-31
    assert oos_mask.sum() == 4  # 01-01 .. 01-04
    # the two masks must be exact complements: no row in both, none in neither
    assert not (is_mask & oos_mask).any()
    assert (is_mask | oos_mask).all()


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def test_entry_below_exit_rejected(tmp_path):
    def m(c):
        c["signal"]["entry"] = 0.3
        c["signal"]["exit"] = 0.5

    with pytest.raises(ValueError, match="must exceed"):
        load_config(write(tmp_path, m))


def test_rolling_sizing_hedge_rejected(tmp_path):
    """The single most costly misconfiguration possible in this project."""

    def m(c):
        c["signal"]["sizing_hedge"] = "rolling"

    with pytest.raises(ValueError, match="not permitted"):
        load_config(write(tmp_path, m))


def test_unknown_hedge_spec_rejected(tmp_path):
    def m(c):
        c["signal"]["signal_hedge"] = "rollling"  # typo

    with pytest.raises(ValueError, match="rolling"):
        load_config(write(tmp_path, m))


def test_split_outside_data_range_rejected(tmp_path):
    def m(c):
        c["split"]["in_sample_end"] = "2030-01-01"

    with pytest.raises(ValueError, match="strictly between"):
        load_config(write(tmp_path, m))


def test_window_too_small_rejected(tmp_path):
    def m(c):
        c["signal"]["window"] = 1

    with pytest.raises(ValueError, match="window"):
        load_config(write(tmp_path, m))


def test_negative_costs_rejected(tmp_path):
    def m(c):
        c["costs"]["commission_bps"] = -1.0

    with pytest.raises(ValueError, match="non-negative"):
        load_config(write(tmp_path, m))


def test_duplicate_pair_names_rejected(tmp_path):
    def m(c):
        c["pairs"]["official"][1]["name"] = "WM/RSG"

    with pytest.raises(ValueError, match="duplicate"):
        load_config(write(tmp_path, m))


def test_identical_legs_rejected(tmp_path):
    def m(c):
        c["pairs"]["official"][0]["b"] = "WM"

    with pytest.raises(ValueError, match="identical legs"):
        load_config(write(tmp_path, m))


def test_missing_required_pair_key_rejected(tmp_path):
    def m(c):
        del c["pairs"]["official"][0]["b"]

    with pytest.raises(ValueError, match="missing required key"):
        load_config(write(tmp_path, m))