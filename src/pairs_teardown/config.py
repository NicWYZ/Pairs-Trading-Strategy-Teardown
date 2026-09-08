"""
Typed loading and validation of the study configuration.

The dataclasses are ``frozen=True`` so nothing downstream can mutate a parameter
mid-run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

__all__ = [
    "Config",
    "DataConfig",
    "SplitConfig",
    "SignalConfig",
    "CostConfig",
    "BacktestConfig",
    "OutputConfig",
    "Pair",
    "load_config",
]


@dataclass(frozen=True)
class DataConfig:
    start: str
    end: str
    price_field: str
    cache_dir: str


@dataclass(frozen=True)
class SplitConfig:
    in_sample_end: str

    def is_mask(self, index: pd.Index) -> pd.Series:
        """Boolean mask selecting the in-sample rows of a DatetimeIndex."""
        return pd.Series(index <= pd.Timestamp(self.in_sample_end), index=index)

    def oos_mask(self, index: pd.Index) -> pd.Series:
        """Boolean mask selecting the out-of-sample rows of a DatetimeIndex."""
        return pd.Series(index > pd.Timestamp(self.in_sample_end), index=index)


@dataclass(frozen=True)
class SignalConfig:
    window: int
    entry: float
    exit: float
    signal_hedge: str
    sizing_hedge: str
    sizing_hedge_max_std: float


@dataclass(frozen=True)
class CostConfig:
    commission_bps: float
    slippage_bps: float


@dataclass(frozen=True)
class BacktestConfig:
    periods_per_year: int


@dataclass(frozen=True)
class OutputConfig:
    results_dir: str
    figures_dir: str


@dataclass(frozen=True)
class Pair:
    """
    One pair to study.

    There is deliberately no tier or category field. Every pair in the config was
    specified from economic reasoning before it was run and is reported whatever
    it did; a schema that cannot express "second-class pair" is what stops one
    being invented after the fact.
    """

    name: str
    a: str
    b: str
    rationale: str


@dataclass(frozen=True)
class Config:
    data: DataConfig
    split: SplitConfig
    signal: SignalConfig
    costs: CostConfig
    backtest: BacktestConfig
    output: OutputConfig
    pairs: tuple[Pair, ...]

    @property
    def tickers(self) -> tuple[str, ...]:
        """Every distinct ticker referenced, in first-appearance order."""
        seen: list[str] = []
        for p in self.pairs:
            for t in (p.a, p.b):
                if t not in seen:
                    seen.append(t)
        return tuple(seen)


def _require(mapping: dict, key: str, where: str) -> object:
    """Fetch a required key or raise a message naming the exact location."""
    if key not in mapping:
        raise ValueError(f"config: missing required key '{key}' in section '{where}'")
    return mapping[key]


def _parse_pairs(raw: list | None) -> tuple[Pair, ...]:
    """Parse the flat `pairs:` list. A rationale is required, not optional."""
    if not isinstance(raw, list):
        raise ValueError("config: 'pairs' must be a list of pair entries")
    return tuple(
        Pair(
            name=str(_require(entry, "name", "pairs")),
            a=str(_require(entry, "a", "pairs")),
            b=str(_require(entry, "b", "pairs")),
            rationale=str(_require(entry, "rationale", "pairs")),
        )
        for entry in raw
    )


def _validate(cfg: Config) -> None:
    """Fail loudly on any configuration that would silently corrupt the study."""
    s = cfg.signal

    if s.window < 2:
        raise ValueError(f"config: signal.window must be >= 2, got {s.window}")
    if s.entry <= s.exit:
        raise ValueError(
            f"config: signal.entry ({s.entry}) must exceed signal.exit ({s.exit}); "
            "otherwise the entry and exit bands cross and the hysteresis rule is "
            "incoherent."
        )
    if s.exit < 0:
        raise ValueError(f"config: signal.exit must be >= 0, got {s.exit}")
    if s.signal_hedge not in {"rolling", "static"}:
        raise ValueError(
            f"config: signal.signal_hedge must be 'rolling' or 'static', got {s.signal_hedge!r}"
        )
    if s.sizing_hedge not in {"rolling", "static"}:
        raise ValueError(
            f"config: signal.sizing_hedge must be 'rolling' or 'static', got {s.sizing_hedge!r}"
        )
    if s.sizing_hedge == "rolling":
        # Not a style preference. A rolling sizing ratio that drifts toward zero
        # destroys market neutrality; the synthetic ground-truth test showed
        # +521% (true ratio) vs -61.6% (rolling estimate).
        raise ValueError(
            "config: signal.sizing_hedge='rolling' is not permitted. Sizing must "
            "use the static hedge ratio; only the SIGNAL may use a rolling one."
        )
    if s.sizing_hedge_max_std <= 0:
        raise ValueError("config: signal.sizing_hedge_max_std must be > 0")

    if cfg.costs.commission_bps < 0 or cfg.costs.slippage_bps < 0:
        raise ValueError("config: cost parameters must be non-negative")

    if cfg.backtest.periods_per_year < 1:
        raise ValueError("config: backtest.periods_per_year must be >= 1")

    start = pd.Timestamp(cfg.data.start)
    end = pd.Timestamp(cfg.data.end)
    split = pd.Timestamp(cfg.split.in_sample_end)
    if not start < end:
        raise ValueError(f"config: data.start ({start.date()}) must precede data.end")
    if not start < split < end:
        raise ValueError(
            f"config: split.in_sample_end ({split.date()}) must fall strictly "
            f"between data.start ({start.date()}) and data.end ({end.date()}); "
            "otherwise one of the two evaluation periods is empty."
        )

    if not cfg.pairs:
        raise ValueError("config: no pairs defined")
    names = [p.name for p in cfg.pairs]
    if len(names) != len(set(names)):
        raise ValueError(f"config: duplicate pair names: {names}")
    for p in cfg.pairs:
        if p.a == p.b:
            raise ValueError(f"config: pair {p.name!r} has identical legs ({p.a})")


def load_config(path: str | Path) -> Config:
    """
    Load, parse, and validate the study configuration.

    Raises FileNotFoundError if the path does not exist, and ValueError with a
    specific message for any invalid setting.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")

    with path.open("r") as fh:
        raw = yaml.safe_load(fh) or {}

    cfg = Config(
        data=DataConfig(**raw["data"]),
        split=SplitConfig(**raw["split"]),
        signal=SignalConfig(**raw["signal"]),
        costs=CostConfig(**raw["costs"]),
        backtest=BacktestConfig(**raw["backtest"]),
        output=OutputConfig(**raw["output"]),
        pairs=_parse_pairs(raw.get("pairs")),
    )
    _validate(cfg)
    return cfg
