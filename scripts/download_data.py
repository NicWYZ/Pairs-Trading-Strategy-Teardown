"""
Fetch and cache adjusted-close prices for all pairs in the study.

Reads tickers and the date range from configs/pairs.yaml so the config is the
single source of truth: adding a pair there is all it takes for its data to be
downloaded.
"""

from __future__ import annotations

import argparse

import pandas as pd

from pairs_teardown.config import load_config
from pairs_teardown.data.clean import align_prices, handle_missing
from pairs_teardown.data.loaders import load_or_download


def load_pair(a: str, b: str, prices_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and clean one pair from the full price panel.
    """
    pair_raw = prices_raw[[a, b]]
    return align_prices(handle_missing(pair_raw))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/pairs.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)

    # The whole ticker list is always downloaded together, which keeps the
    # loader's cache key (sorted tickers + date range) stable across runs.
    prices_raw = load_or_download(
        list(cfg.tickers), cfg.data.start, cfg.data.end, cfg.data.cache_dir
    )

    print("\n--- Per-pair summary after cleaning ---")
    for pair in cfg.pairs:
        px = load_pair(pair.a, pair.b, prices_raw)
        print(f"\n{pair.name}")
        print(f"  Trading days : {len(px)}")
        print(f"  Date range   : {px.index[0].date()} -> {px.index[-1].date()}")
        print(f"  Missing      : {px.isna().sum().to_dict()}")


if __name__ == "__main__":
    main()
