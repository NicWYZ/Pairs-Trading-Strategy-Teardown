"""
Print the loader's parquet cache path for the current config.

The Makefile needs this to express the pipeline's dependency graph. The cache
filename is derived from sorted(tickers) + the date range inside
``load_or_download``, so it changes whenever the pair list or the dates change.
Computing it here keeps that logic in one place instead of hardcoding a filename
that would silently go stale.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pairs_teardown.config import load_config


def cache_path(config_path: str) -> Path:
    cfg = load_config(config_path)
    key = "_".join(sorted(cfg.tickers)) + f"_{cfg.data.start}_{cfg.data.end}".replace("-", "")
    return Path(cfg.data.cache_dir) / f"{key}.parquet"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/pairs.yaml")
    args = ap.parse_args()
    print(cache_path(args.config))


if __name__ == "__main__":
    main()
