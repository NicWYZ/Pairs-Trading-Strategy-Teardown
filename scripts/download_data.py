"""
Fetch and cache adjusted-close prices for all pairs in the study.
"""

from pathlib import Path
import pandas as pd
from pairs_teardown.data.loaders import load_or_download
from pairs_teardown.data.clean import handle_missing, align_prices

PAIRS = [("WM", "RSG"), ("FOXA", "FOX"), ("SPY", "VOO"), ("KO", "PEP"),("MA", "V"), ("XOM", "CVX")]
ALL_TICKERS = [t for pair in PAIRS for t in pair]
START = "2015-01-01"
END = "2024-12-31"
CACHE = Path("data/raw")


def load_pair(a: str, b: str, prices_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and clean one pair from the full price panel.
    """
    pair_raw = prices_raw[[a, b]]
    return align_prices(handle_missing(pair_raw))


def main() -> None:
    prices_raw = load_or_download(ALL_TICKERS, START, END, CACHE)

    print("\n--- Per-pair summary after cleaning ---")
    for a, b in PAIRS:
        pair = load_pair(a, b, prices_raw)
        print(f"\n{a}/{b}")
        print(f"  Trading days : {len(pair)}")
        print(f"  Date range   : {pair.index[0].date()} → {pair.index[-1].date()}")
        print(f"  Missing      : {pair.isna().sum().to_dict()}")


if __name__ == "__main__":
    main()