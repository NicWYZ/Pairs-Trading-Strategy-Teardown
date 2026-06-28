from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf  # type: ignore[import-untyped]

def download_prices(
        tickers: list[str],
        start:str,
        end: str
) -> pd.DataFrame:
    
    """Download daily adjusted-close prices for a list of tickers.

    Parameters
    ----------
    tickers : list of ticker symbols, e.g. ["WM", "RSG"]
    start   : start date string, e.g. "2015-01-01"
    end     : end date string,   e.g. "2024-12-31"

    Returns
    -------
    DataFrame with DatetimeIndex and one column per ticker (adjusted close).
    """
    raw: pd.DataFrame = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False, threads=False) # type: ignore

    # yfinance returns MultiIndex columns when multiple tickers are passed;
    # 'Close' under auto_adjust=True is the adjusted close price.
    prices: pd.DataFrame
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()  # type: ignore[assignment]
    else:
        # Single ticker falls back to flat columns
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices.index.name = "Date"
    # Reorder columns to match the input list order
    cols = [t for t in tickers if t in prices.columns]
    return prices[cols]  # type: ignore[return-value]

def load_or_download(
    tickers: list[str],
    start: str,
    end: str,
    cache_dir: Path | str = Path("data/raw"),
) -> pd.DataFrame:
    """Return cached prices if available; otherwise download and cache as parquet.

    Parameters
    ----------
    tickers   : list of ticker symbols
    start     : start date string
    end       : end date string
    cache_dir : directory for parquet cache files

    Returns
    -------
    DataFrame with DatetimeIndex and one column per ticker.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Stable filename: sorted tickers + date range (no hyphens)
    key = "_".join(sorted(tickers)) + f"_{start}_{end}".replace("-", "")
    cache_file = cache_dir / f"{key}.parquet"

    if cache_file.exists():
        print(f"Loading from cache: {cache_file.name}")
        return pd.read_parquet(cache_file)

    print(f"Downloading {tickers} from Yahoo Finance...")
    prices = download_prices(tickers, start, end)
    prices.to_parquet(cache_file)
    print(f"Cached to: {cache_file}")
    return prices