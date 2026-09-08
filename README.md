# Pairs-Trading Strategy Teardown

[![CI](https://github.com/NicWYZ/Pairs-Trading-Strategy-Teardown/actions/workflows/ci.yml/badge.svg)](https://github.com/NicWYZ/Pairs-Trading-Strategy-Teardown/actions/workflows/ci.yml)

An honest evaluation of whether classic pairs-trading edges survive realistic transaction
costs and a strict out-of-sample test — built as production-grade research code rather than
a notebook monolith.

This is a **teardown, not a strategy pitch**. The goal was never to find something
profitable; it was to implement pairs trading carefully enough that whatever it says can be
believed.

## The finding

Ten economically-linked US large-cap pairs, identical pre-registered parameters, 2015–2024
with a 2021-12-31 in/out-of-sample split, 6 bps per side of cost. Out-of-sample, after
costs:

| | |
|---|---|
| Profitable | **4 of 10** pairs |
| Range | **−42.4%** (UPS/FDX) to **+55.9%** (UNP/CSX) |
| Cross-sectional mean | **+0.6%** — indistinguishable from zero (t = 0.08, p = 0.94) |
| Cross-sectional std dev | **26 percentage points** |
| Cost drag | consumes **~80%** of the mean gross return |

The answer is not "pairs trading works" or "pairs trading fails". It is that **the
dispersion across similar pairs dwarfs the average effect**, so any small-universe study
reports whichever conclusion its pair selection produces. Three independent lines converge
on that:

- **Pair selection** — the first three pairs chosen went 0 for 3; the last four went 3 for 4.
- **Parameter choice** — 9 of 10 pairs change sign across an ordinary grid of lookback
  windows, and the pre-registered window is the *only* one of six with a positive
  cross-sectional mean.
- **The cross-section** — a mean of +0.6% inside a 26pp standard deviation.

At an earlier six-pair stage this project reported a confident negative result. Four more
pairs, chosen the same way and run through identical code, moved the mean from clearly
negative to indistinguishable from zero. Nothing was wrong with the machinery — the
conclusion was simply never as stable as the tables made it look.

Full argument: [`notebooks/05_writeup.ipynb`](notebooks/05_writeup.ipynb).
Tables and figures: [`notebooks/04_backtest_results.ipynb`](notebooks/04_backtest_results.ipynb).

## Run it

```bash
git clone https://github.com/NicWYZ/Pairs-Trading-Strategy-Teardown.git
cd Pairs-Trading-Strategy-Teardown
make install     # uv sync --extra dev
make run         # downloads prices if stale, then the full study
```

`make run` writes `reports/results/metrics.csv`, a `run_manifest.json` recording exactly
what was run, and three figures per pair. It is incremental — the backtest re-runs only if
the config, the package source, or the cached prices changed.

```
make test        make lint        make typecheck
make notebooks   # execute every notebook to check it still runs
make clean       # wipe reports/ (keeps cached prices)
```

No market data is committed. `configs/pairs.yaml` plus the lockfile are enough to
reproduce every number.

## How it is built

```
src/pairs_teardown/
  data/         download + cache (parquet), align and clean
  stats/        ADF, Engle–Granger, OLS hedge ratio
  signals/      rolling z-score, entry/exit rules with hysteresis
  backtest/     the engine (one-bar position lag) and the cost model
  metrics/      Sharpe, drawdown, turnover, gross/net summary
  plotting/     spread, equity, drawdown figures
  study.py      the chain assembled: run a pair end-to-end, tabulate
  config.py     load + validate configs/pairs.yaml
```

All logic lives in tested, importable modules. Notebooks import and call; they contain no
strategy code. 83 tests, all on synthetic data with known answers — the suite never touches
the network.

## The five rules, and where they are enforced

The point of the project is that these are enforced by code and tests, not asserted in
prose:

1. **No look-ahead.** Positions lag one bar; the hedge ratio is fit in-sample only.
   Guarded by `test_engine.py` (changing a future price must not change past P&L) and
   `test_study.py` (shocking out-of-sample prices must not move the fitted ratio). Both were
   verified to go red when the bug is deliberately reintroduced.
2. **Survivorship bias** is acknowledged and its direction stated, not engineered around.
3. **No data-snooping.** Parameters fixed a priori. Every pair is reported whatever it did —
   `Pair` has no tier field, `to_frame` emits no column that could rank pairs, and
   `run_study` has no way to run a subset.
4. **Gross and net always together.** `summary()` returns both in one block; they are not
   separable through the API.
5. **In-sample and out-of-sample always separated.** OOS scored once, reported as-is.

A note on rule 3: this project originally split its pairs into an "official" headline set
and a "sanity check" set. That structure was a mistake — it would have let the same data
support opposite conclusions depending on which tier a pair landed in — and removing it is
what made the real result visible. The writeup discusses this at length rather than quietly
fixing it.

## Caveats

Ten pairs is a small cross-section; the study's own conclusion is partly a statement about
how small. Pairs were chosen in waves, with later waves selected while earlier results were
known. Costs are a flat per-side assumption with no market-impact model, no borrow costs and
no short-availability constraints. All pairs are US large-cap equities over one decade.

Details in [`notebooks/05_writeup.ipynb`](notebooks/05_writeup.ipynb) §4; the full build
guide is [`PROJECT_PLAN.md`](PROJECT_PLAN.md).
