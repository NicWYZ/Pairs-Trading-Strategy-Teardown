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
| Standard error on any one pair's Sharpe | **0.58** (three years of daily data) |
| Positive pairs surviving a Holm correction | **0 of 10** |
| Best observed Sharpe vs. expected best-of-10 under no edge | **1.20 vs. 0.91** |

The answer is not "pairs trading works" or "pairs trading fails". It is that **the
dispersion across similar pairs dwarfs the average effect**, so any small-universe study
reports whichever conclusion its pair selection produces. Five independent lines converge
on that:

- **The premise** — only **1 of 10** pairs is cointegrated in *both* the in-sample and
  out-of-sample windows. The statistical property the method assumes does not persist.
- **Pair selection** — the first three pairs chosen went 0 for 3; the last four went 3 for 4.
- **Parameter choice** — 9 of 10 pairs change sign across an ordinary grid of lookback
  windows, and the pre-registered window is the *only* one of six with a positive
  cross-sectional mean.
- **The cross-section** — a mean of +0.6% inside a 26pp standard deviation.
- **Selection** — no positive pair's Sharpe survives a Holm correction across the ten, and
  the best one (UNP/CSX, 1.20) sits half a standard error above the expected maximum of ten
  strategies with no edge at all. The only Holm-significant result is SPY/VOO, negatively:
  paying 6 bps to trade a spread with no variance loses money with near certainty.

A useful corrective on costs: the per-pair breakeven cost is **bimodal**. Four pairs lose
money at *zero* cost and four clear the 6 bps charge by 4–16x, so for 8 of 10 pairs the cost
assumption barely affects the verdict. Most losing pairs have no gross edge, rather than an
edge eaten by friction.

A second corrective on the premise: cointegration counts depend on the test (Engle–Granger
and Johansen agree on one pair in-sample), and the estimated half-lives of mean reversion
exceed the 60-day signal window for five of ten pairs — and for the slowest two, the
reversion coefficient is within two standard errors of zero, so they cannot be told apart
from random walks.

At an earlier six-pair stage this project reported a confident negative result. Four more
pairs, chosen the same way and run through identical code, moved the mean from clearly
negative to indistinguishable from zero. Nothing was wrong with the machinery — the
conclusion was simply never as stable as the tables made it look.

### The holdout (pre-registered, 2025-01 → 2026-08)

Study 1 above was scored on 2022–2024. After it was written, a replication and one
methodological variant were pre-registered ([`PREREGISTRATION.md`](PREREGISTRATION.md),
committed before any later price was downloaded) and scored once on twenty unseen months:

| | |
|---|---|
| Frozen strategy, replicated | **2 of 10** profitable, mean **−3.5%** (p = 0.36), no pair significant |
| Best holdout Sharpe vs. expected best-of-10 under no edge | **1.24 vs. 1.23** — exactly the noise floor |
| 2022–24 winners that won again | **0 of 4**; Spearman ρ between periods **−0.56** (p = 0.09) |
| Walk-forward variant (annual refit, half-life-set window) | **−7.8 pp** vs. frozen on 2022–24; **+2.9 pp** [−10, +14] on the holdout; individual pairs move by up to 46 pp |

Study 1's numbers did not replicate; its thesis did. The pair a reader would have kept
(UNP/CSX, +55.9%) lost 11.5%; the pair they would have dropped (UPS/FDX, −42.4%) made
24.4%. And the textbook fix for the diagnosed problem — re-estimate what goes stale, size
the window from the half-life — produced no detectable improvement, only a different draw.
Details: [`notebooks/06_holdout.ipynb`](notebooks/06_holdout.ipynb).

Full argument: [`notebooks/05_writeup.ipynb`](notebooks/05_writeup.ipynb).
Tables and figures: [`notebooks/03_backtest_results.ipynb`](notebooks/03_backtest_results.ipynb).
Robustness checks: [`notebooks/04_sensitivity_analysis.ipynb`](notebooks/04_sensitivity_analysis.ipynb).

## Run it

```bash
git clone https://github.com/NicWYZ/Pairs-Trading-Strategy-Teardown.git
cd Pairs-Trading-Strategy-Teardown
make install     # uv sync --extra dev
make run         # downloads prices if stale, then the full study
```

`make run` writes `reports/results/metrics.csv`, `inference.csv` (standard errors,
bootstrap intervals and adjusted p-values for every cell of it), a `run_manifest.json`
recording exactly what was run, and three figures per pair. It is incremental — the backtest re-runs only if
the config, the package source, or the cached prices changed.

```
make run-holdout # the pre-registered holdout: download through 2026-08, run both arms once
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
  stats/        cointegration: OLS hedge ratio, ADF, Engle–Granger, Johansen, OU half-life
                inference: Lo (2002) Sharpe SE, stationary bootstrap, Holm, expected max Sharpe
  signals/      rolling z-score, entry/exit rules with hysteresis
  backtest/     the engine (one-bar position lag) and the cost model
  metrics/      Sharpe (with SE), drawdown, turnover, gross/net summary
  plotting/     spread, equity, drawdown figures
  study.py      the chain assembled: run a pair end-to-end (frozen split, and walk-forward), tabulate
  config.py     load + validate configs/pairs.yaml

notebooks/
  01 data  ->  02 cointegration  ->  03 results  ->  04 sensitivity  ->  05 writeup  ->  06 holdout
PREREGISTRATION.md   the holdout plan, committed before the holdout data existed
```

All logic lives in tested, importable modules. Notebooks import and call; they contain no
strategy code. 137 tests, all on synthetic data with known answers — the suite never
touches the network.

## The statistics

The point estimates are the easy part; the project's statistical content is in what is
attached to them. Each tool has a closed-form or Monte Carlo test in `tests/`.

| Question | Tool | Where |
|---|---|---|
| Is the spread cointegrated? | Engle–Granger (residual-based, with the MacKinnon correction for an estimated hedge ratio) **and** Johansen trace (system-based, symmetric in the legs) | `stats/cointegration.py` |
| How fast does it revert? | Discrete Ornstein–Uhlenbeck fit; half-life = −ln 2 / ln(1+φ) with a delta-method SE | `stats/cointegration.py` |
| How precise is one pair's Sharpe? | Lo (2002) analytic SE; Politis–Romano stationary bootstrap for serially dependent daily P&L | `stats/inference.py` |
| Which of ten results survive? | Holm step-down adjustment (no independence assumption) | `stats/inference.py` |
| Is the best pair better than the best of ten coin flips? | Expected maximum of N null Sharpes (Bailey & López de Prado 2014) | `stats/inference.py` |

The bootstrap scheme (draws, block length, seed) is fixed in `configs/pairs.yaml` before any
interval is seen, and the intervals are written by the pipeline to `inference.csv` rather
than computed in a notebook — the same one-source discipline as the returns.

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
   separable through the API. Every Sharpe in that block carries its standard error.
5. **In-sample and out-of-sample always separated.** OOS scored once, reported as-is. The
   holdout extends this: the plan was committed before the data existed, and the walk-forward
   arm's leak guard (`test_study.py`) checks that no refit sees data from its own segment.

A note on rule 3: this project originally split its pairs into an "official" headline set
and a "sanity check" set. That structure was a mistake — it would have let the same data
support opposite conclusions depending on which tier a pair landed in — and removing it is
what made the real result visible. The writeup discusses this at length rather than quietly
fixing it.

## Caveats

Ten pairs is a small cross-section, and three years is a short out-of-sample window: the
standard error of a Sharpe ratio falls as 1/√T, so halving 0.58 would need twelve years per
pair. The study's own conclusion is partly a statement about how small both are. Pairs were chosen in waves, with later waves selected while earlier results were
known. Costs are a flat per-side assumption with no market-impact model, no borrow costs and
no short-availability constraints. All pairs are US large-cap equities over one decade.

Details in [`notebooks/05_writeup.ipynb`](notebooks/05_writeup.ipynb) §4; the full build
guide is [`PROJECT_PLAN.md`](PROJECT_PLAN.md).
