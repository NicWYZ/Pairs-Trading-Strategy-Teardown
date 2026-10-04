# Pairs-Trading Strategy Teardown

[![CI](https://github.com/NicWYZ/Pairs-Trading-Strategy-Teardown/actions/workflows/ci.yml/badge.svg)](https://github.com/NicWYZ/Pairs-Trading-Strategy-Teardown/actions/workflows/ci.yml)

**Status: complete.** This repository is the final, frozen record of the study. The numbers
below are reproduced exactly by `make run` and `make run-holdout`.

## Abstract

We evaluate a textbook pairs-trading strategy on ten economically-linked US large-cap pairs.
The strategy uses a rolling-hedge spread, a 60-day z-score, and entry at |z| ≥ 2 with exit at
|z| ≤ 0.5. Every parameter is fixed in advance, transaction costs are 6 bps per side, and the
out-of-sample period (2022–2024) is scored once.

Out-of-sample, four of ten pairs are profitable after costs. The cross-sectional mean net
return is +0.6% (t = 0.08, p = 0.94) with a standard deviation of 26 percentage points. No
positive pair survives a Holm correction. The best Sharpe ratio (1.20) is half a standard
error above the expected maximum of ten strategies with no edge (0.91).

The result is not a verdict on pairs trading. It measures what a ten-pair study can resolve.
Pair-to-pair dispersion is so large relative to the mean that the headline is decided by
arbitrary choices: which pairs are chosen, the lookback window, the hedge estimator.

A pre-registered holdout (2025-01 → 2026-08) tests that reading directly. The 2022–2024
ranking does not persist: none of the four winners repeats, and Spearman ρ = −0.56. A
pre-registered methodological fix (annual re-estimation, with the window set from the
half-life) produces no detectable improvement.

## Results

### Main study: 2015–2021 in-sample, 2022–2024 out-of-sample

| Out-of-sample, net of 6 bps/side | |
|---|---|
| Profitable pairs | **4 of 10** |
| Range | **−42.4%** (UPS/FDX) to **+55.9%** (UNP/CSX) |
| Cross-sectional mean | **+0.6%** (t = 0.08, p = 0.94); median −1.9% |
| Cross-sectional standard deviation | **26.3 pp** |
| Cost drag on the mean | +3.1% gross → +0.6% net (**80%**) |
| Standard error of one pair's Sharpe (752 days) | **0.58** |
| Positive pairs significant after Holm | **0 of 10** (only SPY/VOO, negatively) |
| Best Sharpe vs expected best-of-10 with no edge | **1.20 vs 0.91** |

Five lines of evidence point the same way. They are not independent: each shows that, at
this sample size, the noise in any one pair's outcome is several times the average effect.

- **The premise does not persist.** 3 of 10 pairs are cointegrated (Engle–Granger, 5%)
  in-sample and 3 of 10 out-of-sample, but only MA/V is cointegrated in both periods.
  Johansen's test selects a different set and leaves MA/V and SPY/VOO. Half-lives of mean
  reversion exceed the 60-day window for 5 of 10 pairs. For XOM/CVX and UPS/FDX, no
  reversion is detectable at all.
- **Pair selection decides the headline.** The pairs were chosen in three waves. The first
  wave went 0 for 3 out-of-sample; the last went 3 for 4.
- **Parameter choice moves every pair.** Across lookback windows of 40–120 days, 9 of 10
  pairs change sign, and the pre-registered 60 is the only window with a positive mean.
  Switching the signal hedge from rolling to static moves individual pairs by up to 48 pp
  while moving the mean by 1 pp.
- **The cross-section is centred on zero.** A mean of +0.6% sits inside a 26 pp standard
  deviation.
- **The best pair is what selection from noise looks like.** UNP/CSX's raw p = 0.039
  becomes 0.35 after Holm.

On costs, the per-pair breakeven cost is bimodal:

- **No gross edge at any cost:** WM/RSG, SPY/VOO, XOM/CVX and UPS/FDX lose money even with
  costs set to zero.
- **Clear the charge comfortably:** UNP/CSX, DUK/SO, KO/PEP and HD/LOW break even at
  26–100 bps per side.
- **Decided by the cost assumption:** only MA/V and FOXA/FOX.

So most losing pairs have no gross edge, rather than an edge eaten by friction.

### Holdout: pre-registered, 2025-01 → 2026-08

[`PREREGISTRATION.md`](PREREGISTRATION.md) was committed before any price after 2024-12-31
was downloaded, and the holdout was scored once.

| | |
|---|---|
| Arm A, the frozen strategy | **2 of 10** profitable, mean **−3.5%** (p = 0.36), no pair significant |
| Best holdout Sharpe vs expected best-of-10 with no edge | **1.24 vs 1.23** |
| 2022–24 winners profitable again | **0 of 4**; Spearman ρ = **−0.56** (p = 0.09) |
| Arm B − Arm A, walk-forward re-estimation | **−7.8 pp** on 2022–24; **+2.9 pp** [−10, +14] on the holdout |
| Largest single-pair change from Arm B | 67 pp on 2022–24; 46 pp on the holdout |

The 2022–2024 numbers did not replicate, but the thesis did:

- UNP/CSX, the pair a reader would have kept after 2022–2024 (+55.9%), lost 11.5%.
- UPS/FDX, the pair they would have dropped (−42.4%), made 24.4%.

## Study design

| | |
|---|---|
| Universe | 10 pairs, 20 tickers, each specified from economic reasoning before its own backtest (`configs/pairs.yaml`) |
| Data | Yahoo Finance adjusted close, 2015-01-01 → 2024-12-30; holdout through 2026-08-28 |
| Split | In-sample through 2021-12-31; out-of-sample 2022–2024 (752 trading days), scored once |
| Spread | Log prices; rolling 60-day OLS hedge for the signal; static hedge fit in-sample only for sizing |
| Signal | 60-day rolling z-score; enter at \|z\| ≥ 2.0, exit at \|z\| ≤ 0.5, hold in between |
| Execution | Decided at the close of *t*, executed at *t+1* |
| Costs | 1 bp commission + 5 bps slippage per side, on `(1 + \|g\|)` of notional per unit traded |
| Inference | Lo (2002) Sharpe standard error; stationary bootstrap (2,000 draws, mean block 10, seed 0); Holm across pairs; expected maximum of N null Sharpes (Bailey & López de Prado 2014) |

### Methodological rules, enforced in code

1. **No look-ahead.** Positions lag one bar, and the sizing hedge is fit in-sample only.
   `test_engine.py` and `test_study.py` fail if a future or out-of-sample price can move a
   past result.
2. **No data-snooping on parameters.** Window, bands, costs, bootstrap scheme and split are
   fixed in the config. `notebooks/04` sweeps them as a reported finding and never feeds
   back into the config.
3. **No pair can be dropped or demoted.** `Pair` has no tier field, `run_study` cannot run a
   subset, and `to_frame` emits no ranking column.
4. **Gross and net are inseparable.** `summary()` returns both in one block, and every
   Sharpe carries its standard error.
5. **In-sample and out-of-sample are separated.** The holdout extends this with a plan
   committed before its data existed. A leak guard checks that no walk-forward refit sees
   data from its own segment.

## Reproducing

```bash
git clone https://github.com/NicWYZ/Pairs-Trading-Strategy-Teardown.git
cd Pairs-Trading-Strategy-Teardown
make install       # uv sync --extra dev
make run           # download prices, run the main study -> reports/results/
make run-holdout   # the pre-registered holdout       -> reports/results_holdout/
make notebooks     # execute all six notebooks top to bottom
make test lint typecheck
```

No market data is committed. The configs plus `uv.lock` reproduce every number.

`make run` writes:

- `metrics.csv`
- `inference.csv` (standard errors, bootstrap intervals and Holm p-values for every cell)
- the walk-forward tables
- `run_manifest.json`
- three figures per pair

The notebooks read these files rather than recomputing them. Notebook 05 also reads
`sensitivity.csv`, which notebook 04 writes, so run the notebooks in order.

## Repository layout

```
configs/            pairs.yaml (main study), pairs_holdout.yaml (holdout; frozen)
PREREGISTRATION.md  holdout plan, committed before the holdout data was downloaded
src/pairs_teardown/
  data/             download + parquet cache, alignment
  stats/            cointegration (Engle–Granger, Johansen, OU half-life);
                    inference (Lo SE, stationary bootstrap, Holm, expected max Sharpe)
  signals/          rolling hedge, z-score, entry/exit rules with hysteresis
  backtest/         engine (one-bar lag, hedge-stability guards) and cost model
  metrics/          Sharpe with SE, drawdown, turnover, gross/net summary
  study.py          end-to-end pair runs: frozen split (Arm A) and walk-forward (Arm B)
scripts/            download, run, cache-path helper
notebooks/          01 data → 02 cointegration → 03 results → 04 sensitivity
                    → 05 writeup → 06 holdout
tests/              138 tests on synthetic data with known answers; no network access
```

## Limitations

- **Small sample.** Ten pairs over three out-of-sample years cannot resolve a mean this
  close to zero. Against a 26 pp standard deviation, a ±2% mean needs roughly 650 pairs.
  Halving the 0.58 Sharpe standard error needs twelve years per pair.
- **Selection.** Pairs were chosen in waves, waves 2 and 3 with earlier results in view.
  Nothing was dropped, but this is not one-shot pre-registration.
- **Survivorship.** All tickers are companies that still traded in 2026. That biases
  results upward; correcting it needs point-in-time constituents with delisting returns.
- **Costs.** A flat per-side cost with no market-impact model, no borrow costs and no
  short-availability constraints.
- **Sizing.** One spread unit is long $1 of A and short $g of B, so gross exposure differs
  across pairs (1.45× to 2.22×). Returns are not volatility-scaled.
- **FOXA/FOX** has 709 in-sample days against 1,763 for every other pair, because FOX
  listed in 2019.
- **Scope.** US large caps, one decade, daily bars.

The full argument is in [`notebooks/05_writeup.ipynb`](notebooks/05_writeup.ipynb) and the
holdout in [`notebooks/06_holdout.ipynb`](notebooks/06_holdout.ipynb).

## Citation

Cite the tagged release `v1.0`, which is the state of the repository these results come
from.

```
Zhang, N. (2026). Pairs-Trading Strategy Teardown (v1.0) [Computer software].
https://github.com/NicWYZ/Pairs-Trading-Strategy-Teardown
```
