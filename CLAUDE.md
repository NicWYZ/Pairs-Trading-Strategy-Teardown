# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

This repo is in early scaffolding: `src/pairs_teardown/` currently contains only empty
package `__init__.py` files and a top-level `__version__`. There is no `config.py`,
no `data/loaders.py`, no backtest engine, no scripts, and no `configs/pairs.yaml` yet.
**`PROJECT_PLAN.md` is the authoritative build guide** — read it before adding any module.
It specifies exact file purposes, function signatures, dependencies, and a strict build
order (Stage 0 → Stage 8). Follow that order: don't implement the backtest engine before
the data/cointegration/signal layers it depends on, and don't add tooling (pre-commit, CI)
until the core pipeline exists.

## What this project is

A pairs-trading **teardown**, not a strategy pitch. The goal is to honestly test whether
classic pairs-trading edges (on WM/RSG, FOXA/FOX, SPY/VOO) survive realistic transaction
costs and a strict in-sample/out-of-sample split. A post-cost decay in performance is the
expected, correct result — treat it as a finding to report, not a bug to fix.

## Commands

No `Makefile` exists yet (see PROJECT_PLAN.md §5 for the planned `make install/test/lint/data/run`
wrappers). Until it's added, use the underlying tools directly:

```bash
uv pip install -e ".[dev]"   # install package + dev deps (pytest, ruff, mypy, pre-commit)
pytest                        # run all tests (testpaths = ["tests"])
pytest tests/test_engine.py -k look_ahead   # run a single test
ruff check .                  # lint
mypy src                      # type check
```

## Architecture (per PROJECT_PLAN.md)

Pipeline: `raw prices → clean/align → cointegration test → estimate hedge ratio →
construct spread → rolling z-score → entry/exit rules → target positions →
backtest engine (lag positions by 1 bar; apply costs) → equity curve → performance metrics
(gross & net, in-sample & out-of-sample)`. Each arrow is a function in a dedicated module
under `src/pairs_teardown/`:

- `data/loaders.py`, `data/clean.py` — download (yfinance) + cache to parquet, align/clean.
- `stats/cointegration.py` — `estimate_hedge_ratio`, `build_spread`, `adf_pvalue`,
  `engle_granger_pvalue`. Hedge ratio is fit on the in-sample window **only**.
- `signals/spread.py`, `signals/rules.py` — `rolling_zscore` (trailing window only) →
  `target_positions` (entry/exit thresholds with hysteresis: holds prior position
  between thresholds).
- `backtest/engine.py`, `backtest/costs.py` — `run_backtest` is the most important
  function in the repo. It must lag `target_positions` by one bar before computing
  P&L (decision at day *t*, executed at *t+1*) and must never reference future rows
  when computing the return at row *t*. `costs.py` charges commission + slippage
  proportional to position changes.
- `metrics/performance.py` — Sharpe, max drawdown, turnover, summary stats.
- `plotting/charts.py` — spread/z-score, equity curve, drawdown figures for notebooks/reports.
- `config.py` (not yet created) — loads `configs/pairs.yaml` into a typed config
  (pairs, date ranges, z-score window, entry/exit thresholds, cost params, IS/OOS split date).

Orchestration is config-driven: `scripts/run_backtest.py` (not yet created) reads
`configs/pairs.yaml` and runs the full chain for all three pairs, writing a metrics CSV to
`reports/results/` and figures to `reports/figures/`. `data/` and `reports/{figures,results}/`
are gitignored except `.gitkeep` — never commit market data or generated outputs.

## Non-negotiable methodological rules

These five principles drive every design decision and are explicitly tested, not just stated:

1. **No look-ahead bias.** Positions are lagged by one bar in the engine; hedge ratio and
   z-score parameters are estimated on the in-sample window only. Enforced by a dedicated
   look-ahead guard test in `test_engine.py` (altering a future price must not change any
   past P&L value).
2. **Survivorship bias** is acknowledged in the writeup, not engineered around.
3. **No data-snooping.** The three pairs are pre-specified from economic reasoning, not
   mined by scanning combinations. Only a small, pre-declared parameter grid is tuned, and
   only on in-sample data.
4. **Gross vs. net always reported side by side** — every result appears before and after
   transaction costs.
5. **In-sample vs. out-of-sample always separated** — parameters come from IS data only;
   OOS is evaluated once and reported as-is.

When extending this project: keep all logic in `src/pairs_teardown/` modules with type
hints and a matching test in `tests/`; keep notebooks thin (import from the package, call,
plot) — no real logic in notebooks. Tests use synthetic data with known answers (e.g. a
random walk vs. `2*b + stationary_noise` for a known-cointegrated fixture) — never live
downloads in tests.
