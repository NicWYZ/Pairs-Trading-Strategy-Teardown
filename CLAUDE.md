# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

**Stage 4 (backtest engine + costs) is complete.** Completed so far:

- `src/pairs_teardown/data/loaders.py` — `load_or_download` downloads adjusted-close prices
  via yfinance and caches to parquet in `data/raw/`. Cache key encodes tickers + date range;
  re-running is safe.
- `src/pairs_teardown/data/clean.py` — align/clean logic (drops NaN-only rows, handles
  FOXA/FOX which only trades from 2019-03-13 onward after the Disney deal closed).
- `scripts/download_data.py` — entry point; prints per-pair summaries after cleaning.
- Raw data cached at `data/raw/FOX_FOXA_RSG_SPY_VOO_WM_20150101_20241231.parquet`.
- `src/pairs_teardown/stats/cointegration.py` — `estimate_hedge_ratio` (OLS, IS window only),
  `build_spread`, `adf_pvalue`, `engle_granger_pvalue`, and `analyze_pair` (bundles all four
  into a `CointegrationResult` dataclass).
- `src/pairs_teardown/signals/spread.py` — `rolling_zscore` using trailing window only
  (no look-ahead).
- `src/pairs_teardown/signals/rules.py` — `target_positions` with entry/exit thresholds
  and hysteresis (holds prior position between thresholds).
- `src/pairs_teardown/backtest/costs.py` — `CostModel` dataclass; charges commission +
  slippage proportional to position changes.
- `src/pairs_teardown/backtest/engine.py` — `run_backtest` lags target positions by one bar
  (decision at *t*, executed at *t+1*), computes daily P&L, applies costs. 28 tests pass.

Still to build: `metrics/`, `plotting/`, `config.py`, `scripts/run_backtest.py`,
`configs/pairs.yaml`.

### Critical design note: signal hedge ratio vs. sizing hedge ratio

Per PROJECT_PLAN.md §3.1, the engine uses **two distinct hedge ratio estimates**:
- **Signal hedge ratio** (rolling window, e.g. 60-day): used to build the spread and
  z-score. Responsiveness to drift is valuable here.
- **Sizing hedge ratio** (static, full in-sample OLS): used to size the actual leg-level
  trade in the engine. Stability is critical — a noisy rolling estimate whose range crosses
  zero can flip the hedge sign, turning a market-neutral position into a large directional
  bet. Synthetic stress tests showed +521% gross return with the stable estimate vs. -61.6%
  with a noisy rolling one (std 0.31 on WM/RSG). See
  `test_engine.py::test_correct_hedge_sign_neutralizes_common_move_wrong_sign_does_not`.

### Environment note — `ModuleNotFoundError: No module named 'pairs_teardown'`

Root cause: macOS sets the `UF_HIDDEN` flag on the venv and the files uv writes into it, and
CPython 3.12's `site` module silently **skips hidden `.pth` files** — so uv's editable-install
`.pth` is ignored and the package can't be imported. uv re-hides the whole `.venv` tree on
every `uv sync`/reinstall, so any `.pth`-based fix keeps breaking. (This is the venv's own
Python 3.12.9 — not a kernel-selection problem.)

The durable fix is **`sitecustomize.py`** in site-packages: Python auto-imports it at every
interpreter startup, and regular module import is *not* filtered by `UF_HIDDEN` (only `.pth`
processing is). So it puts `src/` on `sys.path` even when the whole venv is hidden and even
with no `PYTHONPATH` set — which covers the VS Code auto-discovered `.venv` kernel, `uv run`,
and plain `python` alike. Belt-and-suspenders: `conftest.py` (repo root, git-tracked) does the
same for `uv run pytest` and survives a full `rm -rf .venv`.

`sitecustomize.py` lives inside the (gitignored) venv, so after
`rm -rf .venv && uv sync --extra dev` run **`./scripts/fix_venv.sh`** to regenerate it. That
script is the one command to run if imports ever break again.

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
uv sync --extra dev           # install/sync all deps including dev (preferred over uv pip install -e)
uv run python scripts/download_data.py   # fetch + cache raw prices
pytest                        # run all tests (testpaths = ["tests"])
pytest tests/test_engine.py -k look_ahead   # run a single test
ruff check .                  # lint
mypy src                      # type check
```

### Environment gotchas

- **Always use `uv sync --extra dev`** to set up the environment, not `uv pip install -e ".[dev]"`.
  Using `uv pip install` can install into a different Python version than what `uv sync` uses,
  corrupting the venv with two `python3.X` directories — the editable `.pth` lands in the wrong
  one and `import pairs_teardown` silently fails. If this happens, `rm -rf .venv && uv sync --extra dev`.
- **`uv run` uses the lockfile's Python** (currently CPython 3.12 via miniconda). Don't override
  with a system Python or the venv will split again.
- **Build backend is `hatchling`** (not setuptools). Miniconda's Python 3.12 skips `.pth` files
  whose names start with `__`, which is exactly what setuptools names its editable install file
  (`__editable__.pairs_teardown-0.1.0.pth`). Hatchling names it `_editable_impl_*` (single
  underscore), which Python processes normally. Do not switch back to setuptools.
- The editable install persists across sessions once the venv exists — no need to reinstall.

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
