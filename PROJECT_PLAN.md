# Pairs-Trading Strategy Teardown — Project Plan

> A rigorous, honest evaluation of whether classic pairs-trading edges survive
> realistic transaction costs and out-of-sample testing, built as production-grade
> research code. This document is the authoritative build guide: anyone following
> it should be able to reproduce the project exactly.

---

## 1. Motivation & Background

Pairs trading is one of the oldest and most-taught quantitative strategies. The idea
is simple: find two assets that move together for an economic reason, and when the gap
(the "spread") between them widens abnormally, bet that it will close — buy the
relatively cheap leg, short the relatively expensive leg, and profit on reconvergence.
Because the strategy is so well documented, it is also the single most replicated
"quant side project" in existence, and almost every replication makes the same mistake:
it stops at a pretty equity curve computed with no trading costs, on data that was used
both to choose the strategy and to test it.

This project is deliberately **not** an attempt to find a profitable strategy. It is a
*teardown*: we implement pairs trading carefully and then interrogate whether its
apparent edge is real once we confront it with the two things tutorials ignore —
realistic transaction costs and a strict separation between the data used to fit the
strategy and the data used to evaluate it. The expected and entirely acceptable finding
is that the edge shrinks or vanishes. In quantitative research, a clean demonstration of
*why* a strategy fails after costs is far more credible than a suspiciously profitable
backtest, which an experienced reader assumes hides a look-ahead bug.

The project carries a second, equal goal: to be built as **production-level research
code** rather than a notebook monolith. All reusable logic lives in tested, importable
Python modules; notebooks are kept thin and used only for exploration and the final
writeup. The discipline this enforces — modular design, unit tests, look-ahead guards,
config-driven experiments, continuous integration — is exactly the difference between
"analysis script" and "research code" that quantitative employers care about, and it
transfers directly to any future research codebase.

We study **ten pairs** chosen to span the full spectrum of how two assets can be linked,
because the *type* of link determines how strong and stable the relationship is, and
contrasting the strategy's behavior across them is what lifts this above a single-pair
demo. Three of them define the range:

- **WM / RSG** (Waste Management / Republic Services) — an economic duopoly. The two
  firms face nearly identical demand, costs, and regulation, so any cointegration is
  *economic*, real but breakable. The most interesting case to analyze.
- **FOXA / FOX** (Fox Corp Class A / Class B) — two share classes of the *same* company.
  Cointegration is near-mechanical and the spread is tiny, making this a clean test of
  whether anything at all survives transaction costs.
- **SPY / VOO** (two S&P 500 ETFs) — a deliberate near-degenerate **control**. They track
  the same index, so they are about as cointegrated as two things can be, yet the spread
  is essentially zero. The punchline: even a near-perfect statistical relationship yields
  nothing tradeable after costs.

The remaining seven are ordinary economic links across different sectors — KO/PEP and
MA/V (duopolies), XOM/CVX (common commodity factor), HD/LOW (housing cycle), UPS/FDX
(parcel volume), UNP/CSX (freight cycle) and DUK/SO (regulated utilities). They exist to
give the study a cross-section: with three pairs you can only report three anecdotes, and
the study's actual finding turned out to be about **dispersion across pairs**, which three
cannot measure at all.

A note on research integrity that must appear in the final writeup: every pair is
**pre-specified from economic reasoning**, not selected by scanning thousands of
combinations for whichever backtested best, and **every pair is reported whatever it did**.

Two honest qualifications on that, both of which belong in the writeup rather than being
smoothed over:

1. The universe was assembled in three waves (see `configs/pairs.yaml`). Each wave was
   fixed before it was run, but waves 2 and 3 were chosen by someone who already knew how
   wave 1 had done. The claim is "pre-specified per wave", not "the whole universe was
   pre-registered in one act".
2. An earlier version of this project split the pairs into an `official` headline tier and
   a `sanity_check` tier. **That was a mistake and has been removed.** Wave 1 went 0 for 3
   out-of-sample while wave 3 went 3 for 4 — so the tiering would have let the same data,
   the same code and the same parameters support opposite headlines depending only on which
   pairs had been written down first. There is now no tier field anywhere in the schema,
   and `test_study.py::test_table_has_no_tier_column` fails if one returns.

---

## 2. What the Project Does (pipeline at a glance)

```
raw prices ─▶ clean/align ─▶ cointegration test ─▶ estimate hedge ratio
   ─▶ construct spread ─▶ rolling z-score ─▶ entry/exit rules ─▶ target positions
   ─▶ backtest engine (lag positions by 1 bar; apply costs) ─▶ equity curve
   ─▶ performance metrics (gross & net, in-sample & out-of-sample) ─▶ writeup
```

Each arrow is a function in a dedicated module. Nothing runs automatically; an
orchestration script (or a thin notebook) calls the chain in order, driven by a single
YAML config file.

---

## 3. Methodological Principles (the core of the project)

These five principles *are* the differentiator. Every component is designed to honor them,
and the writeup is organized around them.

1. **No look-ahead bias.** A decision made at the close of day *t* may only use
   information available up to day *t*, and is executed at day *t+1*. Mechanically, the
   engine lags the target-position series by one bar before computing P&L. The hedge ratio
   and z-score parameters are estimated on the in-sample window only, never on data the
   strategy will later be "tested" on. This is enforced by an explicit unit test.

2. **Survivorship bias is acknowledged.** Free data sources list only currently-traded
   tickers; delisted firms have vanished, flattering any historical strategy. All three
   pairs here are currently-listed, so the bias is small, but the writeup states the
   limitation explicitly.

3. **No data-snooping.** Pairs are pre-specified from economic logic, not mined. We do not
   search over thresholds/windows to maximize the backtest; a small, pre-declared grid is
   tuned on in-sample data only, and the out-of-sample result is reported untouched.

4. **Gross vs. net is always reported side by side.** Every result appears twice: before
   costs (gross) and after a realistic transaction-cost model (net). The collapse from one
   to the other is the headline finding.

5. **In-sample vs. out-of-sample is always separated.** Data is split by date. Hedge ratio
   and any parameter choices come from the in-sample period; the out-of-sample period is
   evaluated once and reported as-is.

### 3.1 Signal hedge ratio vs. sizing hedge ratio (empirically motivated)

The hedge ratio plays two distinct roles, and this project deliberately uses
a DIFFERENT estimate for each:

- **Signal hedge ratio** (rolling, e.g. 60-day window): used to build the
  spread and z-score. Responsiveness to drift is valuable here, and moderate
  estimation noise only shifts the timing of entries/exits.

- **Sizing hedge ratio** (static, full in-sample OLS): used to size the
  actual leg-level trade in the backtest engine. Stability is critical here,
  because sizing noise doesn't just blur a signal -- it can silently break
  the hedge itself. A hedge ratio near zero or of the wrong sign turns a
  market-neutral position into a large, unintended directional bet on the
  common factor between the two assets, whose variance typically dwarfs the
  idiosyncratic mean-reverting signal the strategy is trying to capture.

  Empirical evidence (synthetic stress test, documented in project history):
  identical signal and engine, differing only in which hedge ratio sized the
  trade, produced +521% gross return with the true/stable hedge ratio vs.
  -61.6% with a noisy 60-day rolling estimate whose range crossed zero
  (-1.63 to +5.40 around a true value of 1.0). On the real pairs, WM/RSG
  (the pair with a genuine structural break) had the noisiest rolling hedge
  ratio (std 0.31); SPY/VOO (no real drift) had the most stable (std 0.002)
  -- consistent with the earlier finding that rolling estimation helps
  drifting relationships and hurts stable ones (Section on bias-variance
  tradeoff).

  See `tests/test_engine.py::test_correct_hedge_sign_neutralizes_common_move_wrong_sign_does_not`
  for a minimal, deterministic proof of the underlying mechanism.

---

## 4. Repository Structure

```
pairs-teardown/
├── README.md                     # short public overview (separate from this file)
├── PROJECT_PLAN.md               # this document
├── pyproject.toml                # packaging + dependency + tool configuration
├── uv.lock                       # exact pinned dependency versions (generated)
├── .python-version               # Python version pin for uv (e.g. 3.12)
├── .gitignore
├── .pre-commit-config.yaml       # auto checks before each commit
├── Makefile                      # convenience commands (make test / lint / run)
├── LICENSE                       # MIT
├── .github/
│   └── workflows/
│       └── ci.yml                # run tests + lint on every push
├── src/
│   └── pairs_teardown/           # the importable package
│       ├── __init__.py
│       ├── config.py             # load + validate the YAML config
│       ├── study.py              # run a configured pair end-to-end; tabulate runs
│       ├── data/
│       │   ├── __init__.py
│       │   ├── loaders.py        # download prices (yfinance) + cache to parquet
│       │   └── clean.py          # align dates, handle gaps, adjusted close
│       ├── stats/
│       │   ├── __init__.py
│       │   └── cointegration.py  # ADF, Engle–Granger p-value, OLS hedge ratio
│       ├── signals/
│       │   ├── __init__.py
│       │   ├── spread.py         # build spread, rolling mean/std, z-score
│       │   └── rules.py          # z-score → target positions (entry/exit)
│       ├── backtest/
│       │   ├── __init__.py
│       │   ├── engine.py         # bar-by-bar simulator (lags positions; no look-ahead)
│       │   └── costs.py          # commission + spread/slippage cost model
│       ├── metrics/
│       │   ├── __init__.py
│       │   └── performance.py    # Sharpe, max drawdown, turnover, summary stats
│       └── plotting/
│           ├── __init__.py
│           └── charts.py         # spread/z-score, equity curve, drawdown plots
├── tests/
│   ├── conftest.py               # shared synthetic-data fixtures
│   ├── test_clean.py
│   ├── test_cointegration.py
│   ├── test_spread.py
│   ├── test_rules.py
│   ├── test_engine.py            # *** look-ahead guard + hand-computed P&L ***
│   ├── test_costs.py
│   ├── test_metrics.py
│   ├── test_config.py
│   └── test_study.py             # *** in-sample-only hedge leak guard ***
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_cointegration_analysis.ipynb
│   ├── 03_backtest_results.ipynb   # authoritative results tables + figures
│   ├── 04_sensitivity_analysis.ipynb  # every frozen parameter, swept
│   └── 05_writeup.ipynb            # final narrative + honest conclusions
├── configs/
│   └── pairs.yaml                # tickers, dates, thresholds, costs, IS/OOS split
├── scripts/
│   ├── download_data.py          # fetch + cache all data (entry point)
│   └── run_backtest.py           # full pipeline from config → reports/
├── data/                         # GITIGNORED — never commit market data
│   ├── raw/.gitkeep
│   └── processed/.gitkeep
└── reports/
    ├── figures/.gitkeep
    └── results/.gitkeep          # metrics tables (CSV) written here
```

---

## 5. File-by-File Specification

For each file: its purpose, the packages it uses, and the key functions/contents it should
contain. Function signatures are guidance, not gospel — keep them small, typed, and
single-purpose.

### Root configuration files

**`pyproject.toml`** — the project's identity card and tool control panel. Declares the
package name, runtime dependencies, and the configuration blocks for ruff, mypy, and
pytest. No external Python packages; read by `uv`/`pip` and the dev tools. Minimal contents:

```toml
[project]
name = "pairs-teardown"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "pandas>=2.0",
    "numpy>=1.26",
    "statsmodels>=0.14",
    "yfinance>=0.2",
    "pyyaml>=6.0",
    "matplotlib>=3.8",
    "pyarrow>=15.0",     # parquet caching
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "ruff>=0.4", "mypy>=1.10", "pre-commit>=3.7",
       "pandas-stubs>=2.0", "types-PyYAML>=6.0",
       # a uv venv has no pip, so VS Code cannot install these on demand
       "ipykernel>=6.29", "nbconvert>=7.16"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 100

[tool.mypy]
python_version = "3.11"
ignore_missing_imports = true   # third-party stubs (yfinance etc.) are incomplete

[tool.pytest.ini_options]
testpaths = ["tests"]
```

**`.gitignore`** — must ignore `data/` (except `.gitkeep`), `reports/figures` and
`reports/results` output, `.venv/`, `__pycache__/`, `.ipynb_checkpoints/`, `.mypy_cache/`,
`.ruff_cache/`, `.pytest_cache/`. Start from the standard GitHub Python template and add
`data/raw/*`, `data/processed/*`, `reports/figures/*`, `reports/results/*` (keeping the
`.gitkeep` files).

**`.pre-commit-config.yaml`** — runs `ruff` (lint + format), `ruff-format`, and
`nbstripout` (strip notebook outputs) before each commit. No Python packages directly;
references tool repos.

**`Makefile`** — thin wrappers: `make install`, `make test` (`pytest`), `make lint`
(`ruff check . && mypy src`), `make data` (`python scripts/download_data.py`), `make run`
(`python scripts/run_backtest.py`).

**`.github/workflows/ci.yml`** — on every push/PR: set up Python, `uv pip install -e
".[dev]"`, run `ruff check`, `mypy src`, and `pytest`. This produces the green CI badge.

### Package: `src/pairs_teardown/`

**`__init__.py`** (top level) — marks the package; may expose a tiny curated API (e.g.
`from .config import load_config`). Can be nearly empty.

**`study.py`** — the pipeline chain in one place: `run_pair` (prices -> spread -> z-score
-> positions -> backtest -> metrics for all three periods), `run_study`, and `to_frame`.
Lives in the package, not in `scripts/`, so that notebooks import the chain instead of
reimplementing it and so that the in-sample-only hedge fit is under test. **The static
sizing hedge ratio is fit on in-sample rows only, then frozen across the split.**

**`config.py`** — loads and validates `configs/pairs.yaml` into typed objects.
Packages: `pyyaml`, `dataclasses` (stdlib), `pathlib` (stdlib).
Key contents: a `@dataclass` (e.g. `BacktestConfig`) holding pairs, date ranges, z-score
window, entry/exit thresholds, cost parameters, and the in-sample/out-of-sample split date;
plus `load_config(path) -> BacktestConfig` that parses the YAML and checks required fields.
(Optional upgrade: swap dataclasses for `pydantic` for richer validation.)

**`data/loaders.py`** — downloads daily prices and caches them so you never re-hit the
network unnecessarily. Packages: `yfinance`, `pandas`, `pyarrow`, `pathlib`.
Key functions: `download_prices(tickers, start, end) -> pd.DataFrame` (adjusted close,
one column per ticker); `load_or_download(tickers, start, end, cache_dir) -> pd.DataFrame`
(reads `data/raw/*.parquet` if present, otherwise downloads and writes the cache).

**`data/clean.py`** — turns raw downloads into an analysis-ready, aligned price panel.
Packages: `pandas`, `numpy`.
Key functions: `align_prices(df) -> pd.DataFrame` (inner-join on dates so both legs exist);
`handle_missing(df) -> pd.DataFrame` (forward-fill small gaps or drop; document the choice);
`to_log_prices(df) -> pd.DataFrame` (optional helper). Always work from adjusted close.

**`stats/cointegration.py`** — the statistical core. Packages: `statsmodels`, `numpy`,
`pandas`. Key functions:
- `estimate_hedge_ratio(a, b) -> float` — OLS of price A on price B (with intercept);
  return the slope. **Estimate on the in-sample window only.**
- `build_spread(a, b, hedge_ratio) -> pd.Series` — `a - hedge_ratio * b`.
- `adf_pvalue(series) -> float` — Augmented Dickey–Fuller test for stationarity of the spread.
- `engle_granger_pvalue(a, b) -> float` — `statsmodels.tsa.stattools.coint`, returns p-value.
- A small `CointegrationResult` dataclass bundling hedge ratio + p-values is convenient.

**`signals/spread.py`** — converts the spread into a standardized signal.
Packages: `pandas`, `numpy`. Key functions:
- `rolling_zscore(spread, window) -> pd.Series` — `(spread - rolling_mean) / rolling_std`,
  using a trailing window so it only ever uses past+current data (no look-ahead).

**`signals/rules.py`** — converts the z-score into target positions in the spread.
Packages: `pandas`, `numpy`. Key function:
- `target_positions(zscore, entry, exit) -> pd.Series` — returns +1 (long spread) when
  z < -entry, -1 (short spread) when z > +entry, 0 when |z| < exit, and **holds the prior
  position otherwise** (hysteresis). Position is in units of "the spread"; the engine
  translates that into leg-level trades using the hedge ratio.

**`backtest/engine.py`** — the simulator, and the most important file to get right.
Packages: `pandas`, `numpy`. Key function:
- `run_backtest(prices, target_positions, hedge_ratio, cost_model) -> BacktestResult` —
  **lags `target_positions` by one bar** (decision at *t*, executed at *t+1*), computes
  daily spread P&L, subtracts costs on every position change, and returns a result object
  with the daily returns series, equity curve, and realized turnover. Must contain no
  operation that references future rows when computing the return at row *t*.

**`backtest/costs.py`** — realistic frictions. Packages: `numpy`, `pandas`. Key contents:
- A `CostModel` dataclass with commission (per unit notional) and half-spread/slippage
  (in basis points). `apply(turnover, notional) -> cost` charges costs proportional to the
  size of each position change. Parameters come from config so the teardown can re-run under
  optimistic vs. pessimistic assumptions.

**`metrics/performance.py`** — scoring. Packages: `numpy`, `pandas`. Key functions:
- `sharpe_ratio(returns, periods_per_year=252) -> float`
- `max_drawdown(equity_curve) -> float`
- `turnover(positions) -> float`
- `summary(result) -> dict` — bundles the above plus total/annualized return and hit rate,
  ready to be tabulated gross-vs-net and IS-vs-OOS.

**`plotting/charts.py`** — visualization for notebooks/reports. Packages: `matplotlib`,
`pandas`. Key functions: `plot_spread_zscore(...)`, `plot_equity_curve(...)`,
`plot_drawdown(...)`. Return figures; let callers save them to `reports/figures/`.

### Tests: `tests/`

All tests use synthetic data with known answers — never live downloads.
Packages: `pytest`, `pandas`, `numpy`, plus the package under test.

- **`conftest.py`** — fixtures: a known-cointegrated pair (e.g. `b` = random walk,
  `a = 2*b + stationary_noise`), a non-cointegrated pair (two independent random walks),
  and a tiny hand-built price/position example for the engine.
- **`test_clean.py`** — alignment drops unmatched dates; missing-value policy behaves.
- **`test_cointegration.py`** — hedge ratio recovers the known slope (≈2); the cointegrated
  fixture gives a small p-value, the independent fixture a large one.
- **`test_spread.py`** — z-score has ~0 mean / ~1 std on a stationary input; uses only
  trailing data.
- **`test_rules.py`** — entry/exit/hysteresis transitions fire at the correct thresholds.
- **`test_engine.py`** — **the keystone tests**: (a) a *look-ahead guard* — altering a
  future price must not change any past P&L value, and the return at *t* must depend only on
  the position decided at *t-1*; (b) a *hand-computed P&L* — a 3–4 bar example worked out by
  hand matches the engine to the cent, including costs.
- **`test_costs.py`** — zero turnover ⇒ zero cost; cost scales linearly with position change.
- **`test_metrics.py`** — Sharpe/drawdown match closed-form values on a constructed series.
- **`test_study.py`** — **the second keystone test**: shocking out-of-sample prices must
  leave the fitted static sizing hedge ratio, and every in-sample metric, unchanged. A
  full-sample fit fails it. Without this, the in-sample-only discipline is guaranteed by
  a docstring alone.

### Notebooks: `notebooks/` (thin — import from the package)

- **`01_data_exploration.ipynb`** — download via the package, plot the pairs' prices,
  check ranges and gaps.
- **`02_cointegration_analysis.ipynb`** — for each pair: hedge ratio, Engle–Granger/ADF
  p-values, plot spread and z-score. Discuss which pairs are cointegrated and how stably.
- **`03_backtest_results.ipynb`** — the authoritative results. Reads
  `reports/results/metrics.csv` and `run_manifest.json` (so it cannot disagree with what the
  pipeline computed), tabulates gross-vs-net and IS-vs-OOS for every pair, reconciles the
  cost drag against `turnover x (1+|g|) x bps`, reports the cross-sectional dispersion that
  is the study's actual finding, and regenerates equity/drawdown figures via
  `pairs_teardown.study`. Contains **no** robustness checks — those are notebook 04, kept
  separate so results and sensitivity cannot be mistaken for one another.
- **`04_sensitivity_analysis.ipynb`** — every frozen parameter varied one at a time through
  the same `run_pair`: z-score window, entry/exit bands, cost level (including a per-pair
  breakeven), split date, and rolling-vs-static signal hedge. Writes
  `reports/results/sensitivity.csv`, which notebook 05 reads rather than transcribing.
  Explicitly **not** a parameter search: nothing here may feed back into
  `configs/pairs.yaml`, since adopting a swept value would convert an out-of-sample result
  into an in-sample one.
- **`05_writeup.ipynb`** — the narrative: methodology, results, and the honest conclusion,
  organized around the five principles in Section 3. States the survivorship and
  data-snooping positions explicitly, and cites (not recomputes) notebooks 02 and 04.

### Config, scripts, data, reports

- **`configs/pairs.yaml`** — see Section 7. The single place experiment parameters live.
- **`scripts/download_data.py`** — imports `data.loaders`, downloads + caches all six
  tickers. Run once. Packages: the package + `pyyaml`.
- **`scripts/run_backtest.py`** — imports the full chain, reads the config, runs every pair,
  writes a metrics CSV to `reports/results/` and figures to `reports/figures/`. This is the
  one-command reproduction of the whole study.
- **`data/raw|processed/`** — gitignored caches (parquet). `.gitkeep` preserves the folders.
- **`reports/figures|results/`** — generated outputs; also gitignored except `.gitkeep`.

---

## 6. Tooling & Environment

- **uv** — environment + dependency manager. Creates the virtualenv and writes `uv.lock`
  for exact reproducibility. (Alternatives: `pip` + `venv`, or `poetry`.)
- **ruff** — linter *and* formatter in one fast tool.
- **mypy** — static type checker; pairs with the type hints on every function.
- **pytest** — test runner; discovers and runs everything in `tests/`.
- **pre-commit** — runs ruff + nbstripout automatically before each commit.
- **GitHub Actions** — re-runs lint, types, and tests on a clean machine on every push.

Stage these in: get the package importing and one test passing first; add pre-commit and CI
only once a few real pieces exist (see Section 8). Do not let scaffolding block the science.

---

## 7. Configuration — example `configs/pairs.yaml`

```yaml
# Date range for the whole study
start: "2015-01-01"
end:   "2024-12-31"

# Split date: everything before = in-sample (fit), on/after = out-of-sample (evaluate)
in_sample_end: "2021-12-31"

# Signal parameters (tuned on in-sample ONLY)
zscore_window: 60       # trading days
entry_threshold: 2.0    # enter when |z| exceeds this
exit_threshold: 0.5     # exit when |z| falls below this

# Transaction-cost assumptions (re-run under optimistic vs pessimistic)
costs:
  commission_bps: 1.0   # per side, basis points of notional
  slippage_bps: 5.0     # half-spread / slippage, basis points

pairs:
  - name: "WM_RSG"
    legs: ["WM", "RSG"]
  - name: "FOXA_FOX"
    legs: ["FOXA", "FOX"]
  - name: "SPY_VOO"      # degenerate control
    legs: ["SPY", "VOO"]
```

---

## 8. Build Order — Step-by-Step Guide

Work in stages. Each stage ends with something that runs and is tested before moving on.

### Stage 0 — Environment & skeleton
1. Install VS Code, git, and `uv`.
2. Create the folder tree (empty files are fine), including every `__init__.py` and
   `.gitkeep`. `git init`.
3. Write a minimal `pyproject.toml` (Section 5). `uv venv`, then
   `uv pip install -e ".[dev]"`.
4. Confirm `python -c "import pairs_teardown"` works. Write one trivial test and confirm
   `pytest` passes. **Do not proceed until imports and pytest both work** — this is the
   whole foundation.

### Stage 1 — Data layer
5. Implement `data/loaders.py` and `data/clean.py`. Write `scripts/download_data.py`.
6. Pull WM, RSG, FOXA, FOX, SPY, VOO; cache to `data/raw/`.
7. `tests/test_clean.py`. Build `01_data_exploration.ipynb`; eyeball the prices.

### Stage 2 — Cointegration
8. Implement `stats/cointegration.py` (hedge ratio, spread, ADF, Engle–Granger).
9. `tests/test_cointegration.py` using the synthetic fixtures.
10. `02_cointegration_analysis.ipynb`: report p-values per pair; plot spreads/z-scores.

### Stage 3 — Signals
11. Implement `signals/spread.py` (`rolling_zscore`) and `signals/rules.py`
    (`target_positions`).
12. `tests/test_spread.py`, `tests/test_rules.py`.

### Stage 4 — Backtest engine & costs (the crux)
13. Implement `backtest/costs.py`, then `backtest/engine.py` with the one-bar position lag.
14. Write `tests/test_engine.py` **first-class**: the look-ahead guard and the hand-computed
    P&L example. Then `tests/test_costs.py`. Nothing downstream is trustworthy until these
    pass.

### Stage 5 — Metrics & plotting
15. Implement `metrics/performance.py` and `plotting/charts.py`.
16. `tests/test_metrics.py` against closed-form values.

### Stage 6 — Orchestration
17. Implement `config.py` and write `configs/pairs.yaml`.
18. Write `scripts/run_backtest.py` tying the whole chain together; it should emit a metrics
    CSV and figures. Confirm `make run` reproduces results end-to-end from the config.

### Stage 7 — Analysis & writeup
19. Extract the pipeline chain from `scripts/run_backtest.py` into
    `src/pairs_teardown/study.py` (`run_pair`, `run_study`, `to_frame`), leaving the script
    as a CLI + IO wrapper. Notebooks must import the chain, never reimplement it — that is
    how the old exploratory notebook ended up with a full-sample hedge fit. Add
    `tests/test_study.py`, including the leak guard: shocking out-of-sample prices must not
    move the fitted sizing hedge ratio or any in-sample metric.
20. Retire the exploratory backtest notebook. Its full-sample hedge fit and un-split sample
    mean its numbers can never be quoted as findings; replace it rather than keep it.
21. `03_backtest_results.ipynb`: gross-vs-net and IS-vs-OOS tables for every pair, the
    cost-drag reconciliation, the cross-sectional dispersion statistics, and equity-curve
    and drawdown plots. Results only — no sweeps.
21b. `04_sensitivity_analysis.ipynb`: sweep every frozen parameter one at a time and write
    `reports/results/sensitivity.csv`. Sensitivity lives *after* the results notebook,
    because it is a check on them, not an input to them.
22. `05_writeup.ipynb`: the honest narrative, organized around the five principles. State
    the survivorship and data-snooping positions explicitly. Let the results say what they
    say — a decay to zero after costs is the expected, valid result.

### Stage 8 — Engineering polish
23. Add `.pre-commit-config.yaml` and run `pre-commit install`.
24. Add `.github/workflows/ci.yml`; confirm the badge goes green.
25. Write `README.md` (short public version: what, why, how to run, headline finding).
26. `uv lock` to pin versions.

---

## 9. How to Reproduce (exact commands)

```bash
git clone <repo-url> && cd pairs-teardown
make install                           # uv sync --extra dev (never `uv pip install -e`)
make test                              # all tests pass
make run                               # fetch prices if stale, then full study
                                       #   → reports/results + reports/figures
make notebooks                         # execute every notebook to check it still runs
# then open notebooks/03_backtest_results.ipynb for the tables,
#      notebooks/04_sensitivity_analysis.ipynb for the robustness checks,
#      and notebooks/05_writeup.ipynb for the narrative
```

---

## 10. Definition of Done

- [x] `import pairs_teardown` works in a fresh `uv` environment.
- [x] `pytest` passes, including the look-ahead guard and hand-computed P&L tests.
- [x] `make run` reproduces all metrics and figures from `configs/pairs.yaml` alone.
- [x] Every result is reported **gross and net** and **in-sample and out-of-sample**.
- [x] Hedge ratio and parameters are fit on in-sample data only.
- [x] No market data is committed to git; the download script + lockfile guarantee
      reproducibility.
- [x] CI is green; pre-commit is installed.
- [x] `05_writeup.ipynb` states the honest conclusion and explicitly addresses survivorship
      bias and data-snooping.

---

## 11. Notes for Claude Code

When extending this project: keep all logic in `src/pairs_teardown/` modules with type
hints and a matching test in `tests/`; keep notebooks thin (import, call, plot). Never
introduce an operation in `backtest/engine.py` that reads future rows when computing the
return at the current row — the look-ahead guard test exists to catch exactly that. Treat a
post-cost decay in performance as a correct result to report, not a bug to fix.
