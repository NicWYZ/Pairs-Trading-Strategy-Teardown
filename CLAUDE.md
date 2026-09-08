# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

**All stages (0-8) are complete.** The study runs end-to-end from the config, the results
and narrative notebooks are written, and pre-commit/CI/README/lockfile are in place.

**The universe is 10 pairs and completely flat.** There is no `official` vs `sanity_check`
distinction any more — see the note under "Critical design note: no pair tiers" below, which
explains why it was removed and why re-introducing it would be a real methodological
regression, not a stylistic choice. Completed so far:

- `src/pairs_teardown/data/loaders.py` — `load_or_download` downloads adjusted-close prices
  via yfinance and caches to parquet in `data/raw/`. Cache key encodes tickers + date range;
  re-running is safe.
- `src/pairs_teardown/data/clean.py` — align/clean logic (drops NaN-only rows, handles
  FOXA/FOX which only trades from 2019-03-13 onward after the Disney deal closed).
- Raw data cached at
  `data/raw/CVX_FOX_FOXA_KO_MA_PEP_RSG_SPY_V_VOO_WM_XOM_20150101_20241231.parquet`
  (all 12 tickers are always fetched together so the cache key stays stable).
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
  (decision at *t*, executed at *t+1*), computes daily P&L, applies costs. Includes
  `validate_sizing_hedge_ratio` / `UnstableHedgeRatioError`: a runtime guard that rejects
  any hedge ratio series with std > 0.05 or sign-instability before it can silently corrupt
  the hedge. Pass `skip_hedge_validation=True` only for deliberate stress-test work.
- `src/pairs_teardown/metrics/performance.py` — `sharpe_ratio`, `max_drawdown`, `turnover`,
  and `summary`. All pure functions of a return/position series, so every one is closed-form
  testable. Conventions are fixed and documented in the module docstring: simple daily
  returns, sample std (ddof=1), NaNs dropped rather than filled, drawdown returned as a
  signed non-positive fraction. `summary` duck-types its `result` argument (anything exposing
  `returns`, `gross_returns`, `held_positions`) so `metrics/` has no import dependency on
  `backtest/`, and returns `{"net": {...}, "gross": {...}}` — shaped for direct tabulation
  into the mandatory gross-vs-net table. Hit rate is computed over *active* days only;
  including flat days would measure trade frequency rather than edge.
- `src/pairs_teardown/plotting/charts.py` — `plot_spread_zscore`, `plot_equity_curve`,
  `plot_drawdown`. Each *returns* a `Figure` and never calls `plt.show` or `fig.savefig`, so
  the caller decides display vs. save. No rcParams styling is imposed. Series are passed to
  matplotlib via `.to_numpy()`, not `.values` — `.values` is typed as
  `ndarray | ExtensionArray | Categorical` and fails matplotlib's `ArrayLike` protocol under
  Pylance. `plot_equity_curve` overlays the optional gross curve on the net one so the
  transaction-cost wedge is visible as the gap between the lines.
- `src/pairs_teardown/config.py` — `load_config` parses `configs/pairs.yaml` into frozen
  dataclasses and *validates* it. Validation is not decorative: it rejects `entry <= exit`
  (incoherent hysteresis), a split date outside the data range (one evaluation period would
  be empty), duplicate pair names, and — hard-coded — `sizing_hedge: "rolling"`, which is
  the one misconfiguration that silently destroys market neutrality. `SplitConfig` owns
  `is_mask`/`oos_mask` so the IS/OOS boundary is defined in exactly one place.
- `configs/pairs.yaml` — single source of truth for every parameter, and committed, so it
  doubles as the record of what was run. One flat list of 10 pairs, each with a required
  `rationale`. Adding a pair means writing it here, running it once, and reporting it;
  removing one after seeing its result is the single thing that would invalidate the study.
- `src/pairs_teardown/study.py` — the pipeline chain: `run_pair`, `run_study`, `to_frame`.
  **It fits the static sizing hedge ratio on the in-sample window only**, then freezes it
  across the split; `03_backtest_explore.ipynb` fits on the full sample, which leaks. This
  lives in the package rather than in `scripts/` for two reasons: notebooks import the
  chain instead of reimplementing it (reimplementing it inline is exactly how notebook 03
  acquired its leak), and the discipline is testable. `run_study` has no subset argument
  and `to_frame` emits no tier column — an API that could quietly omit a pair would be a way
  to launder a bad result out of the study.
- `scripts/run_backtest.py` — the reproduction entry point, now only argparse + file
  writing over `study.py`. Writes `reports/results/metrics.csv` (long-form: pair x period
  x gross/net), a `run_manifest.json` recording parameters + hedge ratios + timestamp, and
  three figures per pair, closing matplotlib figures as it goes (10 pairs x 3 = 30 live
  figures otherwise).
- `scripts/download_data.py` — now config-driven (`--config`) rather than hardcoding
  tickers; still prints per-pair summaries after cleaning.
- `scripts/cache_path.py` — prints the loader's parquet cache path for a config, so the
  Makefile can express the pipeline's dependency graph without hardcoding a filename that
  would go stale.
- `Makefile` — `install/test/lint/typecheck/run/notebooks/clean/clean-data`. File targets
  encode the dependency graph, so `make run` re-runs only what is stale. Two non-obvious
  details, both load-bearing: `PYTHON := uv run python` (a bare `python` resolves to conda's
  and cannot import the package), and the download rule ends in `touch $@` (the loader
  returns a cache hit *without* touching the parquet, so without it the rule re-fires on
  every invocation and `make run` is never incremental).
- `notebooks/03_backtest_explore.ipynb` — **superseded**, kept deliberately. Predates the
  pipeline, reimplements it inline, and fits the sizing hedge on the *full* sample with no
  IS/OOS split, so its numbers are diagnostics, not findings; its header says so. Still the
  source for the COVID/2024 spike attributions and the window sweeps.
- `notebooks/04_backtest_results.ipynb` — the authoritative results. Reads `metrics.csv` +
  `run_manifest.json` (so it cannot disagree with the pipeline), tabulates gross-vs-net and
  IS-vs-OOS, reconciles the cost drag against `turnover x (1+|g|) x bps` (actual/predicted
  lands in 0.966-0.998, which is the evidence that the cost story is arithmetic and not an
  unexplained residual), reports the cross-sectional dispersion statistics that are the
  study's actual finding, and regenerates figures through `study.run_study`.
- `notebooks/05_writeup.ipynb` — the narrative, organized around the five principles. Its
  §2.3 documents the removal of the pair tiers as a methodological error the study made and
  corrected. Its §2.6 window sweep is the sharpest evidence: 9 of 10 pairs change sign
  across {40,50,60,75,90,120}, and the pre-registered 60 is the *only* window of the six
  with a positive cross-sectional mean — so the headline +0.6% is the best case, not the
  central case. Reported as a finding about instability, never used to pick a window.
- `.pre-commit-config.yaml` — ruff, ruff-format, nbstripout, plus whitespace/large-file
  checks. Fast hooks only; pytest and mypy stay in CI, because a slow commit hook gets
  bypassed and a bypassed hook enforces nothing. **nbstripout means committed notebooks
  carry no outputs** — run them locally to see figures.
- `.github/workflows/ci.yml` — uv sync, ruff check, ruff format --check, mypy, pytest. The
  suite is fully synthetic, so CI needs no market data and no network.
- `README.md` — the public-facing summary: the finding, how to run it, and where the five
  rules are enforced.

83 tests pass. `charts.py` has no direct tests; the logic that used to sit in
`scripts/run_backtest.py` is now covered via `study.py`. `make lint`, `make typecheck` and
`ruff format --check` are all clean.

**The headline finding changed when the universe grew from 6 to 10 pairs, and that change
is itself the result.** At 6 pairs the study reported a clean negative: costs kill the
edge. At 10 pairs, out-of-sample net: 4 of 10 profitable, range -42.4% (UPS/FDX) to +55.9%
(UNP/CSX), cross-sectional mean +0.6% (t=0.08, p=0.94) inside a 26pp standard deviation.
Costs still consume ~80% of the mean gross return but now flip the sign of only 2 pairs —
most losers had no gross edge at all, a distinction the 6-pair universe hid. So the
conclusion is about **variance, not mean**: dispersion across similar pairs dwarfs the
average effect, and any small-universe study reports whatever its pair selection produces.

**Do not "fix" this back into a clean negative result.** Reverting to a smaller or tiered
universe would restore a tidier headline by discarding the evidence that the headline was
never stable.

Remaining gap: `plotting/charts.py` still has no tests.

### Critical design note: no pair tiers

The study once split its pairs into an `official` headline set and a `sanity_check` set.
**That has been removed and must not come back.** The reason is empirical, not stylistic:
the first three pairs chosen went 0 for 3 out-of-sample while the last four went 3 for 4, so
the tiering would have let identical data, code and parameters support opposite headlines
depending only on which pairs were written down first. It was a machine for confirming
whichever conclusion the author reached for first.

The rule is now enforced structurally rather than by memory:
- `Pair` has no `group`/tier field and `load_config` cannot parse one.
- `to_frame` emits no column that could rank pairs.
- `run_study` has no subset argument, so no pair can be silently omitted from a run.
- `test_study.py::test_table_has_no_tier_column` and
  `test_config.py::test_pairs_load_as_one_flat_list` fail if any of that is undone.

If a future task seems to need a tier ("just report the interesting ones", "split out the
new pairs"), that is the failure mode these guards exist to catch. Report all pairs.

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
classic pairs-trading edges survive realistic transaction costs and a strict
in-sample/out-of-sample split, across 10 economically-linked US large-cap pairs. Whatever
the results say is the finding — a post-cost decay, a null, or a wide dispersion are all
valid outcomes to report, never bugs to fix.

## Commands

The `Makefile` is the entry point; every target shells out through `uv run python`.

```bash
make install        # uv sync --extra dev  (NOT uv pip install -e; see gotchas below)
make run            # full study: download if stale, then all pairs -> reports/
make test           # pytest -q
make lint           # ruff check src tests scripts
make typecheck      # mypy src
make notebooks      # execute every notebook to check it still runs
make clean          # wipe reports/ (keeps the cached prices)
make clean-data     # also wipe data/raw/*.parquet -- forces a re-download
```

`make run` is incremental: it re-runs the backtest only if the config, the package source,
or the cached prices changed. To force a re-run, `make clean` first.

For a single test, bypass make: `uv run pytest tests/test_engine.py -k look_ahead`.

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
- **A uv venv has no `pip`.** So VS Code's "Install ipykernel" prompt — which shells out to
  `python -m pip install ipykernel` in the selected interpreter — can never succeed here,
  and the error it shows does not explain why. Anything a notebook kernel needs must be
  declared in `pyproject.toml`'s dev extras and installed with `uv sync --extra dev`.
  `ipykernel` and `nbconvert` are already there for exactly this reason; add new notebook
  dependencies the same way rather than reaching for pip.
- `make notebooks` executes every notebook and discards the result. It is the cheap check
  that a change to the package did not silently break a notebook — the outputs are thrown
  away so nothing lands in git.
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
- `config.py` — loads `configs/pairs.yaml` into a typed, validated config
  (pairs, date ranges, z-score window, entry/exit thresholds, cost params, IS/OOS split date).
- `study.py` — `run_pair`/`run_study`/`to_frame`: the chain above, assembled. Notebooks and
  scripts both go through this; neither reimplements it.

Orchestration is config-driven: `scripts/run_backtest.py` reads `configs/pairs.yaml` and
calls `study.run_study` for every configured pair, writing a metrics CSV plus a run
manifest to `reports/results/`
and figures to `reports/figures/`. `data/` and `reports/{figures,results}/`
are gitignored except `.gitkeep` — never commit market data or generated outputs.

## Non-negotiable methodological rules

These five principles drive every design decision and are explicitly tested, not just stated:

1. **No look-ahead bias.** Positions are lagged by one bar in the engine; hedge ratio and
   z-score parameters are estimated on the in-sample window only. Enforced by two guard
   tests, not by convention: `test_engine.py` (altering a future price must not change any
   past P&L value) and `test_study.py` (shocking out-of-sample prices must not move the
   fitted sizing hedge ratio, or any in-sample metric). Both fail if the discipline is
   removed — verified by reintroducing a full-sample fit and watching them go red.
2. **Survivorship bias** is acknowledged in the writeup, not engineered around.
3. **No data-snooping.** Pairs are pre-specified from economic reasoning, not mined by
   scanning combinations, and **every pair is reported whatever it did**. There is no tier
   field in the schema and no way to run a subset — enforced by
   `test_study.py::test_table_has_no_tier_column` and
   `test_config.py::test_pairs_load_as_one_flat_list`. Parameters are frozen a priori; the
   window sweep is a reported sensitivity finding, never a parameter search.
4. **Gross vs. net always reported side by side** — every result appears before and after
   transaction costs.
5. **In-sample vs. out-of-sample always separated** — parameters come from IS data only;
   OOS is evaluated once and reported as-is.

When extending this project: keep all logic in `src/pairs_teardown/` modules with type
hints and a matching test in `tests/`; keep notebooks thin (import from the package, call,
plot) — no real logic in notebooks. Tests use synthetic data with known answers (e.g. a
random walk vs. `2*b + stationary_noise` for a known-cointegrated fixture) — never live
downloads in tests.
