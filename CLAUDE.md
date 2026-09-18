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
  `build_spread`, `adf_pvalue`, `engle_granger_pvalue`, `johansen_trace` (system-based,
  symmetric in the legs; returns the trace statistic against tabulated critical values),
  `half_life` (discrete OU / AR(1) fit with a delta-method SE; `inf` when ρ ≥ 1), and
  `analyze_pair` (bundles everything into a `CointegrationResult`). The `adf_pvalue`
  docstring records why an ordinary ADF on an OLS-fitted spread is too liberal
  (Engle–Granger 1987 / Phillips–Ouliaris 1990) — it is a descriptive diagnostic, and
  `engle_granger_pvalue` is the test to cite.
- `src/pairs_teardown/stats/inference.py` — uncertainty and multiplicity: `sharpe_se`
  (Lo 2002, i.i.d. delta method), `sharpe_null_sd`, `sharpe_pvalue`,
  `stationary_bootstrap_indices` + `bootstrap_ci` (Politis–Romano, percentile interval;
  accepts precomputed indices so several statistics share one set of resamples),
  `holm_adjust` (step-down FWER, no independence assumption), and `expected_max_sharpe`
  (Bailey & López de Prado 2014 — the noise floor for "the best of N pairs"). Every function
  is tested against a closed form or a Monte Carlo experiment (`tests/test_inference.py`).
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
  and `summary`. Every metrics block also carries `sharpe_se` (from `stats/inference.py`),
  so no Sharpe leaves the package without its standard error. All pure functions of a
  return/position series, so every one is closed-form testable. Conventions are fixed and documented in the module docstring: simple daily
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
  dataclasses and *validates* it. `InferenceConfig` (`n_boot`, `mean_block`, `ci_level`,
  `seed`) fixes the bootstrap scheme in the committed config so it cannot be tuned after
  seeing which intervals exclude zero. Validation is not decorative: it rejects `entry <= exit`
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
  across the split; the retired exploratory notebook fit on the full sample, which leaks. This
  lives in the package rather than in `scripts/` for two reasons: notebooks import the
  chain instead of reimplementing it (reimplementing it inline is exactly how notebook 03
  acquired its leak), and the discipline is testable. `run_study` has no subset argument
  and `to_frame` emits no tier column — an API that could quietly omit a pair would be a way
  to launder a bad result out of the study. `inference_frame` produces one row per
  (pair, period, basis) with Lo SE, normal-approximation p, stationary-bootstrap intervals
  on Sharpe and total return, and Holm-adjusted p (family = the pairs within one
  period × basis). One index matrix per (pair, period) is shared by gross and net so their
  intervals differ only through costs. `n_trades` is counted **per period** from the full
  position series before slicing (it was once the full-sample count copied into every row).
  `run_pair_walk_forward` / `run_study_walk_forward` are Arm B: identical to `run_pair`
  before the split, then calendar-year segments with the sizing hedge re-fit on data strictly
  before each segment and the window from `walk_forward_window`; z-scores are spliced and
  positions come from one `target_positions` pass; a non-positive refit hedge means a flat
  segment. `segments_frame` records every refit. The engine now charges `|position|·|Δg|` on
  a re-hedge (zero for Arm A) and `validate_step_hedge_ratio` replaces the rolling-noise guard
  on that path.
- `scripts/run_backtest.py` — the reproduction entry point, now only argparse + file
  writing over `study.py`. Writes `reports/results/metrics.csv` (long-form: pair x period
  x gross/net), `inference.csv` (same keys, uncertainty columns), a `run_manifest.json`
  recording parameters + hedge ratios + bootstrap scheme + timestamp, and
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
- `notebooks/01_data_exploration.ipynb` — coverage, price levels, return correlation and
  trading-day alignment for all 10 pairs. Config-driven, not a hardcoded ticker list.
- `notebooks/02_cointegration_analysis.ipynb` — **the key diagnostic.** Tests cointegration
  separately in-sample and out-of-sample: 3/10 pass in-sample, 3/10 out-of-sample, but only
  **1/10 (MA/V) in both**. The property the whole method assumes does not persist. Also shows
  UPS/FDX's hedge ratio changing *sign* across the split (+0.84 -> -0.35), which is why it is
  the worst performer. §2b runs Johansen alongside Engle–Granger: they agree on one pair
  in-sample (MA/V), and EG misses SPY/VOO (p = 0.09) where Johansen's trace is 8x its
  critical value — a count is a property of the test. §3b estimates OU half-lives with SEs:
  5 of 10 in-sample half-lives exceed the 60-day window, and for XOM/CVX and UPS/FDX φ is
  within 2 SE of zero — no reversion detectable (Dickey–Fuller bias means a half-life point
  estimate is always finite, which is why the SE is mandatory). Selecting pairs or windows on any of this would be data-snooping, so nothing
  here filters the universe — they are diagnostics that explain the backtest.
- `notebooks/03_backtest_results.ipynb` — the authoritative results. Reads `metrics.csv` +
  `run_manifest.json` (so it cannot disagree with the pipeline), tabulates gross-vs-net and
  IS-vs-OOS, reconciles the cost drag against `turnover x (1+|g|) x bps`, reports the
  cross-sectional dispersion, regenerates figures through `study.run_study`. §5b reads
  `inference.csv`: the SE on any OOS Sharpe is 0.58; UNP/CSX's raw p = 0.039 becomes 0.35
  after Holm; the only Holm-significant result is SPY/VOO's *negative* Sharpe (a zero-
  variance spread measures the cost of trading it with no noise); and the best observed
  Sharpe (1.20) is half an SE above the expected best of ten null strategies (0.91).
  **Results only — no sweeps**, deliberately, so results and robustness cannot be confused.
- `notebooks/04_sensitivity_analysis.ipynb` — every frozen parameter swept one at a time
  through the same `run_pair`: window, entry/exit bands, cost (with per-pair breakeven),
  split date, rolling-vs-static signal hedge. Writes `reports/results/sensitivity.csv`,
  which notebook 05 **reads rather than transcribes** — same one-source discipline as
  metrics.csv. Nothing here may feed back into the config: adopting a swept value would turn
  an out-of-sample result into an in-sample one.
- `notebooks/06_holdout.ipynb` — reads `reports/results/` and `reports/results_holdout/`
  only. §1 data-integrity check (new download reproduces the manifest's hedge ratios to
  1e-7), §2 Arm A replication, §3 persistence (the pre-registered test of the thesis), §4
  Arm B vs Arm A on both windows with the segment table, §5 conclusion. Fails with a clear
  message if the holdout has not been run.
- `notebooks/05_writeup.ipynb` — the narrative, organized around the five principles. §2.3
  documents the removal of the pair tiers as an error the study made and corrected; §2.3b
  carries the cointegration-persistence finding plus the Johansen and half-life refinements;
  §2.5b carries the per-pair inference and the expected-max-Sharpe argument; §2.6 cites
  notebook 04. The conclusion lists five converging lines, the fifth being selection.
- `.pre-commit-config.yaml` — ruff, ruff-format, nbstripout, plus whitespace/large-file
  checks. Fast hooks only; pytest and mypy stay in CI, because a slow commit hook gets
  bypassed and a bypassed hook enforces nothing. **nbstripout means committed notebooks
  carry no outputs** — run them locally to see figures.
- `.github/workflows/ci.yml` — uv sync, ruff check, ruff format --check, mypy, pytest. The
  suite is fully synthetic, so CI needs no market data and no network.
- `README.md` — the public-facing summary: the finding, how to run it, and where the five
  rules are enforced.

137 tests pass. `charts.py` has no direct tests; the logic that used to sit in
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

**The pre-registered holdout (Sept 2026) is the third half.** `PREREGISTRATION.md` was
committed (e16c645) before any price after 2024-12-31 was downloaded; `make run-holdout`
then ran both arms once on 2025-01 → 2026-08. Arm A (frozen strategy): 2/10 profitable,
mean −3.5%, nothing significant, best Sharpe 1.24 vs expected best-of-10 of 1.23. 0 of the
four 2022–24 winners repeated (ρ = −0.56). Arm B (walk-forward: annual refit on an anchored
expanding window, window = clip(round(2·HL), 20, 250)): −7.8pp vs Arm A on 2022–24,
+2.9pp [−10, +14] on the holdout, per-pair moves up to 46pp — reshuffling, not improvement.
**Do not add a third arm, re-tune Arm B's rule, or re-run the holdout with changes**; a
change after the download is reported as a deviation, not adopted. Outputs live in
`reports/results_holdout/` (gitignored like the rest); notebook 06 reads them.

**The inference layer is the second half of that finding.** Every OOS Sharpe has an SE of
0.58; no positive pair survives Holm; the best pair's Sharpe is what the best of ten null
strategies is expected to produce. `inference.csv` is written by the pipeline (not computed
in notebooks) and its scheme is fixed in the config — do not move the bootstrap into a
notebook where the block length could be tuned after seeing the intervals.

**Notebook order is `01 data -> 02 cointegration -> 03 results -> 04 sensitivity -> 05
writeup`.** Sensitivity comes *after* results because it is a check on them, not an input.
Notebook 05 depends on notebook 04 having run (it reads `sensitivity.csv`); `make notebooks`
executes them in order, so this only bites if 05 is opened standalone, and it fails with a
message saying so rather than silently.

Remaining gaps: `plotting/charts.py` still has no tests. The half-life SE is homoskedastic
OLS (no HAC); the bootstrap block length is fixed a priori rather than Politis–White
selected; Lo's SE is i.i.d. — the bootstrap is the check on it. All three are documented in
the writeup's limitations rather than fixed, deliberately, to keep the project the size of
a first project.

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
  z-score. Responsiveness to drift is valuable here. Selected by `signal.signal_hedge`, which
  `run_pair` genuinely honours — it was once inert (validated, written to the run manifest,
  and ignored), so `test_study.py::test_signal_hedge_setting_changes_the_spread` now guards
  it. Worth knowing: notebook 04 §5 finds the rolling-vs-static choice moves the
  cross-sectional mean by ~1pp while moving *individual pairs* by up to 48pp, which is a
  noise signature rather than an edge.
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
