# Pre-registration: holdout replication and walk-forward variant

**Status.** Written and committed before any price after 2024-12-31 was downloaded. The
commit that introduces this file is the timestamp. Nothing below changes after the
download; if something must change (a data error, a bug), the change is made in a later
commit and reported as a deviation.

## Why

Study 1 (`notebooks/01`–`05`) evaluated ten pre-specified pairs on 2022–2024 with parameters
frozen a priori, and concluded that pair-to-pair dispersion dwarfs the mean effect. It also
diagnosed two mechanical problems: the hedge ratio and the cointegration relationship do not
persist over a three-year horizon, and the pre-registered 60-day window is shorter than the
estimated half-life of reversion for half the pairs. Both diagnoses were made on the judged
data, so any fix adopted now and re-scored on 2022–2024 would be fitting to the test set.

This study does two things with data nobody in the project has looked at:

1. **Replicates Study 1** on 2025-01-01 → 2026-08-31 with nothing changed (Arm A).
2. **Tests the textbook response to the diagnosis** — re-estimate what goes stale, on a
   fixed schedule, with a fixed rule — on the same unseen window (Arm B). Arm B is *also* run
   on 2022–2024, where it is reported as a variant of Study 1 alongside the frozen result,
   never in its place.

## Data

- Prices: yfinance adjusted close, all 20 tickers fetched together, 2015-01-01 → 2026-08-31.
- Study 1 config is untouched. `configs/pairs_holdout.yaml` differs in exactly three
  settings: `data.end`, `split.in_sample_end = 2024-12-31`, and the output directories.
- **Integrity check.** Back-adjustment for dividends paid after 2024-12-31 multiplies every
  earlier price by a constant, which shifts log prices by a constant and leaves OLS slopes
  and returns unchanged. The sizing hedge ratios fit on 2015–2021 from the new download must
  therefore reproduce `reports/results/run_manifest.json` to numerical precision. A
  mismatch means Yahoo revised history, and is reported, not silently accepted.

## Arms

**Arm A — frozen replication.** Window 60, entry 2.0, exit 0.5, rolling signal hedge,
static sizing hedge fit by OLS on all data before the evaluation window. Identical code path
to Study 1 (`study.run_pair`).

**Arm B — walk-forward re-estimation** (`study.run_pair_walk_forward`). Before the split,
identical to Arm A. From the split onward, the evaluation window is cut at the first trading
day of each calendar year. At the start of each segment, using only data strictly before it:

- the sizing hedge ratio is re-fit by OLS on the anchored expanding window
  (2015-01-01 → the day before the segment);
- the signal window is set from the half-life of that fitted spread:
  `window = clip(round(2 × HL), 20, 250)`, with an infinite or undetectable half-life
  mapped to 250. Rationale: a trailing mean should span at least two half-lives (75% of a
  deviation decays); 20 days is the floor for a usable rolling standard deviation; 250 is
  one trading year;
- if the refit hedge ratio is not positive, no hedge exists and the pair is flat for the
  segment;
- the re-hedge is executed through the engine's one-bar lag and charged as a trade in the
  B leg (`|position| × |Δg|` at the cost rate), a cost Study 1's engine did not need.

Entry/exit bands and the cost model are unchanged. Both changes are bundled deliberately:
they are one idea — nothing estimated stays stale — and one arm keeps the comparison to
Arm A attributable to that idea.

Refit dates in the 2022–2024 variant: 2022-01-03, 2023-01-03, 2024-01-02. In the holdout:
2025-01-02, 2026-01-02. (First trading days; the exact dates are whatever the calendar
gives and are recorded in `walk_forward_segments.csv`.)

**Not included.** Volatility-scaled position sizing was considered and excluded: it needs
two more a-priori parameters and a rebalancing rule, changes what total return means, and
would make the Arm B comparison unattributable.

## Endpoints (all ten pairs, both arms, gross and net; net is primary)

1. Cross-sectional mean net Sharpe and net total return on the holdout, with Lo standard
   errors, stationary-bootstrap intervals and Holm-adjusted p-values across pairs, exactly
   as in `inference.csv`. The expected maximum of ten null Sharpes for the holdout length is
   reported as the noise floor.
2. **Persistence (Arm A):** Spearman ρ between each pair's 2022–2024 net return and its
   2025–2026 net return. Study 1's thesis predicts ρ ≈ 0. Four pairs were profitable in
   2022–2024 (KO/PEP, HD/LOW, UNP/CSX, DUK/SO); the thesis predicts about half of them
   repeat. If all four repeat and the losers stay losers, that is evidence against the
   thesis and will be written as such.
3. **Arm B − Arm A**, per pair and as a cross-sectional mean, on both the 2022–2024 window
   and the holdout, with a bootstrap over pairs for the mean difference.
4. Segment table: hedge ratio, half-life, window and traded flag per refit.

## Power, stated in advance

The holdout is ~420 trading days. The standard error of an annualised Sharpe with no edge
is √(252/420) ≈ 0.77. This study can detect a large edge or a replicated ranking; it cannot
resolve a mean near zero, and no result here will be described as "confirming" a small
effect in either direction.

## Rules

- All ten pairs, including SPY/VOO. No pair is dropped, re-labelled or tiered.
- Both arms, on both windows, whatever they say. No third arm is added after the download.
- The holdout is scored once. `make run-holdout` is the command; the outputs go to
  `reports/results_holdout/`.
- Interpretation lives in `notebooks/06_holdout.ipynb`, which reads the CSVs and computes
  nothing that is not a function of them.
