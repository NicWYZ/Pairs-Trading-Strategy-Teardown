# Pairs-trading teardown — task runner.
#
# The file targets encode the pipeline's dependency graph, so Make re-runs only
# what is stale: `make run` re-executes the backtest if the config, the source,
# or the cached prices changed, and skips it otherwise.

# `uv run` resolves the lockfile's interpreter. A bare `python` picks up
# whatever is on PATH (often conda's), which cannot import pairs_teardown.
PYTHON  := uv run python
CONFIG  := configs/pairs.yaml
METRICS := reports/results/metrics.csv
SRC     := $(shell find src/pairs_teardown -name '*.py')
PRICES  := $(shell $(PYTHON) scripts/cache_path.py --config $(CONFIG))

.PHONY: help install test lint typecheck run run-official clean clean-data

help:
	@echo "install       editable install with dev extras"
	@echo "test          run the test suite"
	@echo "lint          ruff check"
	@echo "typecheck     mypy"
	@echo "run           full study (all pairs) -> reports/"
	@echo "run-official  only the three pre-specified pairs"
	@echo "clean         remove generated reports (keeps cached data)"
	@echo "clean-data    also remove the cached price data"
	@echo ""
	@echo "prices cache: $(PRICES)"

# `uv sync`, never `uv pip install -e` -- the latter can target a different
# Python than uv sync uses and split the venv (see CLAUDE.md).
install:
	uv sync --extra dev

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check src tests scripts

typecheck:
	$(PYTHON) -m mypy src

# --- pipeline -------------------------------------------------------------- #
# `touch` is required, not cosmetic: load_or_download returns the cached parquet
# untouched on a hit, so without it the target never becomes newer than its
# prerequisites and this rule re-fires on every single `make run`.
$(PRICES): scripts/download_data.py $(CONFIG)
	$(PYTHON) scripts/download_data.py --config $(CONFIG)
	@touch $@

$(METRICS): $(PRICES) $(CONFIG) $(SRC) scripts/run_backtest.py
	$(PYTHON) scripts/run_backtest.py --config $(CONFIG)

run: $(METRICS)

run-official:
	$(PYTHON) scripts/run_backtest.py --config $(CONFIG) --official-only

clean:
	rm -rf reports/results/* reports/figures/*

clean-data:
	rm -rf data/raw/*.parquet