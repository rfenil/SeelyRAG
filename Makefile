# Seeley Installer RAG — task runner.
#
# Windows note: this needs GNU make. If `make` is missing, install it with
#   winget install ezwinports.make
# or run the recipe bodies directly; each is one or two plain commands.

ifeq ($(OS),Windows_NT)
    VENV_BIN := .venv/Scripts
    PY_BOOT  := python
else
    VENV_BIN := .venv/bin
    PY_BOOT  := python3
endif

PY   := $(VENV_BIN)/python
PIP  := $(PY) -m pip
SRC  := src/seeley_rag

.DEFAULT_GOAL := help
.PHONY: help init install lint format test coverage robots triage acquire parse chunk embed search ask serve rerank-ab novice clean

help:  ## Show this help
	@echo "Targets:"
	@echo "  init      Create .venv, install deps (editable), create data/ dirs"
	@echo "  install   Reinstall deps into an existing .venv"
	@echo "  lint      black --check, isort --check, flake8"
	@echo "  format    black + isort, in place"
	@echo "  test      pytest"
	@echo "  coverage  pytest with coverage on acquire/ (fails under 80%)"
	@echo "  robots    Stage 0 gate: is the portal crawlable?"
	@echo "  triage    Stage 0: PDF corpus triage report"
	@echo "  acquire   Stage 1: crawl the pilot categories"
	@echo "  parse     Stage 2: PDFs and articles -> pages.jsonl"
	@echo "  chunk     Stage 3: pages -> chunks.jsonl + codes.jsonl"
	@echo "  embed     Stage 4: chunks -> LanceDB vector + FTS index"
	@echo "  search    Stage 5: run the retrieval cascade"
	@echo "  ask       Stage 6: ask a question, get a cited answer"
	@echo "  serve     Stage 7: run the REST API"
	@echo "  clean     Remove derived data stages and caches (NEVER data/00_raw)"

init:  ## venv + install + data dirs
	$(PY_BOOT) -m venv .venv
	$(PIP) install --upgrade pip
	$(MAKE) install
	$(PY) -c "from seeley_rag.paths import ensure_dirs; ensure_dirs()"
	@echo "Ready. Next: make robots"

install:  ## Install runtime + dev deps and the package itself, editable
	$(PIP) install -r requirements.txt -r requirements-dev.txt
	$(PIP) install -e .

lint:  ## Check formatting and style without changing anything
	$(PY) -m black --check --diff $(SRC) tests scripts
	$(PY) -m isort --check-only --diff $(SRC) tests scripts
	$(PY) -m flake8 $(SRC) tests scripts

format:  ## Rewrite files to satisfy black + isort
	$(PY) -m isort $(SRC) tests scripts
	$(PY) -m black $(SRC) tests scripts

test:  ## Run the test suite
	$(PY) -m pytest

coverage:  ## Test suite with a hard 80%% floor on the acquire package
	$(PY) -m pytest --cov=seeley_rag.acquire --cov-report=term-missing --cov-fail-under=80

robots:  ## Stage 0 gate. If this fails, no crawl may proceed.
	$(PY) scripts/00_check_robots.py

triage:  ## Stage 0. Usage: make triage PDFS="data/00_raw/pdf/*.pdf"
	$(PY) scripts/01_triage.py $(PDFS)

acquire:  ## Stage 1. Usage: make acquire ARGS="--limit 5 --dry-run"
	$(PY) scripts/02_acquire.py $(ARGS)

parse:  ## Stage 2. Usage: make parse ARGS="--limit 5"
	$(PY) scripts/03_parse.py $(ARGS)

chunk:  ## Stage 3. Usage: make chunk ARGS="--stats"
	$(PY) scripts/04_index.py $(ARGS)

embed:  ## Stage 4. Usage: make embed ARGS="--smoke"
	$(PY) scripts/05_embed.py $(ARGS)

search:  ## Stage 5. Usage: make search ARGS="--demo"
	$(PY) scripts/06_search.py $(ARGS)

ask:  ## Stage 6. Usage: make ask ARGS="--demo"
	$(PY) scripts/07_ask.py $(ARGS)

novice:  ## Stage 5. The queries trade workers actually type. Usage: make novice ARGS="--flags-only"
	$(PY) scripts/10_novice_queries.py $(ARGS)

rerank-ab:  ## Stage 5 B-4. Compare rerank backends. Usage: make rerank-ab ARGS="--plan"
	$(PY) scripts/09_rerank_ab.py $(ARGS)

serve:  ## Stage 7. Usage: make serve ARGS="--reload --port 8080"
	$(PY) scripts/08_serve.py $(ARGS)

clean:  ## Remove derived stages. data/00_raw is immutable and never touched.
	$(PY) -c "from seeley_rag.paths import clean_derived; clean_derived()"
	@echo "Removed derived stages. data/00_raw left intact by design."
