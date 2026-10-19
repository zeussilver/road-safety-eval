# One command: `make all` (creates .venv on first run, then download -> ingest -> ... -> report).
# Each step can also be run alone, e.g. `make evaluate`. Raw downloads are kept by `make clean`.
PY := .venv/bin/python
STEPS := download ingest network treatment snap evaluate yarra report

all: $(PY)
	@for s in $(STEPS); do echo "== $$s"; $(PY) -m rse.$$s || exit 1; done

$(PY):
	@if command -v uv >/dev/null; then uv venv -p 3.12 .venv && VIRTUAL_ENV=.venv uv pip install -r requirements.txt -e .; \
	else python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt -e .; fi

$(STEPS): $(PY)
	$(PY) -m rse.$@

test: $(PY)
	$(PY) -m pytest -q

clean:
	rm -rf data/interim data/processed outputs/metrics.json outputs/tables outputs/figures outputs/map.html

.PHONY: all test clean $(STEPS)
