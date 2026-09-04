# Use the project venv when present, else whatever python3 is on PATH.
# Bare `python`/`uvicorn` only work with the venv activated, which made
# `make api` fail for anyone who had not sourced it -- including a fresh clone.
PYTHON ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)

.PHONY: setup data baseline train calibrate cost evaluate link batch llm-check api test demo clean

setup:      ; $(PYTHON) -m pip install -r requirements.txt
data:       ; $(PYTHON) -m ml.data_prep
baseline:   ; $(PYTHON) -m ml.train_baseline
train:      ; $(PYTHON) -m ml.train_lightgbm
calibrate:  ; $(PYTHON) -m ml.calibrate
cost:       ; $(PYTHON) -m ml.cost_curve
evaluate:   ; $(PYTHON) -m ml.evaluate
link:       ; $(PYTHON) -m ml.link_disputes
batch:      ; $(PYTHON) -m app.services.batch_runner
llm-check:  ; $(PYTHON) -m scripts.check_llm
api:        ; $(PYTHON) -m uvicorn app.main:app --reload
test:       ; $(PYTHON) -m pytest -v
demo:       ; $(PYTHON) -m demo.run_demo $(ARGS)
clean:      ; rm -rf artifacts/*.pkl evaluation/*.json evaluation/charts/*.png \
	                 evaluation/preds/*.parquet evidence/audit_log.jsonl

# Full reproducible path — this is what a reviewer runs
all: data baseline train calibrate cost evaluate link batch test
