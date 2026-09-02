.PHONY: setup data baseline train calibrate cost evaluate link batch api test demo clean

setup:      ; pip install -r requirements.txt
data:       ; python -m ml.data_prep
baseline:   ; python -m ml.train_baseline
train:      ; python -m ml.train_lightgbm
calibrate:  ; python -m ml.calibrate
cost:       ; python -m ml.cost_curve
evaluate:   ; python -m ml.evaluate
link:       ; python -m ml.link_disputes
batch:      ; python -m app.services.batch_runner
api:        ; uvicorn app.main:app --reload
test:       ; pytest -v
demo:       ; python demo/run_demo.py
clean:      ; rm -rf artifacts/*.pkl evaluation/*.json evaluation/charts/*.png \
	                 evaluation/preds/*.parquet evidence/audit_log.jsonl

# Full reproducible path — this is what a reviewer runs
all: data baseline train calibrate cost evaluate link batch test
