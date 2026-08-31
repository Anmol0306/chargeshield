.PHONY: setup data baseline train calibrate cost link api test demo clean

setup:      ; pip install -r requirements.txt
data:       ; python -m ml.data_prep
baseline:   ; python -m ml.train_baseline
train:      ; python -m ml.train_lightgbm
calibrate:  ; python -m ml.calibrate
cost:       ; python -m ml.cost_curve
link:       ; python -m ml.link_disputes
link:       ; python ml/link_disputes.py
api:        ; uvicorn app.main:app --reload
test:       ; pytest -v
demo:       ; python demo/run_demo.py
clean:      ; rm -rf artifacts/*.pkl evaluation/*.json evaluation/charts/*.png

# Full reproducible path — this is what a reviewer runs
all: data baseline train calibrate cost link test
