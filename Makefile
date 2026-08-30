.PHONY: setup data baseline train calibrate cost link api test demo clean

setup:      ; pip install -r requirements.txt
data:       ; python ml/data_prep.py
baseline:   ; python ml/train_baseline.py
train:      ; python ml/train_lightgbm.py
calibrate:  ; python ml/calibrate.py
cost:       ; python ml/cost_curve.py
link:       ; python ml/link_disputes.py
api:        ; uvicorn app.main:app --reload
test:       ; pytest -v
demo:       ; python demo/run_demo.py
clean:      ; rm -rf artifacts/*.pkl evaluation/*.json evaluation/charts/*.png

# Full reproducible path — this is what a reviewer runs
all: data baseline train calibrate cost link test
