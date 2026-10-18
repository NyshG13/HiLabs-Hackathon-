# TermMap-X+ (tag xplus-sf)
#   make install                               pip install -r requirements.txt
#   make setup                                 download trained weights (Hugging Face) + verify + download the 2 public Qwen models
#   make predict INPUT=my_terms.csv OUTPUT=predictions.csv
#   make demo                                  data/splits/test_input.csv -> submission/demo_predictions.csv
#   make test                                  input-guard test (extra columns -> byte-identical output)
PY ?= python


INPUT ?= data/splits/test_input.csv
OUTPUT ?= submission/predictions.csv

.PHONY: install setup verify predict demo test

install:
	$(PY) -m pip install -r requirements.txt

setup:
	$(PY) setup_model.py

verify:
	$(PY) setup_model.py --verify

predict:
	$(PY) predict.py --input $(INPUT) --output $(OUTPUT)

demo:
	$(PY) predict.py --input data/splits/test_input.csv --output submission/demo_predictions.csv

test:
	$(PY) tests/test_sf_guard.py
