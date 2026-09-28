PY ?= python3

.PHONY: install test lint demo field-d clean

install:            ## editable install with LAS/LAZ support and dev tools
	$(PY) -m pip install -e ".[las,dev]"

test:               ## unit + end-to-end tests on synthetic data (~1 min, CPU)
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check junglelook tests

demo:               ## synthetic end-to-end demo, figures in work/demo/
	junglelook demo

field-d:            ## full pipeline on the Field-D LiDAR ROI (needs dataset/, see README)
	junglelook run -c configs/field_d.yaml

clean:
	rm -rf work .pytest_cache .ruff_cache *.egg-info
