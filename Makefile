PYTHON ?= python
EXERCISE ?= 01_numpy_nn

.PHONY: install install-cpu numpy
install:
	$(PYTHON) -m pip install -r $(EXERCISE)/requirements.txt

install-cpu:
	$(PYTHON) -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
	$(PYTHON) -m pip install -r 02_pedestrian_detection/requirements.txt

numpy:
	$(PYTHON) 01_numpy_nn/reference_solution/train.py
