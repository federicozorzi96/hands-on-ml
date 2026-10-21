PYTHON ?= python
EXERCISE ?= 01_numpy_nn
NOTEBOOK ?=

.PHONY: install install-cpu numpy notebook-to-py py-to-notebook
install:
	$(PYTHON) -m pip install -r $(EXERCISE)/requirements.txt

install-cpu:
	$(PYTHON) -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
	$(PYTHON) -m pip install -r 02_pedestrian_detection/requirements.txt

numpy:
	$(PYTHON) 01_numpy_nn/reference_solution/train.py

notebook-to-py:
	@test -n "$(NOTEBOOK)" || (echo "Usage: make notebook-to-py NOTEBOOK=path/to/notebook.ipynb"; exit 1)
	jupytext --to py:percent "$(NOTEBOOK)"

py-to-notebook:
	@test -n "$(NOTEBOOK)" || (echo "Usage: make py-to-notebook NOTEBOOK=path/to/notebook.py"; exit 1)
	jupytext --to ipynb "$(NOTEBOOK)"
