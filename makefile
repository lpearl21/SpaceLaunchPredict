# Makefile for Rocket Launch ML Project

.PHONY: help setup clean data train predict viz all test lint format

help:
	@echo "Available commands:"
	@echo "  make setup    - Install dependencies and set up environment"
	@echo "  make data     - Process raw data and add weather features"
	@echo "  make train    - Train ML models"
	@echo "  make predict  - Run prediction demo"
	@echo "  make viz      - Create visualizations"
	@echo "  make all      - Run complete pipeline"
	@echo "  make test     - Run unit tests"
	@echo "  make lint     - Run code linting"
	@echo "  make clean    - Remove generated files"

setup:
	python -m venv venv
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && pip install -r requirements.txt
	@echo "✅ Setup complete. Activate with: source venv/bin/activate"

data:
	python main.py --mode process

train:
	python main.py --mode train

predict:
	python main.py --mode predict

viz:
	python main.py --mode viz

all:
	python main.py --mode full

test:
	pytest tests/ -v --cov=src

lint:
	flake8 src/ --max-line-length=100
	black --check src/

format:
	black src/
	isort src/

clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache
	rm -rf *.egg-info
	rm -f data/processed/*.csv
	rm -f models/*.pkl
	rm -f reports/figures/*.png
	@echo "✅ Cleanup complete"
