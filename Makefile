.PHONY: help install install-dev test lint fmt build-model demo fleet clean

help:
	@echo "Targets:"
	@echo "  install      install runtime dependencies"
	@echo "  install-dev  install runtime + dev (pytest, ruff)"
	@echo "  test         run the offline test suite"
	@echo "  lint         ruff check"
	@echo "  fmt          ruff format + import sort"
	@echo "  build-model  ollama create the 'novel' model from the base"
	@echo "  demo         run the live compounding demo (needs a model server)"
	@echo "  fleet        report which role models are live (--probe for latency)"
	@echo "  clean        remove caches"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

test:
	python -m pytest

lint:
	ruff check .

fmt:
	ruff check --select I --fix .
	ruff format .

build-model:
	./build_model.sh

demo:
	python examples/compounding_demo.py

fleet:
	python main.py fleet $(ARGS)

clean:
	rm -rf .pytest_cache **/__pycache__ .ruff_cache
