.PHONY: help
.PHONY: default
.PHONY: check-all
default: install


install:
	uv sync --all-extras --all-groups --frozen
	uv tool install pre-commit --with pre-commit-uv --force-reinstall
	uv run pre-commit install

install-docs:
	uv sync --group docs --frozen --no-group dev

update:
	uv lock
	uvx pre-commit autoupdate
	$(MAKE) install

test:
	uv run pytest

check:
	uv run  pre-commit run --all-files

coverage:
	uv run pytest --cov=tab_right --cov-report=xml

cov:
	uv run pytest --cov=tab_right --cov-report=term-missing

mypy:
	uv run mypy tab_right tests/base_architecture --config-file pyproject.toml

# Add doctests target to specifically run doctest validation
doctest: install-docs doc

# Update doc target to run doctests as part of documentation build
doc:
	uv run sphinx-build -M doctest docs/source docs/build/ -W --keep-going
	uv run sphinx-build -M html docs/source docs/build/ -W --keep-going

# Optional target that builds docs but ignores warnings
doc-ignore-warnings:
	uv run sphinx-build -M html docs/source docs/build/

# Run all checks in sequence: tests, code quality, type checking, and documentation
check-all:
	$(MAKE) check
	$(MAKE) test
	$(MAKE) mypy
	$(MAKE) doc
