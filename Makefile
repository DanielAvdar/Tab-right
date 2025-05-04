.PHONY: help
.PHONY: default
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
	uv run mypy tab_right --config-file pyproject.toml

doc:
	uv run sphinx-build -M html docs/source docs/build/

nb:
	uv sync --group nbs --frozen
	uv run python -c "import asyncio; asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())" && uv run jupyter nbconvert --execute --to notebook --inplace nbs/*.ipynb
