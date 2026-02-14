.PHONY: lint lint-verify test validate

lint:
	uv run --no-project --with ruff ruff check . --fix
	uv run --no-project --with ruff ruff format .

lint-verify:
	uv run --no-project --with ruff ruff format --check .
	uv run --no-project --with ruff ruff check .

test:
	PYTHONPATH=. uv run --no-project --with pytest --with fastapi --with jinja2 --with pydantic-settings --with httpx pytest -q

validate: lint-verify test
