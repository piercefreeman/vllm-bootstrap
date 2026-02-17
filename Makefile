.PHONY: lint lint-verify test validate generate-proto

lint:
	uv run --no-project --with ruff ruff check . --fix
	uv run --no-project --with ruff ruff format .

lint-verify:
	uv run --no-project --with ruff ruff format --check .
	uv run --no-project --with ruff ruff check .

test:
	PYTHONPATH=. uv run --no-project --with pytest --with fastapi --with jinja2 --with pydantic-settings --with grpcio --with grpcio-tools --with protobuf pytest -q

validate: lint-verify test

SERVER_OUT = vllm_bootstrap/generated
CLIENT_OUT = vllm-bootstrap-client/vllm_bootstrap_client/generated

generate-proto:
	mkdir -p $(SERVER_OUT) $(CLIENT_OUT)
	uv run python -m grpc_tools.protoc \
		-I proto \
		--python_out=$(SERVER_OUT) \
		--pyi_out=$(SERVER_OUT) \
		--grpc_python_out=$(SERVER_OUT) \
		proto/inference.proto
	uv run python -m grpc_tools.protoc \
		-I proto \
		--python_out=$(CLIENT_OUT) \
		--pyi_out=$(CLIENT_OUT) \
		--grpc_python_out=$(CLIENT_OUT) \
		proto/inference.proto
	touch $(SERVER_OUT)/__init__.py $(CLIENT_OUT)/__init__.py
