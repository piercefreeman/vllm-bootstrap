.PHONY: lint lint-verify test test-vllm-compat validate generate-proto generate-client build-test-image test-integration

lint:
	uv run --no-project --with ruff ruff check . --fix
	uv run --no-project --with ruff ruff format .

lint-verify:
	uv run --no-project --with ruff ruff format --check .
	uv run --no-project --with ruff ruff check .

test:
	PYTHONPATH=. uv run --no-project --with pytest --with fastapi --with jinja2 --with pydantic-settings --with httpx --with grpcio --with grpcio-tools --with protobuf pytest -q

test-vllm-compat:
	PYTHONPATH=. uv run --no-project --with pytest --with vllm pytest -q vllm_bootstrap/__tests__/test_vllm_compat.py

validate: lint-verify test

SERVER_OUT = vllm_bootstrap/generated
CLIENT_OUT = vllm-bootstrap-client/vllm_bootstrap_client/generated
CLIENT_MODELS = vllm-bootstrap-client/vllm_bootstrap_client/models.py

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
	# Fix absolute imports in generated gRPC files to use relative imports
	sed -i.bak 's/^import inference_pb2 as/from . import inference_pb2 as/' $(SERVER_OUT)/inference_pb2_grpc.py $(CLIENT_OUT)/inference_pb2_grpc.py
	rm -f $(SERVER_OUT)/inference_pb2_grpc.py.bak $(CLIENT_OUT)/inference_pb2_grpc.py.bak

generate-client-models:
	PYTHONPATH=. uv run python -c \
		"import json; from vllm_bootstrap.api import app; print(json.dumps(app.openapi()))" \
		> /tmp/vllm-bootstrap-openapi.json
	uv run datamodel-codegen \
		--input /tmp/vllm-bootstrap-openapi.json \
		--input-file-type openapi \
		--output $(CLIENT_MODELS) \
		--output-model-type pydantic_v2.BaseModel \
		--target-python-version 3.11

generate-client: generate-proto generate-client-models

build-test-image:
	docker build -f Dockerfile.test -t vllm-bootstrap-test .

test-integration: build-test-image
	docker run --rm \
		-v $(HOME)/.cache/huggingface:/root/.cache/huggingface \
		vllm-bootstrap-test
