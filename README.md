# vllm-bootstrap

FastAPI control plane for launching and managing vLLM servers with explicit GPU ownership.

## API

- `POST /launch`
  - Launches `vllm.entrypoints.openai.api_server`.
  - Defaults to a new model launch on all visible GPUs.
  - Returns `409` if requested/default GPUs are already owned by an active launch.
- `GET /status/{launch_id}`
  - Returns launch state (`bootstrapping`, `ready`, `stopping`, `stopped`, `failed`) and metadata.
- `GET /logs/{launch_id}?offset=<int>`
  - Returns log chunk and next offset for incremental log streaming.
- `POST /stop/{launch_id}`
  - Stops the managed launch and releases its GPU ownership.

## Run locally

```bash
uv run vllm-bootstrap
```

Example launch request:

```bash
curl -X POST http://localhost:8000/launch \
  -H "content-type: application/json" \
  -d '{"model":"meta-llama/Llama-3.1-8B-Instruct"}'
```

Tail logs:

```bash
curl "http://localhost:8000/logs/<launch_id>?offset=0"
```

## Docker

Build:

```bash
docker build -t vllm-bootstrap:local .
```

Run:

```bash
docker run --rm -p 8000:8000 vllm-bootstrap:local
```

## Key environment variables

- `VLLM_DEFAULT_MODEL` default model for `/launch` when no model is provided.
- `VLLM_GPU_INDICES` comma-separated GPU ids to control visible GPUs for allocation.
- `VLLM_LAUNCH_PORT_START` / `VLLM_LAUNCH_PORT_END` port range for vLLM child processes.
- `VLLM_BOOTSTRAP_LOG_DIR` log directory for child process output.

## Release workflow

Pushing a Git tag like `v0.1.0` triggers `.github/workflows/publish-image.yml`, which builds the Docker image and pushes it to:

- `ghcr.io/<owner>/<repo>:v0.1.0`
- `ghcr.io/<owner>/<repo>:latest`

## Tests

```bash
PYTHONPATH=. uv run --no-project --with pytest pytest -q
```

## Local validation

```bash
make lint-verify
make test
```

Or run both:

```bash
make validate
```
