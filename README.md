# vllm-bootstrap

Self contained docker image to treat a remote server (Runpod, GCP, AWS) as a generic endpoint for speeding up batch inference compute. Deploy the docker image to your remote server and dynamically update the model it's serving, get vllm bootstrap status, issue jobs, etc.

Internally we implement this logic through a FastAPI control plane for managing in-process vLLM instances with explicit GPU ownership, and a gRPC server for inference (embeddings and completions).

If you're interested in a step by step guide for Runpod, check [this out](./docs/RUNPOD.md).

## Hosting vllm

Most often you'll want to point your remote box to our image directly. Check what CUDA version it supports on the host box (this is passed through
to the container). For a device with 12.4:

```
docker pull ghcr.io/piercefreeman/vllm-bootstrap:cuda12.4-latest
```

Images are based on the [official PyTorch Docker images](https://hub.docker.com/r/pytorch/pytorch) and use the latest available PyTorch release for each CUDA version. Because PyTorch eventually stops publishing new images for older CUDA releases, newer CUDA versions will receive more recent PyTorch (and by extension vllm) versions.

If you want to build locally:

```bash
docker build \
  --build-arg BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime \
  -t vllm-bootstrap:cuda12.4-local .
```

Run:

```bash
docker run --rm -p 8000:8000 -p 8001:8001 vllm-bootstrap:cuda12.4-local
```

Port 8000 serves the REST control plane, and port 8001 is the gRPC inference service.

## Key environment variables

- `VLLM_GRPC_PORT` gRPC server port (default `8001`).
- `VLLM_BOOTSTRAP_LOG_DIR` log directory.
- `VLLM_ACCESS_KEY` optional shared key that protects REST routes.
  - `Authorization: Bearer <key>` is accepted.
  - If no auth header is sent, the server returns `401` with a `WWW-Authenticate: Basic` challenge.
  - HTTP Basic auth is accepted with any username and `<key>` as password.

GPU ownership and default allocation are auto-detected from host hardware via `nvidia-smi`.

## Client library

Install the client:

```bash
pip install vllm-bootstrap-client
```

`VLLMManager` provides an async, Pydantic-first interface for both management (REST/httpx) and inference (gRPC). It derives the gRPC address automatically from the base URL (port + 1), or you can pass `grpc_address` explicitly.

```python
from vllm_bootstrap_client import VLLMManager

async with VLLMManager("http://your-server:8000") as manager:
    ...
```

### Embeddings

`manager.run()` handles launch, wait-for-ready, and teardown automatically:

```python
async with manager.run("BAAI/bge-base-en-v1.5", task="embed") as launch:
    vectors = await manager.embed(launch.launch_id, ["Hello world", "Another sentence"])
    # vectors: list[list[float]]
# model is stopped when the block exits
```

### Completions

```python
async with manager.run("meta-llama/Llama-3.1-8B-Instruct", task="generate") as launch:
    result = await manager.complete(
        launch.launch_id,
        prompt="Once upon a time",
        max_tokens=128,
        temperature=0.7,
    )
    print(result.text)
```

### Manual lifecycle

If you need more control, use `launch` / `wait_until_ready` / `stop` directly:

```python
launch = await manager.launch("meta-llama/Llama-3.1-8B-Instruct", task="generate")
await manager.wait_until_ready(launch.launch_id, timeout=300)
# ... use the model ...
await manager.stop(launch.launch_id)
```

### Other operations

```python
await manager.status(launch_id)    # LaunchResponse
await manager.list_launches()      # list[LaunchResponse]
await manager.logs(launch_id)      # LogsResponse
await manager.stats()              # SystemStatsResponse
```

## Run locally

```bash
uv run vllm-bootstrap
```

If your environment does not already include `vllm`, install with the optional extra:

```bash
pip install ".[vllm]"
```

Example launch request:

```bash
curl -X POST http://localhost:8000/launch \
  -H "content-type: application/json" \
  -d '{"model":"meta-llama/Llama-3.1-8B-Instruct", "task":"generate"}'
```

Launch an embedding model:

```bash
curl -X POST http://localhost:8000/launch \
  -H "content-type: application/json" \
  -d '{"model":"BAAI/bge-base-en-v1.5", "task":"embed"}'
```

Tail logs:

```bash
curl "http://localhost:8000/logs/<launch_id>?offset=0"
```

## REST API (port 8000)

Management endpoints for launching, monitoring, and stopping models.

- `POST /launch`
  - Loads a model in-process via `vllm.LLM`.
  - Requires `model` in request body. Optional `task` (`"generate"` or `"embed"`, default `"generate"`), `gpu_ids`, and `extra_kwargs`.
  - Returns `409` if requested/default GPUs are already owned by an active launch.
- `GET /launch`
  - Returns active (non-terminal) launches and their metadata.
- `GET /status/{launch_id}`
  - Returns launch state (`bootstrapping`, `ready`, `stopping`, `stopped`, `failed`) and metadata.
- `GET /logs/{launch_id}?offset=<int>`
  - Returns log chunk and next offset for incremental log streaming.
- `POST /stop/{launch_id}`
  - Stops the managed launch and releases its GPU ownership.
- `GET /stats`
  - Returns host load averages, CPU count, host memory usage, and per-GPU utilization/memory/power stats.

## gRPC API (port 8001)

Inference endpoints for embeddings and completions. Defined in [`proto/inference.proto`](proto/inference.proto).

- `InferenceService.Embed(EmbedRequest) → EmbedResponse` — compute embeddings for a list of texts.
- `InferenceService.Complete(CompleteRequest) → CompleteResponse` — generate a text completion.

Both RPCs require a `launch_id` referencing a model in `ready` state. Task must match the launch (`embed` or `generate`).

## Tests

```bash
make test
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
