# vllm-bootstrap

Self contained docker image to treat a remote server (Runpod, GCP, AWS) as a generic endpoint for speeding up batch inference compute. Deploy the docker image to your remote server and dynamically update the model it's serving, get vllm bootstrap status, issue jobs, etc. 

Internally we implement this logic through a FastAPI control plane for launching and managing vLLM servers with explicit GPU ownership.

If you're interested in a step by step guide for Runpod, check [this out](./docs/RUNPOD.md).

## API

- `POST /launch`
  - Launches `vllm.entrypoints.openai.api_server`.
  - Requires a `model` to be provided in request body.
  - Defaults to launching on all detected GPUs.
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
  - Uses `nvidia-smi` for GPU metrics and includes an error field if GPU stats are unavailable.
- `ANY /proxy/{launch_id}/{upstream_path}`
  - Reverse proxies requests to the launched vLLM server for that `launch_id`.
  - Requires the launch to be in `ready` state; returns `409` otherwise.
  - For compatibility with OpenAI-style clients, payloads are passed through without strict request-body schema validation.

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
  -d '{"model":"meta-llama/Llama-3.1-8B-Instruct"}'
```

Tail logs:

```bash
curl "http://localhost:8000/logs/<launch_id>?offset=0"
```

## Docker

Most often you'll want to point your remote box to our image directly. Check what CUDA version it supports on the host box (this is passed through
to the container). For a device with 12.4:

```
docker pull ghcr.io/piercefreeman/vllm-bootstrap:cuda12.4-latest
```

If you want to build locally:

```bash
docker build \
  --build-arg VLLM_BASE_IMAGE=vllm/vllm-openai:v0.8.5.post1 \
  -t vllm-bootstrap:cuda12.4-local .
```

Run:

```bash
docker run --rm -p 8000:8000 -p 8001:8001 vllm-bootstrap:cuda12.4-local
```

Port 8000 serves the bootstrap control plane API, and port 8001 is the gRPC service.

## Key environment variables

- `VLLM_LAUNCH_PORT_START` / `VLLM_LAUNCH_PORT_END` port range for vLLM child processes.
- `VLLM_BOOTSTRAP_LOG_DIR` log directory for child process output.
- `VLLM_ACCESS_KEY` optional shared key that protects all routes.
  - `Authorization: Bearer <key>` is accepted.
  - If no auth header is sent, the server returns `401` with a `WWW-Authenticate: Basic` challenge.
  - HTTP Basic auth is accepted with any username and `<key>` as password.

GPU ownership and default allocation are auto-detected from host hardware via `nvidia-smi`.

## Tests

```bash
PYTHONPATH=. uv run --no-project --with pytest --with fastapi --with jinja2 --with pydantic-settings --with httpx pytest -q
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
