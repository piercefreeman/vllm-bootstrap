# vllm-bootstrap

Self contained docker image to treat a remote server (Runpod, GCP, AWS) as a generic endpoint for speeding up batch inference compute. Deploy the docker image to your remote server and dynamically update the model it's serving, get vllm bootstrap status, issue jobs, etc. 

Internally we implement this logic through a FastAPI control plane for launching and managing vLLM servers with explicit GPU ownership.

If you're interested in a step by step guide, check [this out](./docs/GETTING_STARTED.md).

## API

- `POST /launch`
  - Launches `vllm.entrypoints.openai.api_server`.
  - Requires a `model` to be provided in request body.
  - Defaults to launching on all detected GPUs.
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

- `VLLM_LAUNCH_PORT_START` / `VLLM_LAUNCH_PORT_END` port range for vLLM child processes.
- `VLLM_BOOTSTRAP_LOG_DIR` log directory for child process output.

GPU ownership and default allocation are auto-detected from host hardware via `nvidia-smi`.

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
