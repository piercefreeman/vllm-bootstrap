ARG BASE_IMAGE=nvidia/cuda:12.1.0-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1

# Install uv for fast dependency resolution and Python management.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml README.md main.py ./
COPY vllm_bootstrap ./vllm_bootstrap

# vllm often reinstalls torch and pulls in additional dependencies beyond
# what the base image provides, so this step can be slow and large.
RUN uv sync --extra vllm

EXPOSE 8000 8001

ENTRYPOINT []
CMD ["uv", "run", "python3", "-m", "vllm_bootstrap"]
