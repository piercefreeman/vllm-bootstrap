ARG BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime
FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1

# Install uv for fast dependency resolution.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml README.md main.py ./
COPY vllm_bootstrap ./vllm_bootstrap

RUN pip freeze | grep "^torch==" > /tmp/torch-constraint.txt && \
    uv pip install --system ".[vllm]" --constraint /tmp/torch-constraint.txt

EXPOSE 8000 8001

ENTRYPOINT []
CMD ["python3", "-m", "vllm_bootstrap"]
