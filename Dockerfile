FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1

# Install uv for fast dependency resolution.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml README.md main.py ./
COPY vllm_bootstrap ./vllm_bootstrap

RUN uv pip install --system ".[vllm]" --constraint <(pip freeze | grep "^torch==")

EXPOSE 8000 8001

ENTRYPOINT []
CMD ["python3", "-m", "vllm_bootstrap"]
