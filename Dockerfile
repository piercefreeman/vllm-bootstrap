FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1

# Install vllm on top of pre-installed pytorch/cuda.
RUN pip install --no-cache-dir vllm

WORKDIR /app

COPY pyproject.toml README.md main.py ./
COPY vllm_bootstrap ./vllm_bootstrap

RUN pip install --no-cache-dir .

EXPOSE 8000 8001

ENTRYPOINT []
CMD ["python3", "-m", "vllm_bootstrap"]
