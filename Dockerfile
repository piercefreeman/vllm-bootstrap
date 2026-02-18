FROM runpod/pytorch:1.0.3-cu1281-torch260-ubuntu2204

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
