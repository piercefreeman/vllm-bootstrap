ARG VLLM_BASE_IMAGE=vllm/vllm-openai:v0.8.5.post1
FROM ${VLLM_BASE_IMAGE}

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md main.py ./
COPY vllm_bootstrap ./vllm_bootstrap

# vllm comes from the base image; package deps here are the control-plane runtime deps.
RUN pip install --no-cache-dir .

EXPOSE 8000 8001

ENTRYPOINT []
CMD ["python3", "-m", "vllm_bootstrap"]
