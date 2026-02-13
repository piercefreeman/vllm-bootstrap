FROM runpod/pytorch:1.0.3-cu1281-torch280-ubuntu2404

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

COPY pyproject.toml README.md main.py ./
COPY vllm_bootstrap ./vllm_bootstrap

RUN uv pip install --system .

COPY run.sh /app/run.sh
RUN chmod +x /app/run.sh

EXPOSE 8000

CMD ["/app/run.sh"]
