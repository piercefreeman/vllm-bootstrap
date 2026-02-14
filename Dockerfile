FROM vllm/vllm-openai:latest

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md main.py ./
COPY vllm_bootstrap ./vllm_bootstrap

RUN pip install --no-deps .

EXPOSE 8000

ENTRYPOINT []
CMD ["uvicorn", "vllm_bootstrap.api:app", "--host", "0.0.0.0", "--port", "8000"]
