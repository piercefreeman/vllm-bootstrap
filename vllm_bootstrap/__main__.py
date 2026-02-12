from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.getenv("VLLM_BOOTSTRAP_HOST", "0.0.0.0")
    port = int(os.getenv("VLLM_BOOTSTRAP_PORT", "8000"))
    uvicorn.run("vllm_bootstrap.api:app", host=host, port=port)


if __name__ == "__main__":
    main()
