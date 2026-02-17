from __future__ import annotations

import logging
import os

import uvicorn

from .api import manager, settings
from .grpc_server import create_grpc_server

logger = logging.getLogger(__name__)


def main() -> None:
    host = os.getenv("VLLM_BOOTSTRAP_HOST", "0.0.0.0")
    port = int(os.getenv("VLLM_BOOTSTRAP_PORT", "8000"))

    grpc_server = create_grpc_server(manager, settings.grpc_port)
    grpc_server.start()
    logger.info("gRPC server started on port %d", settings.grpc_port)

    try:
        uvicorn.run("vllm_bootstrap.api:app", host=host, port=port)
    finally:
        grpc_server.stop(grace=5)
        logger.info("gRPC server stopped")


if __name__ == "__main__":
    main()
