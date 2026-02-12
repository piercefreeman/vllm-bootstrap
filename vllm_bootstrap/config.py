from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True, frozen=True)
class Settings:
    default_model: str
    launch_host: str
    launch_port_start: int
    launch_port_end: int
    log_dir: Path
    stop_timeout_seconds: float
    log_read_chunk_bytes: int
    ready_scan_chunk_bytes: int
    ready_markers: tuple[str, ...]


def load_settings() -> Settings:
    default_markers = ("Application startup complete", "Uvicorn running on")
    marker_env = os.getenv("VLLM_READY_MARKERS", "|".join(default_markers))
    markers = tuple(
        marker.strip() for marker in marker_env.split("|") if marker.strip()
    )

    port_start = int(os.getenv("VLLM_LAUNCH_PORT_START", "8001"))
    port_end = int(os.getenv("VLLM_LAUNCH_PORT_END", "8099"))
    if port_start > port_end:
        raise ValueError("VLLM_LAUNCH_PORT_START must be <= VLLM_LAUNCH_PORT_END")

    return Settings(
        default_model=os.getenv(
            "VLLM_DEFAULT_MODEL", "meta-llama/Llama-3.1-8B-Instruct"
        ),
        launch_host=os.getenv("VLLM_LAUNCH_HOST", "0.0.0.0"),
        launch_port_start=port_start,
        launch_port_end=port_end,
        log_dir=Path(os.getenv("VLLM_BOOTSTRAP_LOG_DIR", "/tmp/vllm-bootstrap-logs")),
        stop_timeout_seconds=float(os.getenv("VLLM_STOP_TIMEOUT_SECONDS", "30")),
        log_read_chunk_bytes=int(
            os.getenv("VLLM_LOG_READ_CHUNK_BYTES", str(1024 * 1024))
        ),
        ready_scan_chunk_bytes=int(
            os.getenv("VLLM_READY_SCAN_CHUNK_BYTES", str(256 * 1024))
        ),
        ready_markers=markers or default_markers,
    )
