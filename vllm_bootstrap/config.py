from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

DEFAULT_READY_MARKERS = ("Application startup complete", "Uvicorn running on")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VLLM_",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    launch_host: str = "0.0.0.0"
    launch_port_start: int = 8001
    launch_port_end: int = 8099
    log_dir: Path = Field(
        default=Path("/tmp/vllm-bootstrap-logs"),
        validation_alias="VLLM_BOOTSTRAP_LOG_DIR",
    )
    stop_timeout_seconds: float = 30.0
    log_read_chunk_bytes: int = 1024 * 1024
    ready_scan_chunk_bytes: int = 256 * 1024
    ready_markers: Annotated[tuple[str, ...], NoDecode] = DEFAULT_READY_MARKERS
    access_key: str | None = None

    @field_validator("ready_markers", mode="before")
    @classmethod
    def _parse_ready_markers(
        cls, value: str | tuple[str, ...] | list[str] | None
    ) -> tuple[str, ...]:
        if value is None:
            return DEFAULT_READY_MARKERS

        if isinstance(value, str):
            markers = tuple(
                marker.strip() for marker in value.split("|") if marker.strip()
            )
            return markers or DEFAULT_READY_MARKERS

        if isinstance(value, (tuple, list)):
            markers = tuple(
                str(marker).strip() for marker in value if str(marker).strip()
            )
            return markers or DEFAULT_READY_MARKERS

        return DEFAULT_READY_MARKERS

    @field_validator("access_key", mode="before")
    @classmethod
    def _normalize_access_key(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    @model_validator(mode="after")
    def _validate_port_range(self) -> "Settings":
        if self.launch_port_start > self.launch_port_end:
            raise ValueError("VLLM_LAUNCH_PORT_START must be <= VLLM_LAUNCH_PORT_END")
        return self


def load_settings() -> Settings:
    return Settings()
