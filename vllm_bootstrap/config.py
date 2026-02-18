from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VLLM_",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    log_dir: Path = Field(
        default=Path("/tmp/vllm-bootstrap-logs"),
        validation_alias="VLLM_BOOTSTRAP_LOG_DIR",
    )
    stop_timeout_seconds: float = 30.0
    log_read_chunk_bytes: int = 1024 * 1024
    grpc_port: int = 8001
    grpc_limit_message_size: bool = False
    access_key: str | None = None

    @field_validator("access_key", mode="before")
    @classmethod
    def _normalize_access_key(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None


def load_settings() -> Settings:
    return Settings()
