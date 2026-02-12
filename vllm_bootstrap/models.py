from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from .manager import LaunchSnapshot, LaunchState, LogSnapshot


def _as_datetime(timestamp: float) -> datetime:
    return datetime.fromtimestamp(timestamp, tz=UTC)


class LaunchRequest(BaseModel):
    model: str | None = Field(
        default=None,
        description="Model id/path passed to vLLM. Defaults to VLLM_DEFAULT_MODEL.",
    )
    gpu_ids: list[int] | None = Field(
        default=None,
        description="Specific GPU ids to allocate. Defaults to all visible GPUs.",
    )
    port: int | None = Field(
        default=None,
        ge=1,
        le=65535,
        description="Optional API port for the launched vLLM process.",
    )
    extra_args: list[str] = Field(
        default_factory=list,
        description="Additional CLI flags appended to vllm.entrypoints.openai.api_server.",
    )


class LaunchResponse(BaseModel):
    launch_id: str
    model: str
    gpu_ids: list[int]
    port: int
    state: LaunchState
    created_at: datetime
    updated_at: datetime
    return_code: int | None = None
    error: str | None = None

    @classmethod
    def from_snapshot(cls, snapshot: LaunchSnapshot) -> "LaunchResponse":
        return cls(
            launch_id=snapshot.launch_id,
            model=snapshot.model,
            gpu_ids=snapshot.gpu_ids,
            port=snapshot.port,
            state=snapshot.state,
            created_at=_as_datetime(snapshot.created_at),
            updated_at=_as_datetime(snapshot.updated_at),
            return_code=snapshot.return_code,
            error=snapshot.error,
        )


class LogsResponse(BaseModel):
    launch_id: str
    offset: int
    next_offset: int
    content: str

    @classmethod
    def from_snapshot(cls, snapshot: LogSnapshot) -> "LogsResponse":
        return cls(
            launch_id=snapshot.launch_id,
            offset=snapshot.offset,
            next_offset=snapshot.next_offset,
            content=snapshot.content,
        )
