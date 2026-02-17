from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from .manager import (
    GPUStatsSnapshot,
    LaunchSnapshot,
    LaunchState,
    LogSnapshot,
    SystemStatsSnapshot,
)


def _as_datetime(timestamp: float) -> datetime:
    return datetime.fromtimestamp(timestamp, tz=UTC)


class LaunchRequest(BaseModel):
    model: str = Field(
        min_length=1,
        description="Model id/path passed to vLLM. Required.",
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


class GPUStatsResponse(BaseModel):
    gpu_id: int
    uuid: str | None = None
    name: str
    utilization_percent: float | None = None
    memory_total_mib: int | None = None
    memory_used_mib: int | None = None
    memory_free_mib: int | None = None
    temperature_c: int | None = None
    power_draw_watts: float | None = None
    power_limit_watts: float | None = None

    @classmethod
    def from_snapshot(cls, snapshot: GPUStatsSnapshot) -> "GPUStatsResponse":
        return cls(
            gpu_id=snapshot.gpu_id,
            uuid=snapshot.uuid,
            name=snapshot.name,
            utilization_percent=snapshot.utilization_percent,
            memory_total_mib=snapshot.memory_total_mib,
            memory_used_mib=snapshot.memory_used_mib,
            memory_free_mib=snapshot.memory_free_mib,
            temperature_c=snapshot.temperature_c,
            power_draw_watts=snapshot.power_draw_watts,
            power_limit_watts=snapshot.power_limit_watts,
        )


class SystemStatsResponse(BaseModel):
    collected_at: datetime
    load_avg_1m: float | None = None
    load_avg_5m: float | None = None
    load_avg_15m: float | None = None
    cpu_count: int | None = None
    memory_total_bytes: int | None = None
    memory_available_bytes: int | None = None
    memory_used_bytes: int | None = None
    memory_utilization_percent: float | None = None
    host_memory_error: str | None = None
    gpu_count: int
    gpus: list[GPUStatsResponse]
    nvidia_smi_error: str | None = None

    @classmethod
    def from_snapshot(cls, snapshot: SystemStatsSnapshot) -> "SystemStatsResponse":
        return cls(
            collected_at=_as_datetime(snapshot.collected_at),
            load_avg_1m=snapshot.load_avg_1m,
            load_avg_5m=snapshot.load_avg_5m,
            load_avg_15m=snapshot.load_avg_15m,
            cpu_count=snapshot.cpu_count,
            memory_total_bytes=snapshot.memory_total_bytes,
            memory_available_bytes=snapshot.memory_available_bytes,
            memory_used_bytes=snapshot.memory_used_bytes,
            memory_utilization_percent=snapshot.memory_utilization_percent,
            host_memory_error=snapshot.host_memory_error,
            gpu_count=snapshot.gpu_count,
            gpus=[
                GPUStatsResponse.from_snapshot(gpu_snapshot)
                for gpu_snapshot in snapshot.gpus
            ],
            nvidia_smi_error=snapshot.nvidia_smi_error,
        )
