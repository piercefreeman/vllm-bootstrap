from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates

from .auth import build_access_key_middleware
from .config import load_settings
from .manager import (
    LaunchConflictError,
    LaunchManagerError,
    LaunchNotFoundError,
    LaunchValidationError,
    VLLMEnvironmentManager,
)
from .models import LaunchRequest, LaunchResponse, LogsResponse, SystemStatsResponse


settings = load_settings()
manager = VLLMEnvironmentManager(settings=settings)
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    manager.stop_all()


app = FastAPI(
    title="vllm-bootstrap",
    version="0.1.0",
    description="FastAPI wrapper for launching and managing vLLM jobs with explicit GPU ownership.",
    lifespan=lifespan,
)


enforce_access_key = build_access_key_middleware(
    access_key_getter=lambda: settings.access_key
)
app.middleware("http")(enforce_access_key)


def _to_http_error(error: LaunchManagerError) -> HTTPException:
    if isinstance(error, LaunchNotFoundError):
        return HTTPException(status_code=404, detail=str(error))
    if isinstance(error, LaunchConflictError):
        return HTTPException(status_code=409, detail=str(error))
    if isinstance(error, LaunchValidationError):
        return HTTPException(status_code=400, detail=str(error))
    return HTTPException(status_code=500, detail=str(error))


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    active_launches = manager.list_launches(include_terminal=False)
    launches = [
        {
            "launch_id": snapshot.launch_id,
            "model": snapshot.model,
            "gpu_ids": ", ".join(str(gpu_id) for gpu_id in snapshot.gpu_ids),
            "task": snapshot.task,
            "state": snapshot.state.value,
            "updated_at": datetime.fromtimestamp(snapshot.updated_at, tz=UTC),
        }
        for snapshot in reversed(active_launches)
    ]
    context = {
        "request": request,
        "launches": launches,
        "generated_at": datetime.now(tz=UTC),
    }
    return templates.TemplateResponse(request, "home.html", context)


@app.post("/launch", response_model=LaunchResponse, status_code=201)
def launch(request: LaunchRequest) -> LaunchResponse:
    try:
        snapshot = manager.launch(
            model=request.model,
            gpu_ids=request.gpu_ids,
            task=request.task,
            extra_kwargs=request.extra_kwargs,
        )
    except LaunchManagerError as error:
        raise _to_http_error(error) from error
    return LaunchResponse.from_snapshot(snapshot)


@app.get("/launch", response_model=list[LaunchResponse])
def list_launches() -> list[LaunchResponse]:
    try:
        snapshots = manager.list_launches(include_terminal=False)
    except LaunchManagerError as error:
        raise _to_http_error(error) from error
    return [LaunchResponse.from_snapshot(snapshot) for snapshot in snapshots]


@app.get("/status/{launch_id}", response_model=LaunchResponse)
def status(launch_id: str) -> LaunchResponse:
    try:
        snapshot = manager.get_status(launch_id)
    except LaunchManagerError as error:
        raise _to_http_error(error) from error
    return LaunchResponse.from_snapshot(snapshot)


@app.get("/logs/{launch_id}", response_model=LogsResponse)
def logs(launch_id: str, offset: int = Query(default=0, ge=0)) -> LogsResponse:
    try:
        snapshot = manager.read_logs(launch_id=launch_id, offset=offset)
    except LaunchManagerError as error:
        raise _to_http_error(error) from error
    return LogsResponse.from_snapshot(snapshot)


@app.post("/stop/{launch_id}", response_model=LaunchResponse)
def stop(launch_id: str) -> LaunchResponse:
    try:
        snapshot = manager.stop(launch_id)
    except LaunchManagerError as error:
        raise _to_http_error(error) from error
    return LaunchResponse.from_snapshot(snapshot)


@app.get("/stats", response_model=SystemStatsResponse)
def stats() -> SystemStatsResponse:
    snapshot = manager.get_system_stats()

    per_gpu_stats = ", ".join(
        (
            f"gpu={gpu_snapshot.gpu_id}:"
            f"util={gpu_snapshot.utilization_percent}"
            f" mem={gpu_snapshot.memory_used_mib}/{gpu_snapshot.memory_total_mib}MiB"
        )
        for gpu_snapshot in snapshot.gpus
    )
    logger.info(
        "System stats snapshot load_1m=%s load_5m=%s load_15m=%s memory_utilization=%s gpu_count=%s %s",
        snapshot.load_avg_1m,
        snapshot.load_avg_5m,
        snapshot.load_avg_15m,
        snapshot.memory_utilization_percent,
        snapshot.gpu_count,
        per_gpu_stats,
    )
    if snapshot.nvidia_smi_error:
        logger.warning("nvidia-smi stats unavailable: %s", snapshot.nvidia_smi_error)
    if snapshot.host_memory_error:
        logger.warning("host memory stats unavailable: %s", snapshot.host_memory_error)

    return SystemStatsResponse.from_snapshot(snapshot)
