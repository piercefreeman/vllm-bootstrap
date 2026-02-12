from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

from .config import load_settings
from .manager import (
    LaunchConflictError,
    LaunchManagerError,
    LaunchNotFoundError,
    LaunchValidationError,
    VLLMEnvironmentManager,
)
from .models import LaunchRequest, LaunchResponse, LogsResponse


settings = load_settings()
manager = VLLMEnvironmentManager(settings=settings)


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


def _to_http_error(error: LaunchManagerError) -> HTTPException:
    if isinstance(error, LaunchNotFoundError):
        return HTTPException(status_code=404, detail=str(error))
    if isinstance(error, LaunchConflictError):
        return HTTPException(status_code=409, detail=str(error))
    if isinstance(error, LaunchValidationError):
        return HTTPException(status_code=400, detail=str(error))
    return HTTPException(status_code=500, detail=str(error))


@app.post("/launch", response_model=LaunchResponse, status_code=201)
def launch(request: LaunchRequest | None = None) -> LaunchResponse:
    launch_request = request or LaunchRequest()
    try:
        snapshot = manager.launch(
            model=launch_request.model,
            gpu_ids=launch_request.gpu_ids,
            port=launch_request.port,
            extra_args=launch_request.extra_args,
        )
    except LaunchManagerError as error:
        raise _to_http_error(error) from error
    return LaunchResponse.from_snapshot(snapshot)


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
