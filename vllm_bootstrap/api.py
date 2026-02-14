from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

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
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


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


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    active_launches = manager.list_launches(include_terminal=False)
    launches = [
        {
            "launch_id": snapshot.launch_id,
            "model": snapshot.model,
            "gpu_ids": ", ".join(str(gpu_id) for gpu_id in snapshot.gpu_ids),
            "port": snapshot.port,
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
            port=request.port,
            extra_args=request.extra_args,
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
