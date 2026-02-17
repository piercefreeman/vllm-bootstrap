from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
import logging
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates

from .auth import build_access_key_middleware
from .config import load_settings
from .manager import (
    LaunchConflictError,
    LaunchManagerError,
    LaunchNotFoundError,
    LaunchState,
    LaunchValidationError,
    VLLMEnvironmentManager,
)
from .models import LaunchRequest, LaunchResponse, LogsResponse, SystemStatsResponse


settings = load_settings()
manager = VLLMEnvironmentManager(settings=settings)
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
logger = logging.getLogger(__name__)
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
PROXY_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"]


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


def _resolve_upstream_host(host: str) -> str:
    if host in {"0.0.0.0", "::"}:
        return "127.0.0.1"
    return host


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


@app.api_route("/proxy/{launch_id}/{upstream_path:path}", methods=PROXY_METHODS)
async def proxy_to_vllm(
    launch_id: str, upstream_path: str, request: Request
) -> Response:
    try:
        snapshot = manager.get_status(launch_id)
    except LaunchManagerError as error:
        raise _to_http_error(error) from error

    if snapshot.state != LaunchState.READY:
        raise HTTPException(
            status_code=409,
            detail=f"Launch {launch_id} is not ready (state={snapshot.state.value})",
        )

    upstream_host = _resolve_upstream_host(settings.launch_host)
    normalized_path = upstream_path.lstrip("/")
    upstream_url = f"http://{upstream_host}:{snapshot.port}/{normalized_path}"
    if request.url.query:
        upstream_url = f"{upstream_url}?{request.url.query}"

    upstream_headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS | {"host", "content-length"}
    }
    body = await request.body()

    try:
        async with httpx.AsyncClient(follow_redirects=False, timeout=None) as client:
            upstream_response = await client.request(
                method=request.method,
                url=upstream_url,
                headers=upstream_headers,
                content=body,
            )
    except httpx.RequestError as error:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to reach launch {launch_id} upstream server: {error}",
        ) from error

    response_headers = {
        key: value
        for key, value in upstream_response.headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS | {"content-length"}
    }
    return Response(
        content=upstream_response.content,
        status_code=upstream_response.status_code,
        headers=response_headers,
    )


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
