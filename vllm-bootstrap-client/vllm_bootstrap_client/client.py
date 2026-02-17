from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import urlparse

import grpc.aio
import httpx

from .generated import inference_pb2, inference_pb2_grpc
from .models import (
    LaunchResponse,
    LaunchState,
    LogsResponse,
    SystemStatsResponse,
)


class VLLMManager:
    """Async client for vllm-bootstrap.

    Uses httpx for REST management endpoints and gRPC for inference.
    """

    def __init__(
        self,
        base_url: str,
        *,
        grpc_address: str | None = None,
        access_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._access_key = access_key
        self._timeout = timeout

        if grpc_address is None:
            parsed = urlparse(self._base_url)
            host = parsed.hostname or "localhost"
            rest_port = parsed.port or 8000
            grpc_address = f"{host}:{rest_port + 1}"

        self._grpc_address = grpc_address
        self._http = httpx.AsyncClient(
            base_url=self._base_url,
            headers=self._auth_headers(),
            timeout=timeout,
        )
        self._channel = grpc.aio.insecure_channel(self._grpc_address)
        self._stub = inference_pb2_grpc.InferenceServiceStub(self._channel)

    def _auth_headers(self) -> dict[str, str]:
        if self._access_key:
            return {"Authorization": f"Bearer {self._access_key}"}
        return {}

    def _grpc_metadata(self) -> list[tuple[str, str]] | None:
        if self._access_key:
            return [("authorization", f"Bearer {self._access_key}")]
        return None

    async def close(self) -> None:
        await self._http.aclose()
        await self._channel.close()

    async def __aenter__(self) -> VLLMManager:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    # -- REST management endpoints --

    async def launch(
        self,
        model: str,
        *,
        task: str = "generate",
        gpu_ids: list[int] | None = None,
        extra_kwargs: dict[str, Any] | None = None,
    ) -> LaunchResponse:
        body: dict[str, Any] = {"model": model, "task": task}
        if gpu_ids is not None:
            body["gpu_ids"] = gpu_ids
        if extra_kwargs is not None:
            body["extra_kwargs"] = extra_kwargs
        resp = await self._http.post("/launch", json=body)
        resp.raise_for_status()
        return LaunchResponse.model_validate(resp.json())

    async def status(self, launch_id: str) -> LaunchResponse:
        resp = await self._http.get(f"/status/{launch_id}")
        resp.raise_for_status()
        return LaunchResponse.model_validate(resp.json())

    async def list_launches(self) -> list[LaunchResponse]:
        resp = await self._http.get("/launch")
        resp.raise_for_status()
        return [LaunchResponse.model_validate(item) for item in resp.json()]

    async def stop(self, launch_id: str) -> LaunchResponse:
        resp = await self._http.post(f"/stop/{launch_id}")
        resp.raise_for_status()
        return LaunchResponse.model_validate(resp.json())

    async def logs(self, launch_id: str, offset: int = 0) -> LogsResponse:
        resp = await self._http.get(f"/logs/{launch_id}", params={"offset": offset})
        resp.raise_for_status()
        return LogsResponse.model_validate(resp.json())

    async def stats(self) -> SystemStatsResponse:
        resp = await self._http.get("/stats")
        resp.raise_for_status()
        return SystemStatsResponse.model_validate(resp.json())

    async def wait_until_ready(
        self,
        launch_id: str,
        *,
        poll_interval: float = 2.0,
        timeout: float | None = None,
    ) -> LaunchResponse:
        """Poll status until the launch reaches READY (or a terminal state)."""
        deadline = asyncio.get_event_loop().time() + timeout if timeout else None
        while True:
            snapshot = await self.status(launch_id)
            if snapshot.state == LaunchState.ready:
                return snapshot
            if snapshot.state in (LaunchState.stopped, LaunchState.failed):
                raise RuntimeError(
                    f"Launch {launch_id} reached terminal state: {snapshot.state.value}"
                    + (f" ({snapshot.error})" if snapshot.error else "")
                )
            if deadline is not None and asyncio.get_event_loop().time() >= deadline:
                raise TimeoutError(
                    f"Launch {launch_id} not ready after {timeout}s (state={snapshot.state.value})"
                )
            await asyncio.sleep(poll_interval)

    # -- lifecycle helpers --

    @asynccontextmanager
    async def run(
        self,
        model: str,
        *,
        task: str = "generate",
        gpu_ids: list[int] | None = None,
        extra_kwargs: dict[str, Any] | None = None,
        ready_timeout: float = 600,
        ready_poll_interval: float = 2.0,
    ) -> AsyncIterator[LaunchResponse]:
        """Launch a model, wait until ready, yield, and stop on exit."""
        snapshot = await self.launch(
            model, task=task, gpu_ids=gpu_ids, extra_kwargs=extra_kwargs
        )
        try:
            snapshot = await self.wait_until_ready(
                snapshot.launch_id,
                timeout=ready_timeout,
                poll_interval=ready_poll_interval,
            )
            yield snapshot
        finally:
            await self.stop(snapshot.launch_id)

    # -- gRPC inference endpoints --

    async def embed(self, launch_id: str, texts: list[str]) -> list[list[float]]:
        response = await self._stub.Embed(
            inference_pb2.EmbedRequest(launch_id=launch_id, texts=texts),
            metadata=self._grpc_metadata(),
        )
        return [list(e.values) for e in response.embeddings]

    async def complete(
        self,
        launch_id: str,
        prompts: list[str],
        max_tokens: int,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        json_schema: str | None = None,
        regex: str | None = None,
        grammar: str | None = None,
        choice: list[str] | None = None,
    ) -> list[inference_pb2.Completion]:
        kwargs: dict[str, Any] = {
            "launch_id": launch_id,
            "prompts": prompts,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p

        gd_kwargs: dict[str, Any] = {}
        if json_schema is not None:
            gd_kwargs["json_schema"] = json_schema
        if regex is not None:
            gd_kwargs["regex"] = regex
        if grammar is not None:
            gd_kwargs["grammar"] = grammar
        if choice is not None:
            gd_kwargs["choice"] = choice
        if gd_kwargs:
            kwargs["guided_decoding"] = inference_pb2.GuidedDecodingParams(**gd_kwargs)

        response = await self._stub.Complete(
            inference_pb2.CompleteRequest(**kwargs),
            metadata=self._grpc_metadata(),
        )
        return list(response.completions)
