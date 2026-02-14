from __future__ import annotations

import base64
import time

from fastapi.testclient import TestClient

import vllm_bootstrap.api as api_module
from vllm_bootstrap.manager import LaunchNotFoundError, LaunchSnapshot, LaunchState


class _StubManager:
    def __init__(self, snapshots: list[LaunchSnapshot]) -> None:
        self._snapshots = {snapshot.launch_id: snapshot for snapshot in snapshots}

    def list_launches(self, *, include_terminal: bool = False) -> list[LaunchSnapshot]:
        snapshots = list(self._snapshots.values())
        if include_terminal:
            return snapshots
        return [
            snapshot
            for snapshot in snapshots
            if snapshot.state not in {LaunchState.STOPPED, LaunchState.FAILED}
        ]

    def get_status(self, launch_id: str) -> LaunchSnapshot:
        snapshot = self._snapshots.get(launch_id)
        if snapshot is None:
            raise LaunchNotFoundError(f"Unknown launch_id: {launch_id}")
        return snapshot

    def stop_all(self) -> None:
        return


def _snapshot(
    *, launch_id: str, state: LaunchState, model: str = "model-a"
) -> LaunchSnapshot:
    now = time.time()
    return LaunchSnapshot(
        launch_id=launch_id,
        model=model,
        gpu_ids=[0, 1],
        port=8001,
        state=state,
        created_at=now,
        updated_at=now,
        return_code=None,
        error=None,
    )


def test_home_health_page_renders_active_launches(
    monkeypatch,
) -> None:
    snapshots = [
        _snapshot(launch_id="active-1", state=LaunchState.READY, model="model-ready"),
        _snapshot(
            launch_id="terminal-1", state=LaunchState.STOPPED, model="model-stopped"
        ),
    ]
    monkeypatch.setattr(api_module, "manager", _StubManager(snapshots))

    with TestClient(api_module.app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "vllm-bootstrap Home Health" in response.text
    assert "Active Launches: 1" in response.text
    assert "active-1" in response.text
    assert "model-ready" in response.text
    assert "terminal-1" not in response.text
    assert "model-stopped" not in response.text


def test_home_health_page_empty_state(monkeypatch) -> None:
    monkeypatch.setattr(api_module, "manager", _StubManager([]))

    with TestClient(api_module.app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "No running vLLM launches." in response.text


def test_list_launches_returns_active_launches_with_metadata(monkeypatch) -> None:
    snapshots = [
        _snapshot(launch_id="ready-1", state=LaunchState.READY, model="model-ready"),
        _snapshot(
            launch_id="stopped-1", state=LaunchState.STOPPED, model="model-stopped"
        ),
    ]
    monkeypatch.setattr(api_module, "manager", _StubManager(snapshots))
    monkeypatch.setattr(api_module.settings, "access_key", None)

    with TestClient(api_module.app) as client:
        response = client.get("/launch")

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
    assert len(payload) == 1
    assert payload[0]["launch_id"] == "ready-1"
    assert payload[0]["model"] == "model-ready"
    assert payload[0]["gpu_ids"] == [0, 1]
    assert payload[0]["port"] == 8001
    assert payload[0]["state"] == "ready"
    assert "created_at" in payload[0]
    assert "updated_at" in payload[0]


def test_access_key_returns_basic_challenge_without_auth_header(monkeypatch) -> None:
    monkeypatch.setattr(api_module, "manager", _StubManager([]))
    monkeypatch.setattr(api_module.settings, "access_key", "secret-key")

    with TestClient(api_module.app) as client:
        response = client.get("/")

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == 'Basic realm="vllm-bootstrap"'


def test_access_key_accepts_bearer_and_basic_auth(monkeypatch) -> None:
    monkeypatch.setattr(api_module, "manager", _StubManager([]))
    monkeypatch.setattr(api_module.settings, "access_key", "secret-key")

    basic_token = base64.b64encode(b"user:secret-key").decode()

    with TestClient(api_module.app) as client:
        bearer_response = client.get(
            "/", headers={"Authorization": "Bearer secret-key"}
        )
        basic_response = client.get(
            "/", headers={"Authorization": f"Basic {basic_token}"}
        )
        invalid_response = client.get("/", headers={"Authorization": "Bearer wrong"})

    assert bearer_response.status_code == 200
    assert basic_response.status_code == 200
    assert invalid_response.status_code == 401


class _MockUpstreamResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        content: bytes = b"{}",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.content = content
        self.headers = headers or {"content-type": "application/json"}


class _MockAsyncClient:
    last_request: dict | None = None
    response: _MockUpstreamResponse = _MockUpstreamResponse()
    error: Exception | None = None

    def __init__(self, *_, **__) -> None:
        return

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None

    async def request(self, **kwargs):
        _MockAsyncClient.last_request = kwargs
        if _MockAsyncClient.error is not None:
            raise _MockAsyncClient.error
        return _MockAsyncClient.response


def test_proxy_to_vllm_forwards_request(monkeypatch) -> None:
    snapshot = _snapshot(launch_id="ready-1", state=LaunchState.READY)
    monkeypatch.setattr(api_module, "manager", _StubManager([snapshot]))
    monkeypatch.setattr(api_module.settings, "access_key", None)
    monkeypatch.setattr(api_module.settings, "launch_host", "0.0.0.0")
    monkeypatch.setattr(api_module.httpx, "AsyncClient", _MockAsyncClient)
    _MockAsyncClient.error = None
    _MockAsyncClient.response = _MockUpstreamResponse(
        status_code=201,
        content=b'{"ok":true}',
        headers={"content-type": "application/json", "x-upstream": "yes"},
    )

    with TestClient(api_module.app) as client:
        response = client.post(
            "/proxy/ready-1/v1/chat/completions?stream=false",
            json={"model": "test-model"},
            headers={"x-custom-header": "abc"},
        )

    assert response.status_code == 201
    assert response.json() == {"ok": True}
    assert response.headers["x-upstream"] == "yes"
    assert _MockAsyncClient.last_request is not None
    assert (
        _MockAsyncClient.last_request["url"]
        == "http://127.0.0.1:8001/v1/chat/completions?stream=false"
    )
    assert _MockAsyncClient.last_request["method"] == "POST"
    assert b'"model":"test-model"' in _MockAsyncClient.last_request["content"]
    assert _MockAsyncClient.last_request["headers"]["x-custom-header"] == "abc"


def test_proxy_to_vllm_requires_ready_launch(monkeypatch) -> None:
    snapshot = _snapshot(launch_id="boot-1", state=LaunchState.BOOTSTRAPPING)
    monkeypatch.setattr(api_module, "manager", _StubManager([snapshot]))
    monkeypatch.setattr(api_module.settings, "access_key", None)

    with TestClient(api_module.app) as client:
        response = client.get("/proxy/boot-1/v1/models")

    assert response.status_code == 409
    assert "not ready" in response.json()["detail"]


def test_proxy_to_vllm_propagates_unknown_launch(monkeypatch) -> None:
    monkeypatch.setattr(api_module, "manager", _StubManager([]))
    monkeypatch.setattr(api_module.settings, "access_key", None)

    with TestClient(api_module.app) as client:
        response = client.get("/proxy/missing-launch/v1/models")

    assert response.status_code == 404


def test_proxy_to_vllm_returns_bad_gateway_for_upstream_error(monkeypatch) -> None:
    snapshot = _snapshot(launch_id="ready-2", state=LaunchState.READY)
    monkeypatch.setattr(api_module, "manager", _StubManager([snapshot]))
    monkeypatch.setattr(api_module.settings, "access_key", None)
    monkeypatch.setattr(api_module.httpx, "AsyncClient", _MockAsyncClient)
    _MockAsyncClient.error = api_module.httpx.ConnectError("connection refused")

    with TestClient(api_module.app) as client:
        response = client.get("/proxy/ready-2/v1/models")

    assert response.status_code == 502
    assert "Failed to reach launch ready-2 upstream server" in response.json()["detail"]
