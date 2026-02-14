from __future__ import annotations

import base64
import time

from fastapi.testclient import TestClient

import vllm_bootstrap.api as api_module
from vllm_bootstrap.manager import LaunchSnapshot, LaunchState


class _StubManager:
    def __init__(self, snapshots: list[LaunchSnapshot]) -> None:
        self._snapshots = snapshots

    def list_launches(self, *, include_terminal: bool = False) -> list[LaunchSnapshot]:
        if include_terminal:
            return list(self._snapshots)
        return [
            snapshot
            for snapshot in self._snapshots
            if snapshot.state not in {LaunchState.STOPPED, LaunchState.FAILED}
        ]

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
