from __future__ import annotations

import base64
import time

from fastapi.testclient import TestClient

import vllm_bootstrap.api as api_module
from vllm_bootstrap.manager import (
    GPUStatsSnapshot,
    LaunchNotFoundError,
    LaunchSnapshot,
    LaunchState,
    SystemStatsSnapshot,
)


class _StubManager:
    def __init__(
        self,
        snapshots: list[LaunchSnapshot],
        *,
        system_stats: SystemStatsSnapshot | None = None,
    ) -> None:
        self._snapshots = {snapshot.launch_id: snapshot for snapshot in snapshots}
        self._system_stats = system_stats or _stats_snapshot()

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

    def get_system_stats(self) -> SystemStatsSnapshot:
        return self._system_stats


def _snapshot(
    *, launch_id: str, state: LaunchState, model: str = "model-a"
) -> LaunchSnapshot:
    now = time.time()
    return LaunchSnapshot(
        launch_id=launch_id,
        model=model,
        gpu_ids=[0, 1],
        task="generate",
        state=state,
        created_at=now,
        updated_at=now,
        error=None,
    )


def _stats_snapshot(
    *,
    nvidia_smi_error: str | None = None,
) -> SystemStatsSnapshot:
    now = time.time()
    return SystemStatsSnapshot(
        collected_at=now,
        load_avg_1m=0.7,
        load_avg_5m=0.6,
        load_avg_15m=0.5,
        cpu_count=16,
        memory_total_bytes=64_000_000_000,
        memory_available_bytes=32_000_000_000,
        memory_used_bytes=32_000_000_000,
        memory_utilization_percent=50.0,
        host_memory_error=None,
        gpu_count=1 if nvidia_smi_error is None else 0,
        gpus=[]
        if nvidia_smi_error
        else [
            GPUStatsSnapshot(
                gpu_id=0,
                uuid="GPU-abc",
                name="NVIDIA H100 80GB HBM3",
                utilization_percent=76.0,
                memory_total_mib=81559,
                memory_used_mib=42000,
                memory_free_mib=39559,
                temperature_c=42,
                power_draw_watts=248.5,
                power_limit_watts=700.0,
            )
        ],
        nvidia_smi_error=nvidia_smi_error,
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
    assert payload[0]["task"] == "generate"
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


def test_stats_returns_host_and_gpu_metrics(monkeypatch) -> None:
    monkeypatch.setattr(
        api_module, "manager", _StubManager([], system_stats=_stats_snapshot())
    )
    monkeypatch.setattr(api_module.settings, "access_key", None)

    with TestClient(api_module.app) as client:
        response = client.get("/stats")

    assert response.status_code == 200
    payload = response.json()
    assert payload["load_avg_1m"] == 0.7
    assert payload["cpu_count"] == 16
    assert payload["memory_total_bytes"] == 64_000_000_000
    assert payload["memory_utilization_percent"] == 50.0
    assert payload["gpu_count"] == 1
    assert payload["gpus"][0]["gpu_id"] == 0
    assert payload["gpus"][0]["memory_used_mib"] == 42000
    assert payload["gpus"][0]["utilization_percent"] == 76.0
    assert payload["nvidia_smi_error"] is None


def test_stats_surfaces_nvidia_smi_error_without_failing(monkeypatch) -> None:
    monkeypatch.setattr(
        api_module,
        "manager",
        _StubManager(
            [],
            system_stats=_stats_snapshot(
                nvidia_smi_error="nvidia-smi is unavailable on this host."
            ),
        ),
    )
    monkeypatch.setattr(api_module.settings, "access_key", None)

    with TestClient(api_module.app) as client:
        response = client.get("/stats")

    assert response.status_code == 200
    payload = response.json()
    assert payload["gpu_count"] == 0
    assert payload["gpus"] == []
    assert payload["nvidia_smi_error"] == "nvidia-smi is unavailable on this host."
