from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest

from vllm_bootstrap.config import Settings
from vllm_bootstrap.manager import (
    LaunchConflictError,
    LaunchState,
    LaunchValidationError,
    VLLMEnvironmentManager,
    _LaunchRecord,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> str:
    return (FIXTURES_DIR / name).read_text()


def _make_settings(tmp_path: Path) -> Settings:
    return Settings(
        log_dir=tmp_path / "logs",
        stop_timeout_seconds=2.0,
        log_read_chunk_bytes=64 * 1024,
        grpc_port=8001,
    )


@pytest.fixture
def manager(tmp_path: Path) -> VLLMEnvironmentManager:
    instance = VLLMEnvironmentManager(settings=_make_settings(tmp_path))
    try:
        yield instance
    finally:
        instance.stop_all()


def test_launch_requires_model(
    monkeypatch: pytest.MonkeyPatch, manager: VLLMEnvironmentManager
) -> None:
    nvidia_smi_fixture = _load_fixture("nvidia_smi_query_gpu_index.txt")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        subprocess, "check_output", lambda *_a, **_kw: nvidia_smi_fixture
    )
    with pytest.raises(LaunchValidationError):
        manager.launch(model="   ", gpu_ids=None, task="generate")


def test_launch_validates_task(
    monkeypatch: pytest.MonkeyPatch, manager: VLLMEnvironmentManager
) -> None:
    nvidia_smi_fixture = _load_fixture("nvidia_smi_query_gpu_index.txt")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        subprocess, "check_output", lambda *_a, **_kw: nvidia_smi_fixture
    )
    with pytest.raises(LaunchValidationError, match="task must be"):
        manager.launch(model="some-model", gpu_ids=None, task="invalid")


def test_generate_and_embed_not_ready(
    monkeypatch: pytest.MonkeyPatch, manager: VLLMEnvironmentManager
) -> None:
    """Test that generate/embed raise when launch is not ready."""
    nvidia_smi_fixture = _load_fixture("nvidia_smi_query_gpu_index.txt")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        subprocess, "check_output", lambda *_a, **_kw: nvidia_smi_fixture
    )

    now = time.time()
    record = _LaunchRecord(
        launch_id="test-not-ready",
        model="some-model",
        gpu_ids=[0],
        task="generate",
        state=LaunchState.BOOTSTRAPPING,
        created_at=now,
        updated_at=now,
    )
    manager._launches["test-not-ready"] = record

    with pytest.raises(LaunchConflictError, match="not ready"):
        manager.generate("test-not-ready", ["hello"], {})


def test_discover_gpu_ids_parses_real_nvidia_smi_output(
    monkeypatch: pytest.MonkeyPatch, manager: VLLMEnvironmentManager
) -> None:
    nvidia_smi_fixture = _load_fixture("nvidia_smi_query_gpu_index.txt")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        subprocess, "check_output", lambda *_a, **_kw: nvidia_smi_fixture
    )

    gpu_ids = manager._discover_gpu_ids()
    assert gpu_ids == [0, 1]


def test_discover_gpu_ids_nvidia_smi_unavailable(
    monkeypatch: pytest.MonkeyPatch, manager: VLLMEnvironmentManager
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    def _raise_file_not_found(*_a, **_kw):
        raise FileNotFoundError("nvidia-smi not found")

    monkeypatch.setattr(subprocess, "check_output", _raise_file_not_found)

    with pytest.raises(LaunchValidationError, match="nvidia-smi is unavailable"):
        manager._discover_gpu_ids()


def test_discover_gpu_ids_nvidia_smi_failure(
    monkeypatch: pytest.MonkeyPatch, manager: VLLMEnvironmentManager
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    def _raise_called_process_error(*_a, **_kw):
        raise subprocess.CalledProcessError(
            returncode=1, cmd=["nvidia-smi"], output="NVIDIA-SMI has failed"
        )

    monkeypatch.setattr(subprocess, "check_output", _raise_called_process_error)

    with pytest.raises(LaunchValidationError, match="Failed to discover GPUs"):
        manager._discover_gpu_ids()


def test_parse_gpu_stats_parses_real_nvidia_smi_query_output(
    manager: VLLMEnvironmentManager,
) -> None:
    nvidia_smi_fixture = _load_fixture("nvidia_smi_query_gpu_stats.txt")
    parsed = manager._parse_gpu_stats(nvidia_smi_fixture)

    assert len(parsed) == 2
    assert parsed[0].gpu_id == 0
    assert parsed[0].uuid == "GPU-1234abcd-0000-1111-2222-333344445555"
    assert parsed[0].name == "NVIDIA H100 80GB HBM3"
    assert parsed[0].utilization_percent == 81.0
    assert parsed[0].memory_total_mib == 81559
    assert parsed[0].memory_used_mib == 40960
    assert parsed[0].temperature_c == 43
    assert parsed[0].power_draw_watts == 251.37
    assert parsed[0].power_limit_watts == 700.0


def test_get_system_stats_handles_missing_nvidia_smi(
    monkeypatch: pytest.MonkeyPatch, manager: VLLMEnvironmentManager
) -> None:
    def _raise_file_not_found(*_a, **_kw):
        raise FileNotFoundError("nvidia-smi not found")

    monkeypatch.setattr(subprocess, "check_output", _raise_file_not_found)

    stats = manager.get_system_stats()
    assert stats.gpu_count == 0
    assert stats.gpus == []
    assert stats.nvidia_smi_error == "nvidia-smi is unavailable on this host."
