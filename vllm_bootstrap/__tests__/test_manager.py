from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from vllm_bootstrap.config import Settings
from vllm_bootstrap.manager import (
    LaunchConflictError,
    LaunchState,
    LaunchValidationError,
    VLLMEnvironmentManager,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> str:
    return (FIXTURES_DIR / name).read_text()


def _wait_until(
    predicate, timeout_seconds: float = 5.0, interval_seconds: float = 0.05
) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(interval_seconds)
    raise AssertionError("condition was not met before timeout")


def _make_settings(tmp_path: Path) -> Settings:
    return Settings(
        log_dir=tmp_path / "logs",
        stop_timeout_seconds=2.0,
        log_read_chunk_bytes=64 * 1024,
        grpc_port=8001,
    )


def _make_mock_llm():
    """Create a mock vllm.LLM instance."""
    return MagicMock()


def _install_fake_vllm(monkeypatch, llm_side_effect=None):
    """Install a fake vllm module in sys.modules so import vllm works.

    Returns the mock LLM class so tests can inspect calls.
    """
    mock_llm_class = MagicMock()
    if llm_side_effect is not None:
        mock_llm_class.side_effect = llm_side_effect

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = mock_llm_class

    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    return mock_llm_class


@pytest.fixture
def manager(tmp_path: Path) -> VLLMEnvironmentManager:
    instance = VLLMEnvironmentManager(settings=_make_settings(tmp_path))
    try:
        yield instance
    finally:
        instance.stop_all()


def test_launch_defaults_to_all_gpus_and_conflicts(
    monkeypatch: pytest.MonkeyPatch, manager: VLLMEnvironmentManager
) -> None:
    nvidia_smi_fixture = _load_fixture("nvidia_smi_query_gpu_index.txt")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        subprocess, "check_output", lambda *_a, **_kw: nvidia_smi_fixture
    )

    mock_llm = _make_mock_llm()
    _install_fake_vllm(monkeypatch, llm_side_effect=lambda *a, **kw: mock_llm)

    first_launch = manager.launch(
        model="meta-llama/Llama-3.1-8B-Instruct",
        gpu_ids=None,
        task="generate",
    )
    assert first_launch.gpu_ids == [0, 1]

    _wait_until(
        lambda: manager.get_status(first_launch.launch_id).state == LaunchState.READY
    )

    with pytest.raises(LaunchConflictError):
        manager.launch(
            model="meta-llama/Llama-3.1-8B-Instruct",
            gpu_ids=None,
            task="generate",
        )

    stopped = manager.stop(first_launch.launch_id)
    assert stopped.state == LaunchState.STOPPED


def test_logs_populated_from_vllm_logger(
    monkeypatch: pytest.MonkeyPatch, manager: VLLMEnvironmentManager
) -> None:
    nvidia_smi_fixture = _load_fixture("nvidia_smi_query_gpu_index.txt")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        subprocess, "check_output", lambda *_a, **_kw: nvidia_smi_fixture
    )

    mock_llm = _make_mock_llm()

    def _fake_llm_init(*args, **kwargs):
        vllm_logger = logging.getLogger("vllm")
        vllm_logger.info("Loading model weights")
        vllm_logger.info("Model loaded successfully")
        return mock_llm

    _install_fake_vllm(monkeypatch, llm_side_effect=_fake_llm_init)

    launch = manager.launch(
        model="meta-llama/Llama-3.1-8B-Instruct",
        gpu_ids=None,
        task="generate",
    )

    _wait_until(lambda: manager.get_status(launch.launch_id).state == LaunchState.READY)

    logs = manager.read_logs(launch.launch_id, 0)
    assert "Loading model weights" in logs.content
    assert "Model loaded successfully" in logs.content


def test_list_launches_filters_terminal_states(
    monkeypatch: pytest.MonkeyPatch, manager: VLLMEnvironmentManager
) -> None:
    nvidia_smi_fixture = _load_fixture("nvidia_smi_query_gpu_index.txt")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        subprocess, "check_output", lambda *_a, **_kw: nvidia_smi_fixture
    )

    mock_llm = _make_mock_llm()
    _install_fake_vllm(monkeypatch, llm_side_effect=lambda *a, **kw: mock_llm)

    launch = manager.launch(
        model="meta-llama/Llama-3.1-8B-Instruct",
        gpu_ids=None,
        task="generate",
    )
    _wait_until(lambda: manager.get_status(launch.launch_id).state == LaunchState.READY)

    active = manager.list_launches()
    assert [snapshot.launch_id for snapshot in active] == [launch.launch_id]

    manager.stop(launch.launch_id)
    assert manager.list_launches() == []

    with_terminal = manager.list_launches(include_terminal=True)
    assert len(with_terminal) == 1
    assert with_terminal[0].state == LaunchState.STOPPED


def test_launch_honors_requested_gpus(
    monkeypatch: pytest.MonkeyPatch, manager: VLLMEnvironmentManager
) -> None:
    nvidia_smi_fixture = _load_fixture("nvidia_smi_query_gpu_index.txt")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        subprocess, "check_output", lambda *_a, **_kw: nvidia_smi_fixture
    )

    captured_env = {}
    mock_llm = _make_mock_llm()

    def _fake_llm_init(*args, **kwargs):
        captured_env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        return mock_llm

    _install_fake_vllm(monkeypatch, llm_side_effect=_fake_llm_init)

    first_launch = manager.launch(
        model="custom-model",
        gpu_ids=[1],
        task="generate",
    )
    assert first_launch.gpu_ids == [1]

    _wait_until(
        lambda: manager.get_status(first_launch.launch_id).state == LaunchState.READY
    )
    assert captured_env["CUDA_VISIBLE_DEVICES"] == "1"


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


def test_launch_with_embed_task(
    monkeypatch: pytest.MonkeyPatch, manager: VLLMEnvironmentManager
) -> None:
    nvidia_smi_fixture = _load_fixture("nvidia_smi_query_gpu_index.txt")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        subprocess, "check_output", lambda *_a, **_kw: nvidia_smi_fixture
    )

    mock_llm = _make_mock_llm()
    _install_fake_vllm(monkeypatch, llm_side_effect=lambda *a, **kw: mock_llm)

    launch = manager.launch(
        model="BAAI/bge-base-en-v1.5",
        gpu_ids=[0],
        task="embed",
    )
    assert launch.task == "embed"

    _wait_until(lambda: manager.get_status(launch.launch_id).state == LaunchState.READY)

    llm, task = manager.get_llm(launch.launch_id)
    assert task == "embed"
    assert llm is mock_llm


def test_get_llm_raises_for_not_ready(
    monkeypatch: pytest.MonkeyPatch, manager: VLLMEnvironmentManager
) -> None:
    nvidia_smi_fixture = _load_fixture("nvidia_smi_query_gpu_index.txt")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        subprocess, "check_output", lambda *_a, **_kw: nvidia_smi_fixture
    )

    block_event = threading.Event()
    mock_llm = _make_mock_llm()

    def _blocking_llm_init(*args, **kwargs):
        block_event.wait(timeout=10)
        return mock_llm

    _install_fake_vllm(monkeypatch, llm_side_effect=_blocking_llm_init)

    launch = manager.launch(
        model="some-model",
        gpu_ids=[0],
        task="generate",
    )
    assert launch.state == LaunchState.BOOTSTRAPPING

    with pytest.raises(LaunchConflictError, match="not ready"):
        manager.get_llm(launch.launch_id)

    block_event.set()


def test_llm_load_failure_sets_failed_state(
    monkeypatch: pytest.MonkeyPatch, manager: VLLMEnvironmentManager
) -> None:
    nvidia_smi_fixture = _load_fixture("nvidia_smi_query_gpu_index.txt")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        subprocess, "check_output", lambda *_a, **_kw: nvidia_smi_fixture
    )

    def _failing_llm_init(*args, **kwargs):
        raise RuntimeError("Model not found on disk")

    _install_fake_vllm(monkeypatch, llm_side_effect=_failing_llm_init)

    launch = manager.launch(
        model="fake-model",
        gpu_ids=None,
        task="generate",
    )

    _wait_until(
        lambda: manager.get_status(launch.launch_id).state == LaunchState.FAILED
    )

    status = manager.get_status(launch.launch_id)
    assert status.state == LaunchState.FAILED
    assert "Model not found on disk" in status.error


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
