from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

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
    base_port = 42000 + (os.getpid() % 1000)
    return Settings(
        launch_host="127.0.0.1",
        launch_port_start=base_port,
        launch_port_end=base_port + 100,
        log_dir=tmp_path / "logs",
        stop_timeout_seconds=2.0,
        log_read_chunk_bytes=64 * 1024,
        ready_scan_chunk_bytes=64 * 1024,
        ready_markers=("READY_MARKER",),
    )


def _fake_server_command(*, extra_lines: list[str]) -> list[str]:
    lines_literal = ", ".join(repr(line) for line in extra_lines)
    script = (
        "import os, signal, sys, time\n"
        "def _term(*_):\n"
        "    sys.exit(0)\n"
        "signal.signal(signal.SIGTERM, _term)\n"
        "signal.signal(signal.SIGINT, _term)\n"
        'print(f\'CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES", "")}\', flush=True)\n'
        "print('READY_MARKER', flush=True)\n"
        f"for _line in [{lines_literal}]:\n"
        "    print(_line, flush=True)\n"
        "while True:\n"
        "    time.sleep(0.1)\n"
    )
    return [sys.executable, "-u", "-c", script]


def _vllm_error_server_command() -> list[str]:
    fixture_path = FIXTURES_DIR / "vllm_startup_error.txt"
    script = (
        "import sys\n"
        f"data = open({str(fixture_path)!r}).read()\n"
        "sys.stdout.write(data)\n"
        "sys.stdout.flush()\n"
        "sys.exit(1)\n"
    )
    return [sys.executable, "-u", "-c", script]


def _split_marker_server_command() -> list[str]:
    script = (
        "import signal, sys, time\n"
        "def _term(*_):\n"
        "    sys.exit(0)\n"
        "signal.signal(signal.SIGTERM, _term)\n"
        "sys.stdout.write('ABC')\n"
        "sys.stdout.flush()\n"
        "time.sleep(0.05)\n"
        "sys.stdout.write('DEFG\\n')\n"
        "sys.stdout.flush()\n"
        "while True:\n"
        "    time.sleep(0.1)\n"
    )
    return [sys.executable, "-u", "-c", script]


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
    monkeypatch.setattr(subprocess, "check_output", lambda *_a, **_kw: nvidia_smi_fixture)
    monkeypatch.setattr(
        manager,
        "_build_command",
        lambda **_: _fake_server_command(extra_lines=["bootstrapping complete"]),
    )

    first_launch = manager.launch(
        model="meta-llama/Llama-3.1-8B-Instruct",
        gpu_ids=None,
        port=None,
        extra_args=[],
    )
    assert first_launch.gpu_ids == [0, 1]

    _wait_until(
        lambda: manager.get_status(first_launch.launch_id).state == LaunchState.READY
    )

    with pytest.raises(LaunchConflictError):
        manager.launch(
            model="meta-llama/Llama-3.1-8B-Instruct",
            gpu_ids=None,
            port=None,
            extra_args=[],
        )

    stopped = manager.stop(first_launch.launch_id)
    assert stopped.state == LaunchState.STOPPED


def test_logs_follow_offset(
    monkeypatch: pytest.MonkeyPatch, manager: VLLMEnvironmentManager
) -> None:
    nvidia_smi_fixture = _load_fixture("nvidia_smi_query_gpu_index.txt")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(subprocess, "check_output", lambda *_a, **_kw: nvidia_smi_fixture)
    monkeypatch.setattr(
        manager,
        "_build_command",
        lambda **_: _fake_server_command(extra_lines=["line-one", "line-two"]),
    )

    launch = manager.launch(
        model="meta-llama/Llama-3.1-8B-Instruct",
        gpu_ids=None,
        port=None,
        extra_args=[],
    )

    def _has_log_content() -> bool:
        return "line-one" in manager.read_logs(launch.launch_id, 0).content

    _wait_until(_has_log_content)

    first_chunk = manager.read_logs(launch.launch_id, 0)
    second_chunk = manager.read_logs(launch.launch_id, first_chunk.next_offset)
    assert "line-one" in first_chunk.content
    assert second_chunk.content == ""


def test_launch_honors_requested_gpus_and_port(
    monkeypatch: pytest.MonkeyPatch, manager: VLLMEnvironmentManager
) -> None:
    nvidia_smi_fixture = _load_fixture("nvidia_smi_query_gpu_index.txt")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(subprocess, "check_output", lambda *_a, **_kw: nvidia_smi_fixture)
    monkeypatch.setattr(
        manager,
        "_build_command",
        lambda **_: _fake_server_command(extra_lines=["using selected gpu ids"]),
    )

    first_launch = manager.launch(
        model="custom-model",
        gpu_ids=[1],
        port=43123,
        extra_args=[],
    )
    assert first_launch.gpu_ids == [1]
    assert first_launch.port == 43123

    _wait_until(
        lambda: manager.get_status(first_launch.launch_id).state == LaunchState.READY
    )
    logs = manager.read_logs(first_launch.launch_id, 0)
    assert "CUDA_VISIBLE_DEVICES=1" in logs.content

    with pytest.raises(LaunchConflictError):
        manager.launch(
            model="other-model",
            gpu_ids=[0],
            port=43123,
            extra_args=[],
        )


def test_ready_marker_detected_across_log_chunks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _make_settings(tmp_path)
    settings = Settings(
        launch_host=settings.launch_host,
        launch_port_start=settings.launch_port_start,
        launch_port_end=settings.launch_port_end,
        log_dir=settings.log_dir,
        stop_timeout_seconds=settings.stop_timeout_seconds,
        log_read_chunk_bytes=settings.log_read_chunk_bytes,
        ready_scan_chunk_bytes=3,
        ready_markers=("ABCDEFG",),
    )
    manager = VLLMEnvironmentManager(settings=settings)

    try:
        nvidia_smi_fixture = _load_fixture("nvidia_smi_query_gpu_index.txt")
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        monkeypatch.setattr(subprocess, "check_output", lambda *_a, **_kw: nvidia_smi_fixture)
        monkeypatch.setattr(
            manager, "_build_command", lambda **_: _split_marker_server_command()
        )

        launch = manager.launch(
            model="meta-llama/Llama-3.1-8B-Instruct",
            gpu_ids=None,
            port=None,
            extra_args=[],
        )
        _wait_until(
            lambda: manager.get_status(launch.launch_id).state == LaunchState.READY
        )
    finally:
        manager.stop_all()


def test_launch_requires_model(
    monkeypatch: pytest.MonkeyPatch, manager: VLLMEnvironmentManager
) -> None:
    nvidia_smi_fixture = _load_fixture("nvidia_smi_query_gpu_index.txt")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(subprocess, "check_output", lambda *_a, **_kw: nvidia_smi_fixture)
    with pytest.raises(LaunchValidationError):
        manager.launch(model="   ", gpu_ids=None, port=None, extra_args=[])


def test_discover_gpu_ids_parses_real_nvidia_smi_output(
    monkeypatch: pytest.MonkeyPatch, manager: VLLMEnvironmentManager
) -> None:
    nvidia_smi_fixture = _load_fixture("nvidia_smi_query_gpu_index.txt")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(subprocess, "check_output", lambda *_a, **_kw: nvidia_smi_fixture)

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


def test_vllm_error_output_triggers_failed_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    base_port = 42000 + (os.getpid() % 1000)
    settings = Settings(
        launch_host="127.0.0.1",
        launch_port_start=base_port,
        launch_port_end=base_port + 100,
        log_dir=tmp_path / "logs",
        stop_timeout_seconds=2.0,
        log_read_chunk_bytes=64 * 1024,
        ready_scan_chunk_bytes=64 * 1024,
        ready_markers=("Application startup complete", "Uvicorn running on"),
    )
    manager = VLLMEnvironmentManager(settings=settings)

    try:
        nvidia_smi_fixture = _load_fixture("nvidia_smi_query_gpu_index.txt")
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        monkeypatch.setattr(subprocess, "check_output", lambda *_a, **_kw: nvidia_smi_fixture)
        monkeypatch.setattr(
            manager,
            "_build_command",
            lambda **_: _vllm_error_server_command(),
        )

        launch = manager.launch(
            model="fake-model",
            gpu_ids=None,
            port=None,
            extra_args=[],
        )

        _wait_until(
            lambda: manager.get_status(launch.launch_id).state == LaunchState.FAILED
        )

        status = manager.get_status(launch.launch_id)
        assert status.state == LaunchState.FAILED
        assert status.return_code != 0

        logs = manager.read_logs(launch.launch_id, 0)
        assert "OSError" in logs.content
        assert "fake-model" in logs.content
    finally:
        manager.stop_all()
