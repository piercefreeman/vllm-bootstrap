from __future__ import annotations

import multiprocessing
import os
import subprocess
import time
from multiprocessing.connection import Connection
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
    return Settings(
        log_dir=tmp_path / "logs",
        stop_timeout_seconds=2.0,
        log_read_chunk_bytes=64 * 1024,
        grpc_port=8001,
    )


# -- Picklable fake worker functions (module-level for spawn compatibility) --


def _fake_worker(
    cmd_conn: Connection,
    log_queue: multiprocessing.Queue,
    gpu_ids: list[int],
    model: str,
    task: str,
    extra_kwargs: dict,
) -> None:
    """A fake worker that signals ready and handles generate/embed/shutdown."""
    cmd_conn.send(("ready",))
    while True:
        try:
            msg = cmd_conn.recv()
        except (EOFError, OSError):
            break
        cmd = msg[0]
        if cmd == "shutdown":
            break
        elif cmd == "generate":
            _, prompts, params_dict = msg
            results = [
                {"text": f"generated:{p}", "prompt_tokens": 3, "completion_tokens": 2}
                for p in prompts
            ]
            cmd_conn.send(("result", results))
        elif cmd == "embed":
            _, texts = msg
            results = [[0.1, 0.2, 0.3] for _ in texts]
            cmd_conn.send(("result", results))


def _fake_worker_with_logs(
    cmd_conn: Connection,
    log_queue: multiprocessing.Queue,
    gpu_ids: list[int],
    model: str,
    task: str,
    extra_kwargs: dict,
) -> None:
    """A fake worker that emits log lines before signaling ready."""
    log_queue.put("Loading model weights")
    log_queue.put("Model loaded successfully")
    cmd_conn.send(("ready",))
    while True:
        try:
            msg = cmd_conn.recv()
        except (EOFError, OSError):
            break
        if msg[0] == "shutdown":
            break


def _failing_worker(
    cmd_conn: Connection,
    log_queue: multiprocessing.Queue,
    gpu_ids: list[int],
    model: str,
    task: str,
    extra_kwargs: dict,
) -> None:
    """A fake worker that sends an error during startup."""
    cmd_conn.send(("error", "Model not found on disk"))


def _crash_worker(
    cmd_conn: Connection,
    log_queue: multiprocessing.Queue,
    gpu_ids: list[int],
    model: str,
    task: str,
    extra_kwargs: dict,
) -> None:
    """A fake worker that signals ready then exits (simulating crash)."""
    cmd_conn.send(("ready",))
    # Exit immediately without handling any commands


def _kwargs_capture_worker(
    cmd_conn: Connection,
    log_queue: multiprocessing.Queue,
    gpu_ids: list[int],
    model: str,
    task: str,
    extra_kwargs: dict,
) -> None:
    """A fake worker that captures its kwargs via log_queue for test inspection."""
    log_queue.put(f"model={model}")
    log_queue.put(f"task={task}")
    log_queue.put(f"gpu_ids={gpu_ids}")
    log_queue.put(f"tensor_parallel_size={len(gpu_ids)}")
    log_queue.put(f"extra_kwargs={extra_kwargs}")
    cmd_conn.send(("ready",))
    while True:
        try:
            msg = cmd_conn.recv()
        except (EOFError, OSError):
            break
        if msg[0] == "shutdown":
            break


# -- Tests --


@pytest.fixture
def manager(tmp_path: Path) -> VLLMEnvironmentManager:
    instance = VLLMEnvironmentManager(
        settings=_make_settings(tmp_path), _worker_fn=_fake_worker
    )
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


def test_logs_populated_from_subprocess(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    nvidia_smi_fixture = _load_fixture("nvidia_smi_query_gpu_index.txt")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        subprocess, "check_output", lambda *_a, **_kw: nvidia_smi_fixture
    )

    mgr = VLLMEnvironmentManager(
        settings=_make_settings(tmp_path), _worker_fn=_fake_worker_with_logs
    )
    try:
        launch = mgr.launch(
            model="meta-llama/Llama-3.1-8B-Instruct",
            gpu_ids=None,
            task="generate",
        )

        _wait_until(lambda: mgr.get_status(launch.launch_id).state == LaunchState.READY)

        # Give log drainer a moment to pick up the queue items
        _wait_until(
            lambda: (
                "Loading model weights" in mgr.read_logs(launch.launch_id, 0).content
            ),
            timeout_seconds=3.0,
        )

        logs = mgr.read_logs(launch.launch_id, 0)
        assert "Loading model weights" in logs.content
        assert "Model loaded successfully" in logs.content
    finally:
        mgr.stop_all()


def test_list_launches_filters_terminal_states(
    monkeypatch: pytest.MonkeyPatch, manager: VLLMEnvironmentManager
) -> None:
    nvidia_smi_fixture = _load_fixture("nvidia_smi_query_gpu_index.txt")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        subprocess, "check_output", lambda *_a, **_kw: nvidia_smi_fixture
    )

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
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    nvidia_smi_fixture = _load_fixture("nvidia_smi_query_gpu_index.txt")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        subprocess, "check_output", lambda *_a, **_kw: nvidia_smi_fixture
    )

    mgr = VLLMEnvironmentManager(
        settings=_make_settings(tmp_path), _worker_fn=_kwargs_capture_worker
    )
    try:
        first_launch = mgr.launch(
            model="custom-model",
            gpu_ids=[1],
            task="generate",
        )
        assert first_launch.gpu_ids == [1]

        _wait_until(
            lambda: mgr.get_status(first_launch.launch_id).state == LaunchState.READY
        )

        # The kwargs_capture_worker sends gpu_ids via log_queue
        _wait_until(
            lambda: "gpu_ids=[1]" in mgr.read_logs(first_launch.launch_id, 0).content,
            timeout_seconds=3.0,
        )
        logs = mgr.read_logs(first_launch.launch_id, 0)
        assert "gpu_ids=[1]" in logs.content
    finally:
        mgr.stop_all()


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
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    nvidia_smi_fixture = _load_fixture("nvidia_smi_query_gpu_index.txt")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        subprocess, "check_output", lambda *_a, **_kw: nvidia_smi_fixture
    )

    mgr = VLLMEnvironmentManager(
        settings=_make_settings(tmp_path), _worker_fn=_kwargs_capture_worker
    )
    try:
        launch = mgr.launch(
            model="BAAI/bge-base-en-v1.5",
            gpu_ids=[0],
            task="embed",
        )
        assert launch.task == "embed"

        _wait_until(lambda: mgr.get_status(launch.launch_id).state == LaunchState.READY)

        # Verify embed task was passed to worker
        _wait_until(
            lambda: "task=embed" in mgr.read_logs(launch.launch_id, 0).content,
            timeout_seconds=3.0,
        )
        logs = mgr.read_logs(launch.launch_id, 0)
        assert "task=embed" in logs.content
        assert "model=BAAI/bge-base-en-v1.5" in logs.content
    finally:
        mgr.stop_all()


def test_launch_generate_does_not_pass_runner_or_convert(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify that generate task does not set runner/convert kwargs."""
    nvidia_smi_fixture = _load_fixture("nvidia_smi_query_gpu_index.txt")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        subprocess, "check_output", lambda *_a, **_kw: nvidia_smi_fixture
    )

    mgr = VLLMEnvironmentManager(
        settings=_make_settings(tmp_path), _worker_fn=_kwargs_capture_worker
    )
    try:
        launch = mgr.launch(
            model="meta-llama/Llama-3.1-8B-Instruct",
            gpu_ids=[0],
            task="generate",
        )

        _wait_until(lambda: mgr.get_status(launch.launch_id).state == LaunchState.READY)

        _wait_until(
            lambda: "task=generate" in mgr.read_logs(launch.launch_id, 0).content,
            timeout_seconds=3.0,
        )
        logs = mgr.read_logs(launch.launch_id, 0)
        assert "task=generate" in logs.content
        assert "model=meta-llama/Llama-3.1-8B-Instruct" in logs.content
        assert "tensor_parallel_size=1" in logs.content
    finally:
        mgr.stop_all()


def test_generate_and_embed_not_ready(
    monkeypatch: pytest.MonkeyPatch, manager: VLLMEnvironmentManager
) -> None:
    """Test that generate/embed raise when launch is not ready."""
    nvidia_smi_fixture = _load_fixture("nvidia_smi_query_gpu_index.txt")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        subprocess, "check_output", lambda *_a, **_kw: nvidia_smi_fixture
    )

    # Use a worker that blocks so we can test the BOOTSTRAPPING state
    from vllm_bootstrap.manager import LaunchState, _LaunchRecord
    import threading

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


def test_llm_load_failure_sets_failed_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    nvidia_smi_fixture = _load_fixture("nvidia_smi_query_gpu_index.txt")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        subprocess, "check_output", lambda *_a, **_kw: nvidia_smi_fixture
    )

    mgr = VLLMEnvironmentManager(
        settings=_make_settings(tmp_path), _worker_fn=_failing_worker
    )
    try:
        launch = mgr.launch(
            model="fake-model",
            gpu_ids=None,
            task="generate",
        )

        _wait_until(
            lambda: mgr.get_status(launch.launch_id).state == LaunchState.FAILED
        )

        status = mgr.get_status(launch.launch_id)
        assert status.state == LaunchState.FAILED
        assert "Model not found on disk" in status.error
    finally:
        mgr.stop_all()


def test_subprocess_crash_detected_on_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that a subprocess crash is detected when sending a command."""
    nvidia_smi_fixture = _load_fixture("nvidia_smi_query_gpu_index.txt")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        subprocess, "check_output", lambda *_a, **_kw: nvidia_smi_fixture
    )

    mgr = VLLMEnvironmentManager(
        settings=_make_settings(tmp_path), _worker_fn=_crash_worker
    )
    try:
        launch = mgr.launch(
            model="some-model",
            gpu_ids=[0],
            task="generate",
        )

        _wait_until(lambda: mgr.get_status(launch.launch_id).state == LaunchState.READY)

        # The subprocess has exited, so sending a command should fail
        with pytest.raises(LaunchConflictError, match="crashed"):
            mgr.generate(launch.launch_id, ["hello"], {})

        status = mgr.get_status(launch.launch_id)
        assert status.state == LaunchState.FAILED
    finally:
        mgr.stop_all()


def test_generate_returns_results(
    monkeypatch: pytest.MonkeyPatch, manager: VLLMEnvironmentManager
) -> None:
    nvidia_smi_fixture = _load_fixture("nvidia_smi_query_gpu_index.txt")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        subprocess, "check_output", lambda *_a, **_kw: nvidia_smi_fixture
    )

    launch = manager.launch(
        model="test-model",
        gpu_ids=[0],
        task="generate",
    )
    _wait_until(lambda: manager.get_status(launch.launch_id).state == LaunchState.READY)

    results = manager.generate(launch.launch_id, ["hello", "world"], {})
    assert len(results) == 2
    assert results[0]["text"] == "generated:hello"
    assert results[1]["text"] == "generated:world"
    assert results[0]["prompt_tokens"] == 3
    assert results[0]["completion_tokens"] == 2


def test_embed_returns_results(
    monkeypatch: pytest.MonkeyPatch, manager: VLLMEnvironmentManager
) -> None:
    nvidia_smi_fixture = _load_fixture("nvidia_smi_query_gpu_index.txt")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        subprocess, "check_output", lambda *_a, **_kw: nvidia_smi_fixture
    )

    launch = manager.launch(
        model="test-model",
        gpu_ids=[0],
        task="embed",
    )
    _wait_until(lambda: manager.get_status(launch.launch_id).state == LaunchState.READY)

    results = manager.embed(launch.launch_id, ["hello", "world"])
    assert len(results) == 2
    assert results[0] == [0.1, 0.2, 0.3]
    assert results[1] == [0.1, 0.2, 0.3]


def test_generate_rejects_embed_task(
    monkeypatch: pytest.MonkeyPatch, manager: VLLMEnvironmentManager
) -> None:
    nvidia_smi_fixture = _load_fixture("nvidia_smi_query_gpu_index.txt")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        subprocess, "check_output", lambda *_a, **_kw: nvidia_smi_fixture
    )

    launch = manager.launch(
        model="test-model",
        gpu_ids=[0],
        task="embed",
    )
    _wait_until(lambda: manager.get_status(launch.launch_id).state == LaunchState.READY)

    with pytest.raises(LaunchConflictError, match="expected 'generate'"):
        manager.generate(launch.launch_id, ["hello"], {})


def test_embed_rejects_generate_task(
    monkeypatch: pytest.MonkeyPatch, manager: VLLMEnvironmentManager
) -> None:
    nvidia_smi_fixture = _load_fixture("nvidia_smi_query_gpu_index.txt")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        subprocess, "check_output", lambda *_a, **_kw: nvidia_smi_fixture
    )

    launch = manager.launch(
        model="test-model",
        gpu_ids=[0],
        task="generate",
    )
    _wait_until(lambda: manager.get_status(launch.launch_id).state == LaunchState.READY)

    with pytest.raises(LaunchConflictError, match="expected 'embed'"):
        manager.embed(launch.launch_id, ["hello"])


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
