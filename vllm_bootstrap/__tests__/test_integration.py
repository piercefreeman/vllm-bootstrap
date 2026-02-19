"""Integration tests that run against real vLLM models.

These tests require vLLM to be installed (CPU or GPU build) and are
automatically skipped when vLLM is not available — e.g. when running
``make test`` outside Docker.
"""

from __future__ import annotations

import os
import signal
import time
from pathlib import Path

import pytest

vllm = pytest.importorskip("vllm")

from vllm_bootstrap.config import Settings  # noqa: E402
from vllm_bootstrap.manager import (  # noqa: E402
    LaunchConflictError,
    LaunchState,
    VLLMEnvironmentManager,
)

GENERATE_MODEL = "facebook/opt-125m"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
LOAD_TIMEOUT = 120  # CPU model loading can be slow


def _make_settings(tmp_path: Path) -> Settings:
    return Settings(
        log_dir=tmp_path / "logs",
        stop_timeout_seconds=10.0,
        log_read_chunk_bytes=64 * 1024,
        grpc_port=8001,
    )


def _wait_until(
    predicate, timeout_seconds: float = LOAD_TIMEOUT, interval_seconds: float = 0.5
) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(interval_seconds)
    raise AssertionError("condition was not met before timeout")


# ---------------------------------------------------------------------------
# Session-scoped fixtures — load each model once for the entire test run
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def session_tmp(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("integration")


@pytest.fixture(scope="session")
def generate_launch(session_tmp: Path):
    """Launch a real generate model and wait until READY."""
    mgr = VLLMEnvironmentManager(settings=_make_settings(session_tmp / "gen"))
    launch = mgr.launch(model=GENERATE_MODEL, gpu_ids=[0], task="generate")
    _wait_until(lambda: mgr.get_status(launch.launch_id).state == LaunchState.READY)
    yield mgr, launch
    mgr.stop_all()


@pytest.fixture(scope="session")
def embed_launch(session_tmp: Path):
    """Launch a real embedding model and wait until READY."""
    mgr = VLLMEnvironmentManager(settings=_make_settings(session_tmp / "emb"))
    launch = mgr.launch(model=EMBED_MODEL, gpu_ids=[0], task="embed")
    _wait_until(lambda: mgr.get_status(launch.launch_id).state == LaunchState.READY)
    yield mgr, launch
    mgr.stop_all()


# ---------------------------------------------------------------------------
# Generate tests
# ---------------------------------------------------------------------------


def test_generate_returns_results(generate_launch) -> None:
    mgr, launch = generate_launch
    results = mgr.generate(launch.launch_id, ["Hello world", "How are you?"], {})
    assert len(results) == 2
    for r in results:
        assert isinstance(r["text"], str)
        assert len(r["text"]) > 0
        assert r["prompt_tokens"] > 0
        assert r["completion_tokens"] > 0


def test_generate_rejects_embed_task(generate_launch) -> None:
    mgr, launch = generate_launch
    with pytest.raises(LaunchConflictError, match="expected 'embed'"):
        mgr.embed(launch.launch_id, ["hello"])


# ---------------------------------------------------------------------------
# Embed tests
# ---------------------------------------------------------------------------


def test_embed_returns_results(embed_launch) -> None:
    mgr, launch = embed_launch
    results = mgr.embed(launch.launch_id, ["Hello world", "How are you?"])
    assert len(results) == 2
    for vec in results:
        assert isinstance(vec, list)
        assert len(vec) > 0
        assert all(isinstance(v, float) for v in vec)


def test_embed_rejects_generate_task(embed_launch) -> None:
    mgr, launch = embed_launch
    with pytest.raises(LaunchConflictError, match="expected 'generate'"):
        mgr.generate(launch.launch_id, ["hello"], {})


# ---------------------------------------------------------------------------
# Streaming tests
# ---------------------------------------------------------------------------


def test_generate_stream_returns_results(generate_launch) -> None:
    mgr, launch = generate_launch
    items = list(
        mgr.generate_stream(launch.launch_id, ["Tell me a story"], {"max_tokens": 16})
    )
    assert len(items) == 1
    assert isinstance(items[0]["text"], str)
    assert len(items[0]["text"]) > 0


def test_generate_stream_batch(generate_launch) -> None:
    mgr, launch = generate_launch
    prompts = ["Hello", "World", "Foo"]
    items = list(mgr.generate_stream(launch.launch_id, prompts, {"max_tokens": 8}))
    assert len(items) == len(prompts)
    for item in items:
        assert "text" in item
        assert "prompt_tokens" in item
        assert "completion_tokens" in item


def test_embed_stream_returns_results(embed_launch) -> None:
    mgr, launch = embed_launch
    items = list(mgr.embed_stream(launch.launch_id, ["Hello", "World"]))
    assert len(items) == 2
    for vec in items:
        assert isinstance(vec, list)
        assert len(vec) > 0


# ---------------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------------


def test_logs_contain_vllm_output(generate_launch) -> None:
    mgr, launch = generate_launch
    logs = mgr.read_logs(launch.launch_id, 0)
    # vLLM always prints something during model loading
    assert len(logs.content) > 0


# ---------------------------------------------------------------------------
# Launch status / metadata
# ---------------------------------------------------------------------------


def test_launch_status(generate_launch) -> None:
    mgr, launch = generate_launch
    status = mgr.get_status(launch.launch_id)
    assert status.state == LaunchState.READY
    assert status.model == GENERATE_MODEL
    assert status.gpu_ids == [0]
    assert status.task == "generate"


def test_embed_launch_metadata(embed_launch) -> None:
    mgr, launch = embed_launch
    status = mgr.get_status(launch.launch_id)
    assert status.state == LaunchState.READY
    assert status.model == EMBED_MODEL
    assert status.task == "embed"


# ---------------------------------------------------------------------------
# GPU conflict
# ---------------------------------------------------------------------------


def test_gpu_conflict_after_launch(generate_launch) -> None:
    mgr, launch = generate_launch
    with pytest.raises(LaunchConflictError):
        mgr.launch(model="another-model", gpu_ids=[0], task="generate")


# ---------------------------------------------------------------------------
# Stop lifecycle
# ---------------------------------------------------------------------------


def test_stop_transitions_to_stopped(session_tmp: Path) -> None:
    """Launch a model, stop it, verify STOPPED state."""
    mgr = VLLMEnvironmentManager(settings=_make_settings(session_tmp / "stop_test"))
    launch = mgr.launch(model=GENERATE_MODEL, gpu_ids=[0], task="generate")
    _wait_until(lambda: mgr.get_status(launch.launch_id).state == LaunchState.READY)

    stopped = mgr.stop(launch.launch_id)
    assert stopped.state == LaunchState.STOPPED

    # Should no longer appear in active list
    assert mgr.list_launches() == []
    # But should appear with include_terminal
    terminal = mgr.list_launches(include_terminal=True)
    assert len(terminal) == 1
    assert terminal[0].state == LaunchState.STOPPED


# ---------------------------------------------------------------------------
# Failure cases
# ---------------------------------------------------------------------------


def test_invalid_model_sets_failed_state(session_tmp: Path) -> None:
    mgr = VLLMEnvironmentManager(settings=_make_settings(session_tmp / "invalid_model"))
    launch = mgr.launch(
        model="this-model-definitely-does-not-exist-anywhere/fake",
        gpu_ids=[0],
        task="generate",
    )
    _wait_until(
        lambda: mgr.get_status(launch.launch_id).state == LaunchState.FAILED,
        timeout_seconds=60,
    )
    status = mgr.get_status(launch.launch_id)
    assert status.state == LaunchState.FAILED
    assert status.error is not None
    mgr.stop_all()


def test_subprocess_crash_detected(session_tmp: Path) -> None:
    """Kill the worker process and verify the manager detects the crash."""
    mgr = VLLMEnvironmentManager(settings=_make_settings(session_tmp / "crash_test"))
    launch = mgr.launch(model=GENERATE_MODEL, gpu_ids=[0], task="generate")
    _wait_until(lambda: mgr.get_status(launch.launch_id).state == LaunchState.READY)

    # Find and kill the worker process
    record = mgr._launches[launch.launch_id]
    worker_pid = record._process.pid
    os.kill(worker_pid, signal.SIGKILL)

    # Next command should detect the crash
    with pytest.raises(LaunchConflictError, match="crashed"):
        mgr.generate(launch.launch_id, ["hello"], {})

    status = mgr.get_status(launch.launch_id)
    assert status.state == LaunchState.FAILED
    mgr.stop_all()
