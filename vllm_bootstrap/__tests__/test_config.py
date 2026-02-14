from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from vllm_bootstrap.config import DEFAULT_READY_MARKERS, load_settings


def test_load_settings_reads_access_key_and_ready_markers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    log_dir = tmp_path / "custom-logs"
    monkeypatch.setenv("VLLM_BOOTSTRAP_LOG_DIR", str(log_dir))
    monkeypatch.setenv("VLLM_READY_MARKERS", "marker-a| marker-b ")
    monkeypatch.setenv("VLLM_ACCESS_KEY", "  shared-key  ")

    settings = load_settings()

    assert settings.log_dir == log_dir
    assert settings.ready_markers == ("marker-a", "marker-b")
    assert settings.access_key == "shared-key"


def test_load_settings_uses_default_ready_markers_when_env_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_READY_MARKERS", "  |  ")

    settings = load_settings()

    assert settings.ready_markers == DEFAULT_READY_MARKERS


def test_load_settings_validates_port_range(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_LAUNCH_PORT_START", "9000")
    monkeypatch.setenv("VLLM_LAUNCH_PORT_END", "8000")

    with pytest.raises(
        ValidationError, match="VLLM_LAUNCH_PORT_START must be <= VLLM_LAUNCH_PORT_END"
    ):
        load_settings()
