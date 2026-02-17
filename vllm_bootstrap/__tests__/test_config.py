from __future__ import annotations

from pathlib import Path

import pytest

from vllm_bootstrap.config import load_settings


def test_load_settings_reads_access_key_and_grpc_port(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    log_dir = tmp_path / "custom-logs"
    monkeypatch.setenv("VLLM_BOOTSTRAP_LOG_DIR", str(log_dir))
    monkeypatch.setenv("VLLM_ACCESS_KEY", "  shared-key  ")
    monkeypatch.setenv("VLLM_GRPC_PORT", "9001")

    settings = load_settings()

    assert settings.log_dir == log_dir
    assert settings.access_key == "shared-key"
    assert settings.grpc_port == 9001
