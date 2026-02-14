from __future__ import annotations

import base64

from vllm_bootstrap.auth import _extract_access_key


def test_extract_access_key_returns_none_without_header() -> None:
    assert _extract_access_key(None) is None


def test_extract_access_key_parses_bearer_token() -> None:
    assert _extract_access_key("Bearer secret-token") == "secret-token"


def test_extract_access_key_parses_basic_password() -> None:
    token = base64.b64encode(b"user:secret-token").decode()
    assert _extract_access_key(f"Basic {token}") == "secret-token"


def test_extract_access_key_rejects_invalid_basic_payload() -> None:
    assert _extract_access_key("Basic not-base64") is None


def test_extract_access_key_rejects_basic_without_separator() -> None:
    token = base64.b64encode(b"just-user").decode()
    assert _extract_access_key(f"Basic {token}") is None


def test_extract_access_key_rejects_unsupported_scheme() -> None:
    assert _extract_access_key("Digest abc123") is None
