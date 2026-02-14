from __future__ import annotations

import base64
import binascii
import secrets
from collections.abc import Awaitable, Callable

from fastapi import Request
from fastapi.responses import JSONResponse, Response

AccessKeyGetter = Callable[[], str | None]
CallNext = Callable[[Request], Awaitable[Response]]


def _extract_access_key(authorization_header: str | None) -> str | None:
    if not authorization_header:
        return None

    scheme, _, credential = authorization_header.partition(" ")
    if not credential:
        return None

    normalized_scheme = scheme.lower()
    if normalized_scheme == "bearer":
        token = credential.strip()
        return token or None

    if normalized_scheme != "basic":
        return None

    try:
        decoded = base64.b64decode(credential, validate=True).decode("utf-8")
    except (UnicodeDecodeError, ValueError, binascii.Error):
        return None

    has_separator = ":" in decoded
    _, _, password = decoded.partition(":")
    if not has_separator:
        return None

    normalized_password = password.strip()
    return normalized_password or None


def _unauthorized_response() -> JSONResponse:
    return JSONResponse(
        status_code=401,
        content={"detail": "Unauthorized"},
        headers={"WWW-Authenticate": 'Basic realm="vllm-bootstrap"'},
    )


def build_access_key_middleware(*, access_key_getter: AccessKeyGetter):
    async def enforce_access_key(request: Request, call_next: CallNext) -> Response:
        configured_key = access_key_getter()
        if not configured_key:
            return await call_next(request)

        provided_key = _extract_access_key(request.headers.get("authorization"))
        if provided_key is None:
            return _unauthorized_response()

        if not secrets.compare_digest(provided_key, configured_key):
            return _unauthorized_response()

        return await call_next(request)

    return enforce_access_key
