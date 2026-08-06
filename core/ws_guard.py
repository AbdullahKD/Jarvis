"""
WebSocket admission control: origin checking, token auth, frame validation.

Pure functions, no FastAPI import, so they can be tested without standing up
the server. ``server.py`` is 3,264 lines and imports the entire application —
anything left inside it is untestable by construction, which is how both of
the holes below survived.

**Why this exists.** Starlette's ``BaseHTTPMiddleware`` only intercepts scope
type ``"http"``. The ``/ws`` endpoint never reached ``_BasicAuthMiddleware``,
so wherever ``JARVIS_AUTH_PASSWORD`` was set — exactly the setup where auth
matters, e.g. the server reachable over a tailnet or LAN — the HTTP surface was
locked and the WebSocket beside it was an open, full-capability channel: read
inbox, send email, drive mac_control, run file operations.

Separately, and more relevant to a local install: **browsers do not apply the
same-origin policy to WebSocket connections.** With no password set at all, any
page the user happened to visit could open ``ws://localhost:8000/ws`` and talk
to Jarvis. That's cross-site WebSocket hijacking, and origin checking is the
fix.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import secrets
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

# Frames larger than this are refused. Without a cap one client can drive the
# process out of memory with a single message.
DEFAULT_MAX_FRAME_BYTES = 256 * 1024

# Close codes
CLOSE_POLICY_VIOLATION = 1008
CLOSE_TOO_LARGE = 1009


# ── Token ───────────────────────────────────────────────────────────────────


def derive_ws_token(password: str) -> str:
    """Token derived from the configured password.

    Deterministic, so the UI can fetch it once per page load and the server
    keeps no session state. It is not a bearer secret in its own right: it's
    only obtainable from ``GET /ws-token``, which *is* behind Basic Auth, so
    holding it already implies the ability to authenticate over HTTP.

    Domain-separated (``jarvis-ws``) so this value can never be confused with
    any other HMAC of the same key.
    """
    if not password:
        return ""
    return hmac.new(password.encode(), b"jarvis-ws", hashlib.sha256).hexdigest()


# ── Origin ──────────────────────────────────────────────────────────────────


def origin_allowed(origin: Optional[str], allowed: Iterable[str]) -> bool:
    """Whether a handshake's Origin header is acceptable.

    A missing Origin means a non-browser client (curl, the CLI, the test
    suite). Those are allowed through here and still face the token check —
    rejecting them would break every scripted use. The header is only
    trustworthy *because* browsers set it and won't let script override it,
    which is precisely the attacker we're guarding against.
    """
    if not origin:
        return True
    normalised = origin.rstrip("/").lower()
    return normalised in {o.rstrip("/").lower() for o in allowed if o}


# ── Authentication ──────────────────────────────────────────────────────────


def authenticate(
    *,
    password: str,
    user: str = "admin",
    query_token: Optional[str] = None,
    authorization_header: Optional[str] = None,
) -> bool:
    """Whether a handshake may proceed.

    No password configured → open, matching the HTTP middleware's behaviour so
    local development is unauthenticated as before.
    """
    if not password:
        return True

    expected = derive_ws_token(password)
    if query_token and secrets.compare_digest(query_token, expected):
        return True

    # Non-browser clients can present Basic credentials directly; browsers
    # cannot set headers on a WebSocket handshake, which is why the token
    # path exists at all.
    if authorization_header and authorization_header.startswith("Basic "):
        try:
            decoded = base64.b64decode(
                authorization_header[6:], validate=True
            ).decode("utf-8", errors="ignore")
        except (binascii.Error, ValueError):
            return False
        supplied_user, sep, supplied_pw = decoded.partition(":")
        if not sep:
            return False
        return (secrets.compare_digest(supplied_user, user)
                and secrets.compare_digest(supplied_pw, password))

    return False


# ── Frame validation ────────────────────────────────────────────────────────


@dataclass(slots=True)
class FrameResult:
    """Outcome of validating one inbound frame."""

    message: Optional[str] = None
    error: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.message is not None

    @property
    def ignore(self) -> bool:
        """Valid frame carrying nothing to do (e.g. an empty message)."""
        return self.error is None and self.message is None


def validate_frame(data: str, *,
                   max_bytes: int = DEFAULT_MAX_FRAME_BYTES) -> FrameResult:
    """Parse and validate an inbound WebSocket frame.

    The version this replaces called ``json.loads`` outside every try block,
    with an outer handler catching only ``(WebSocketDisconnect, RuntimeError)``.
    A non-JSON frame killed the session with ``JSONDecodeError``; JSON that
    wasn't an object (``[]``, ``"hi"``, ``5``) killed it with ``AttributeError``
    on ``.get()``. Either was a one-line denial of service.
    """
    if len(data.encode("utf-8", errors="ignore")) > max_bytes:
        return FrameResult(error=f"Message too large (limit {max_bytes // 1024} KB).")

    try:
        payload = json.loads(data)
    except json.JSONDecodeError as exc:
        return FrameResult(error=f"Malformed JSON: {exc.msg}")

    if not isinstance(payload, dict):
        return FrameResult(
            error='Expected a JSON object, e.g. {"message": "..."}.')

    message = payload.get("message", "")
    if not isinstance(message, str):
        return FrameResult(error="'message' must be a string.")

    if not message.strip():
        return FrameResult(payload=payload)      # nothing to do, not an error

    return FrameResult(message=message, payload=payload)


__all__ = [
    "derive_ws_token", "origin_allowed", "authenticate", "validate_frame",
    "FrameResult", "DEFAULT_MAX_FRAME_BYTES",
    "CLOSE_POLICY_VIOLATION", "CLOSE_TOO_LARGE",
]
