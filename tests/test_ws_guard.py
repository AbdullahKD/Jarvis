"""Tests for core.ws_guard — WebSocket admission control.

Covers audit items 1.3 (the /ws endpoint bypassed Basic Auth entirely, and
accepted cross-origin handshakes) and 1.4 (a single malformed frame killed the
session).
"""

from __future__ import annotations

import base64
import json

import pytest

from core.ws_guard import (
    DEFAULT_MAX_FRAME_BYTES,
    authenticate,
    derive_ws_token,
    origin_allowed,
    validate_frame,
)

PASSWORD = "hunter2-correct-horse"
ALLOWED = ["http://localhost:8000", "http://127.0.0.1:8000"]


def _basic(user: str, pw: str) -> str:
    return "Basic " + base64.b64encode(f"{user}:{pw}".encode()).decode()


# ── Token derivation ────────────────────────────────────────────────────────


def test_token_is_deterministic_and_password_bound():
    assert derive_ws_token(PASSWORD) == derive_ws_token(PASSWORD)
    assert derive_ws_token(PASSWORD) != derive_ws_token(PASSWORD + "x")


def test_token_does_not_leak_the_password():
    tok = derive_ws_token(PASSWORD)
    assert PASSWORD not in tok
    assert len(tok) == 64          # sha256 hex


def test_no_password_yields_no_token():
    assert derive_ws_token("") == ""


# ── Origin checking (cross-site WebSocket hijacking) ────────────────────────


@pytest.mark.parametrize("origin", ALLOWED + [
    "http://localhost:8000/",          # trailing slash
    "HTTP://LOCALHOST:8000",           # case
])
def test_allowed_origins_pass(origin):
    assert origin_allowed(origin, ALLOWED) is True


@pytest.mark.parametrize("origin", [
    "https://evil.example",
    "http://localhost:9999",
    "http://localhost",                # no port — a different origin
    "null",                            # sandboxed iframe / file://
    "http://localhost.evil.example",   # suffix trick
])
def test_foreign_origins_are_rejected(origin):
    """Browsers don't apply same-origin policy to WebSocket, so without this
    any page the user visits can drive Jarvis on localhost."""
    assert origin_allowed(origin, ALLOWED) is False


def test_missing_origin_is_allowed_for_non_browser_clients():
    """curl, the CLI and the test suite send no Origin. Rejecting them would
    break every scripted use — and they still face the token check."""
    assert origin_allowed(None, ALLOWED) is True
    assert origin_allowed("", ALLOWED) is True


# ── Authentication ──────────────────────────────────────────────────────────


def test_no_password_configured_leaves_the_socket_open():
    """Matches the HTTP middleware, so local dev is unauthenticated as before."""
    assert authenticate(password="") is True
    assert authenticate(password="", query_token="nonsense") is True


def test_valid_token_is_accepted():
    assert authenticate(password=PASSWORD,
                        query_token=derive_ws_token(PASSWORD)) is True


@pytest.mark.parametrize("token", [None, "", "wrong", "0" * 64])
def test_missing_or_wrong_token_is_rejected(token):
    """The live hole: with a password set, /ws accepted anyone."""
    assert authenticate(password=PASSWORD, query_token=token) is False


def test_basic_credentials_are_accepted_for_non_browser_clients():
    assert authenticate(password=PASSWORD, user="admin",
                        authorization_header=_basic("admin", PASSWORD)) is True


@pytest.mark.parametrize("header", [
    _basic("admin", "wrong"),
    _basic("wronguser", PASSWORD),
    "Basic !!!not-base64!!!",
    "Bearer " + derive_ws_token(PASSWORD),   # wrong scheme
    "Basic " + base64.b64encode(b"no-colon-here").decode(),
    "",
])
def test_bad_authorization_headers_are_rejected(header):
    assert authenticate(password=PASSWORD, user="admin",
                        authorization_header=header) is False


def test_token_for_a_different_password_is_rejected():
    assert authenticate(password=PASSWORD,
                        query_token=derive_ws_token("other")) is False


# ── Frame validation ────────────────────────────────────────────────────────


def test_valid_frame():
    r = validate_frame(json.dumps({"message": "hello"}))
    assert r.ok
    assert r.message == "hello"
    assert r.error is None


@pytest.mark.parametrize("raw", [
    "not json at all",
    "{",
    '{"message": }',
    "",
])
def test_malformed_json_is_an_error_not_a_crash(raw):
    """This used to raise JSONDecodeError past the outer handler and drop the
    connection — a one-line denial of service."""
    r = validate_frame(raw)
    assert r.error is not None
    assert r.message is None
    assert "JSON" in r.error


@pytest.mark.parametrize("raw", ["[]", '"hi"', "5", "true", "null"])
def test_non_object_json_is_an_error(raw):
    """`payload.get(...)` on a list or string raised AttributeError and killed
    the session the same way."""
    r = validate_frame(raw)
    assert r.error is not None
    assert "JSON object" in r.error


def test_non_string_message_is_rejected():
    r = validate_frame(json.dumps({"message": {"nested": "object"}}))
    assert r.error == "'message' must be a string."


@pytest.mark.parametrize("msg", ["", "   ", "\n\t "])
def test_empty_message_is_ignored_not_an_error(msg):
    r = validate_frame(json.dumps({"message": msg}))
    assert r.ignore is True
    assert r.error is None


def test_missing_message_key_is_ignored():
    r = validate_frame(json.dumps({"something_else": 1}))
    assert r.ignore is True


def test_oversized_frame_is_refused():
    big = json.dumps({"message": "x" * (DEFAULT_MAX_FRAME_BYTES + 1000)})
    r = validate_frame(big)
    assert r.error is not None
    assert "too large" in r.error


def test_frame_size_limit_is_configurable():
    payload = json.dumps({"message": "x" * 200})
    assert validate_frame(payload, max_bytes=10).error is not None
    assert validate_frame(payload, max_bytes=100_000).ok


def test_size_is_measured_in_bytes_not_characters():
    """Browsers send raw UTF-8, so an emoji costs 4 bytes and 1 character.
    Counting characters would let a 4x-larger payload through a
    byte-denominated limit. (json.dumps escapes non-ASCII by default, which is
    NOT what arrives over the wire — hence ensure_ascii=False here.)"""
    payload = json.dumps({"message": "😀" * 20_000}, ensure_ascii=False)
    assert len(payload) < 50_000                       # characters
    assert len(payload.encode("utf-8")) > 80_000       # bytes
    assert validate_frame(payload, max_bytes=50_000).error is not None
    assert validate_frame(payload, max_bytes=200_000).ok


def test_payload_is_returned_for_valid_frames():
    r = validate_frame(json.dumps({"message": "hi", "session_id": "s1"}))
    assert r.payload["session_id"] == "s1"


# ── The combination that was actually exploitable ───────────────────────────


def test_authenticated_deployment_rejects_a_browser_from_another_site():
    """Password set, attacker page open in the user's browser: the Origin is
    theirs, and they have no token. Both gates must fail closed."""
    assert origin_allowed("https://evil.example", ALLOWED) is False
    assert authenticate(password=PASSWORD, query_token=None) is False


def test_local_unauthenticated_install_still_rejects_a_foreign_origin():
    """The common case: no password at all. Auth is open by design, so origin
    checking is the only thing standing between a visited web page and the
    user's inbox."""
    assert authenticate(password="") is True
    assert origin_allowed("https://evil.example", ALLOWED) is False
