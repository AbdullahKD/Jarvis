"""
J.A.R.V.I.S Web Server
FastAPI backend with:
- WebSocket for real-time streaming chat
- HTTP /chat endpoint as fallback
- HTTP /sidebar endpoint for widget data
- Static file serving for the UI
"""

from __future__ import annotations

import os
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY", "False")

import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import aiohttp

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, Form
from fastapi.responses import JSONResponse
import json as _json

class SafeJSONResponse(JSONResponse):
    """JSONResponse that converts sets to lists automatically."""
    def render(self, content) -> bytes:
        def make_serializable(obj):
            if isinstance(obj, set):
                return list(obj)
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [make_serializable(i) for i in obj]
            return obj
        return _json.dumps(
            make_serializable(content),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config.logging_config import setup_logging
from orchestrator import JarvisOrchestrator
from agents.finex_agent import FinExAgent
from tools.reminders import ReminderScheduler

# Configure the Jarvis logger as early as possible so polling/debug logs are
# routed (and silenced at the default INFO level) before any agent imports
# start emitting them. Set JARVIS_LOG_LEVEL=DEBUG to surface the noisy poll
# logs (e.g. cricket fetch counts).
setup_logging()

# ── App setup ──────────────────────────────────────────────────────────────
#
# FastAPI deprecated @app.on_event in favour of a lifespan context manager.
# The deprecation is worth more than silencing a warning: one function now
# owns both halves, so everything startup creates has an obvious place to be
# torn down. The old handler started a reminder scheduler and two background
# tasks and stopped none of them — under `--reload` that leaves an extra
# scheduler polling the same SQLite file after every restart, and Ctrl-C
# printed "Task was destroyed but it is pending".
#
# Defined above `app` because FastAPI takes it as a constructor argument.
# Everything the body touches — `jarvis`, `VOICE_ENABLED`, `_voice_preflight` —
# is resolved when the function *runs*, not when it's defined, so those can
# keep living further down the module.
@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler = ReminderScheduler(jarvis.reminders)
    scheduler.start()

    # ── Warm the Ollama model so the first user query isn't cold ──
    async def _warmup():
        try:
            t0 = time.time()
            async for _ in jarvis.llm.chat_stream(
                [{"role": "user", "content": "ping"}],
                max_tokens=4,
            ):
                pass
            print(f"🔥 LLM warmup complete in {time.time()-t0:.2f}s")
        except Exception as exc:
            print(f"⚠️  LLM warmup failed: {type(exc).__name__}: {exc}")

    app.state.warmup_task = asyncio.ensure_future(_warmup())

    # ── Pre-warm the voice subsystem so the first /voice/start is snappy ──
    # Also validates the ElevenLabs API key against ElevenLabs' /v1/user so
    # bad permissions surface in the server log immediately, not 20 seconds
    # later when the user clicks the button.
    if VOICE_ENABLED:
        app.state.voice_task = asyncio.ensure_future(_voice_preflight())
    else:
        app.state.voice_task = None
        print("🔇 Voice disabled (JARVIS_VOICE_ENABLED=false) — no ElevenLabs "
              "calls, Whisper not loaded.")

    yield

    # ── shutdown ──
    # Cancel before awaiting: both of these sit in network calls that can
    # outlast the process otherwise. Failures are swallowed deliberately —
    # nothing here should be able to turn a clean exit into a traceback.
    await scheduler.stop()
    for label, task in (("LLM warmup", getattr(app.state, "warmup_task", None)),
                        ("voice preflight", getattr(app.state, "voice_task", None))):
        if task is None or task.done():
            continue
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            # Not redundant with the Exception clause below: since 3.8
            # CancelledError inherits BaseException, so `except Exception`
            # alone would let it escape and abort the rest of the shutdown.
            pass
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️  {label} did not shut down cleanly: "
                  f"{type(exc).__name__}: {exc}")
    print("👋 Jarvis shut down cleanly.")


app = FastAPI(title="J.A.R.V.I.S", version="1.0.0", lifespan=lifespan)


async def _voice_preflight() -> None:
    """Background voice subsystem warmup + API key check. Never raises."""
    import threading

    # 1. ElevenLabs API key validation
    key = os.getenv("ELEVENLABS_API_KEY", "").strip()
    if not key:
        print("⚠️  Voice: ELEVENLABS_API_KEY is empty in .env — voice replies will fail.")
    else:
        fingerprint = f"...{key[-4:]}" if len(key) >= 4 else "(short)"
        print(f"🎙️  Voice: ElevenLabs key fingerprint {fingerprint}")
        try:
            import aiohttp
            timeout = aiohttp.ClientTimeout(total=8)
            async with aiohttp.ClientSession(timeout=timeout) as s:
                async with s.get(
                    "https://api.elevenlabs.io/v1/user",
                    headers={"xi-api-key": key},
                ) as r:
                    if r.status == 200:
                        body = await r.json()
                        tier = body.get("subscription", {}).get("tier", "?")
                        chars_used = body.get("subscription", {}).get("character_count", "?")
                        chars_limit = body.get("subscription", {}).get("character_limit", "?")
                        print(f"✅  Voice: ElevenLabs key OK ({tier}, used {chars_used}/{chars_limit} chars)")
                    elif r.status == 401:
                        body = await r.text()
                        print(f"❌  Voice: ElevenLabs key REJECTED (401). Body: {body[:200]}")
                        print("    Fix: regenerate the key with `text_to_speech: Access` enabled.")
                    else:
                        body = await r.text()
                        print(f"⚠️  Voice: ElevenLabs returned HTTP {r.status}: {body[:200]}")
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️  Voice: couldn't reach ElevenLabs to validate key: {exc}")

    # 2. Audio output device — honour AUDIO_OUTPUT_DEVICE if set
    try:
        import sounddevice as sd
        from voice.config import load_config as _vlc
        _cfg_for_audio = _vlc()
        if _cfg_for_audio.output_device is not None:
            current_in, _ = sd.default.device
            sd.default.device = (current_in, _cfg_for_audio.output_device)
            print(f"🔊  Voice: output device overridden to #{_cfg_for_audio.output_device} via AUDIO_OUTPUT_DEVICE")
        default_out_idx = sd.default.device[1]
        try:
            default_out_name = sd.query_devices(default_out_idx)["name"]
        except Exception:  # noqa: BLE001
            default_out_name = f"device #{default_out_idx}"
        print(f"🔊  Voice: audio output device = {default_out_name!r}  (set AUDIO_OUTPUT_DEVICE=N in .env to override; list via GET /voice/devices)")
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️  Voice: couldn't enumerate audio devices: {exc}")

    # 3. Whisper model pre-load (eager, not lazy)
    def _load_whisper():
        try:
            t0 = time.time()
            from voice.config import load_config
            from voice.stt import StreamingSTT
            cfg = load_config()
            stt = StreamingSTT(cfg)
            stt.warmup()
            print(f"✅  Voice: Whisper {cfg.whisper_model} loaded in {time.time()-t0:.1f}s")
            app.state.warm_stt = stt
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️  Voice: Whisper warmup failed: {type(exc).__name__}: {exc}")

    threading.Thread(target=_load_whisper, daemon=True).start()

# CORS: the UI is served from this same origin, so cross-origin browser
# access is not needed in normal operation. A wildcard here + credentials
# would let ANY website open in the user's browser drive the Jarvis API
# (read inbox, fire mac_control, send email). Add extra origins via
# JARVIS_ALLOWED_ORIGINS, comma-separated.
_allowed_origins = [
    o.strip() for o in os.getenv(
        "JARVIS_ALLOWED_ORIGINS",
        "http://localhost:8000,http://127.0.0.1:8000",
    ).split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Optional HTTP Basic Auth ───────────────────────────────────────────────
# If JARVIS_AUTH_PASSWORD is set, every request must include a matching
# Basic-Auth header. The browser handles this transparently with a login
# popup, and curl users can pass `-u admin:<password>`.
#
# Leave it unset for ordinary local use; set it whenever the server is bound
# to anything other than loopback (a tailnet, a LAN address, an SSH tunnel
# you share), because that is the point at which the API stops being yours
# alone. The WS token in core/ws_guard.py keys off the same variable.
import base64 as _b64
from fastapi import Request as _Request, Response as _Response
from starlette.middleware.base import BaseHTTPMiddleware as _BaseHTTPMiddleware

_AUTH_USER = os.getenv("JARVIS_AUTH_USER", "admin")
_AUTH_PASS = os.getenv("JARVIS_AUTH_PASSWORD", "")
# Public paths that bypass auth — the liveness probe must stay reachable so a
# supervisor (launchd, a shell loop) can tell "down" from "needs a password".
_AUTH_PUBLIC_PATHS = {"/healthz"}


class _BasicAuthMiddleware(_BaseHTTPMiddleware):
    async def dispatch(self, request: _Request, call_next):
        if not _AUTH_PASS:
            return await call_next(request)
        if request.url.path in _AUTH_PUBLIC_PATHS:
            return await call_next(request)
        header = request.headers.get("authorization", "")
        if header.startswith("Basic "):
            try:
                decoded = _b64.b64decode(header[6:]).decode("utf-8", errors="ignore")
                user, _, pw = decoded.partition(":")
                # compare_digest: constant-time comparison so response timing
                # can't be used to guess the credentials character by character.
                import secrets as _secrets
                if (_secrets.compare_digest(user, _AUTH_USER)
                        and _secrets.compare_digest(pw, _AUTH_PASS)):
                    return await call_next(request)
            except Exception:
                pass
        return _Response(
            status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="Jarvis", charset="UTF-8"'},
        )


if _AUTH_PASS:
    app.add_middleware(_BasicAuthMiddleware)


@app.get("/healthz")
async def _healthz():
    """Unauthenticated liveness probe — stays open so a supervisor can tell
    "process down" from "process up, needs a password"."""
    return {"ok": True}


# ── Voice kill-switch ──────────────────────────────────────────────────────
# Voice is OFF by default: no ElevenLabs calls (no fees), no Whisper model
# loaded into RAM, no voice preflight at startup, voice button hidden in the
# UI. Set JARVIS_VOICE_ENABLED=true in .env to bring it all back.
VOICE_ENABLED = os.getenv(
    "JARVIS_VOICE_ENABLED", "false"
).strip().lower() in ("1", "true", "yes", "on")

_VOICE_DISABLED_RESPONSE = {
    "ok": False,
    "disabled": True,
    "error": "Voice is disabled (set JARVIS_VOICE_ENABLED=true in .env to enable).",
}


def _voice_is_active() -> bool:
    """True when the voice session is mid-turn (listening/thinking/speaking).
    The sidebar + live-tick endpoints check this and tell the UI to back off
    so we don't fan out 30+ HTTP requests at the exact moment the user is
    talking to Jarvis. Falls open (returns False) when the session hasn't
    been instantiated yet."""
    sess = globals().get("_voice_session")
    if sess is None:
        return False
    try:
        state = sess.status().get("state", "idle")
    except Exception:
        return False
    return state in ("listening", "thinking", "speaking")


@app.get("/voice/active")
async def voice_active():
    """Lightweight flag the UI polls every few seconds to decide whether to
    suspend heavy refresh loops (sidebar, live-tick) during a voice turn.
    Also carries `enabled` so the UI can hide the voice button entirely."""
    return {"active": _voice_is_active(), "enabled": VOICE_ENABLED}

# Shared orchestrator instance
jarvis = JarvisOrchestrator()


# FinEx is built on first use, not at import.
#
# Constructing it opens a Postgres connection and a second ChromaDB client,
# and imports psycopg2 — on the startup path, for a feature most sessions
# never touch. The orchestrator already treats it as lazy (see its `finex`
# property); this module-level eager build defeated that, which is why
# "FinExAgent ready" printed on every boot.
_finex_instance = None


def get_finex():
    """The process-wide FinExAgent, constructed on first call."""
    global _finex_instance
    if _finex_instance is None:
        _finex_instance = FinExAgent()
    return _finex_instance


class _LazyFinExProxy:
    """Keeps `finex.chat(...)` working at the ~6 existing call sites without
    building the agent until one of them actually runs."""

    def __getattr__(self, item):
        return getattr(get_finex(), item)

    def __bool__(self) -> bool:
        return True


finex = _LazyFinExProxy()

UI_DIR = Path(__file__).parent / "ui"


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/")
async def serve_ui():
    """Serve the main UI."""
    return FileResponse(UI_DIR / "index.html")


class ChatRequest(BaseModel):
    message: str
    # The UI tracks conversation history in-memory and sends it with each
    # request so the HTTP /chat endpoint has the same context the WebSocket
    # path gets. Without this, follow-ups like "elaborate" / "give me more
    # detail" have no prior exchange to expand on and the orchestrator
    # has to ask "what would you like me to elaborate on?" — or worse,
    # routes the bare word through the LLM and gets a capability list.
    history: list = []


@app.post("/chat")
async def chat(req: ChatRequest):
    """HTTP chat endpoint — fallback when WebSocket unavailable."""
    response = await jarvis.handle(
        req.message,
        conversation_history=req.history,
    )
    return {
        "success": response.success,
        "message": response.message,
        "latency_ms": response.latency_ms,
    }


# ── Voice endpoints ────────────────────────────────────────────────────────
#
# A single voice "turn" = listen → think → speak. The UI polls /voice/status
# 4× per second to drive the orb animation. The session is lazy-loaded on
# first use so server boot doesn't pull Whisper into memory unless needed.

import threading as _voice_threading

_voice_session = None
_voice_session_lock = _voice_threading.Lock()
_voice_stylizer = None  # PersonaStylizer (lazy-built alongside the session)


# Abbreviations that end in "." but are NOT sentence boundaries. Keeps the
# TTS sentence-splitter from speaking "Mr." and "Stark" as two choppy chunks.
_SPEECH_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc",
    "e.g", "i.e", "a.m", "p.m", "approx", "no", "fig", "inc", "ltd", "co",
}


def _extract_speech_chunks(buf: str):
    """
    Split `buf` into complete, speakable chunks on sentence boundaries.

    Returns (chunks, remainder). Unlike a naive `(?<=[.!?])\\s+` split this
    does NOT break on:
      - decimals  ("3.5", "v2.5")          → digit . digit
      - abbreviations ("Mr.", "e.g.")      → known short tokens
    so ElevenLabs receives whole sentences and Jarvis sounds natural instead
    of stuttering on fragments.
    """
    chunks: list[str] = []
    last = 0
    i = 0
    n = len(buf)
    while i < n:
        c = buf[i]
        if c in ".!?":
            # A boundary only counts if followed by whitespace (or end).
            if i + 1 < n and not buf[i + 1].isspace():
                # e.g. decimal "3.5" or "v2.5" — not a boundary.
                i += 1
                continue
            if c == "." and i > 0 and buf[i - 1].isdigit():
                # trailing decimal point with space after ("section 3. ") is
                # rare; treat conservatively as non-boundary.
                i += 1
                continue
            seg = buf[last:i + 1].strip()
            words = seg.split()
            tail_word = words[-1].rstrip(".!?").lower() if words else ""
            if tail_word in _SPEECH_ABBREVIATIONS:
                i += 1
                continue
            if seg:
                chunks.append(seg)
                last = i + 1
        i += 1
    return chunks, buf[last:]


async def _ensure_voice_session():
    """Lazy-build the VoiceSession with a streaming brain bound to FastAPI's event loop."""
    global _voice_session, _voice_stylizer
    if _voice_session is not None:
        return _voice_session

    main_loop = asyncio.get_running_loop()

    # Persona stylizing is disabled — replies go orchestrator → TTS directly.
    _voice_stylizer = None

    def _voice_brain_streaming(transcript: str, *, tts, on_state, cancel_event, on_partial=None) -> str:
        """
        Streaming voice brain.

        Pipes `orchestrator.handle_stream(...)` chunks through a sentence
        splitter into a queue, and a worker thread inside this function
        drives `tts.speak_sentence_stream(queue)` to play each completed
        sentence via ElevenLabs Flash v2.5 the moment it lands. The user
        hears Jarvis start talking within ~1 s of the LLM beginning to
        stream, instead of waiting for the whole reply first.

        Falls back to a single tts.speak() call when the orchestrator
        returns the whole answer in one chunk (Tier 1 / FinEx fast path).
        """
        import re as _re
        import queue as _queue
        import threading as _threading
        import time

        print(f"🧠  [brain] Streaming orchestrator: {transcript!r}")
        t0 = time.perf_counter()

        sentence_q: "_queue.Queue[Optional[str]]" = _queue.Queue()
        full_text_holder = {"text": ""}
        tts_started = _threading.Event()

        # TTS worker: drains the sentence queue and plays each sentence in
        # arrival order via ElevenLabs streaming. on_state flips the UI
        # orb to "speaking" the moment the first audio plays.
        def _on_first_audio():
            tts_started.set()
            on_state()

        def _tts_worker():
            try:
                tts.speak_sentence_stream(sentence_q, on_first_audio=_on_first_audio)
            except Exception as exc:  # noqa: BLE001
                print(f"🔊  [brain] TTS worker error: {exc!r}")

        worker = _threading.Thread(target=_tts_worker, name="voice-tts-worker", daemon=True)
        worker.start()

        # Async producer that runs on the main event loop and chops chunks
        # into sentences as they stream out of the orchestrator.
        async def _produce() -> str:
            buf = ""
            full_text = ""
            try:
                async for event in jarvis.handle_stream(
                    transcript,
                    voice_mode=True,
                ):
                    if cancel_event.is_set():
                        break
                    etype = event.get("type")
                    if etype == "chunk":
                        chunk = event.get("text", "")
                        if not chunk:
                            continue
                        buf += chunk
                        full_text += chunk
                        # Push the accumulating text to the UI so the user sees
                        # a live transcript of what Jarvis is saying.
                        if on_partial is not None:
                            try:
                                on_partial(full_text.strip())
                            except Exception:  # noqa: BLE001
                                pass
                        # Split on real sentence boundaries (decimal- and
                        # abbreviation-aware) — push every complete sentence
                        # to the queue, keep the trailing partial in `buf`.
                        ready, buf = _extract_speech_chunks(buf)
                        for s in ready:
                            if s:
                                sentence_q.put(s)
                    elif etype == "response":
                        # Final canonical text from the orchestrator. Overrides
                        # the streamed accumulator (the orchestrator may have
                        # post-processed via _enforce_single_paragraph).
                        msg = event.get("message")
                        if msg:
                            full_text = msg
                            if on_partial is not None:
                                try:
                                    on_partial(full_text.strip())
                                except Exception:  # noqa: BLE001
                                    pass
            finally:
                # Flush any trailing buffer.
                tail = buf.strip()
                if tail:
                    sentence_q.put(tail)
                # If we never streamed any sentence (Tier 1 or FinEx fast-path
                # returned a single chunk), make sure the final text gets
                # spoken in full.
                if not tts_started.is_set() and full_text.strip() and not tail:
                    sentence_q.put(full_text.strip())
                sentence_q.put(None)  # sentinel
            return full_text

        fut = asyncio.run_coroutine_threadsafe(_produce(), main_loop)
        try:
            full_text = fut.result(timeout=120)
            full_text_holder["text"] = full_text
        except Exception as exc:  # noqa: BLE001
            print(f"❌  [brain] Orchestrator FAILED: {type(exc).__name__}: {exc}")
            sentence_q.put(None)  # let worker exit
            return f"I hit an error, sir: {exc}"

        # Wait for TTS to finish speaking the queued sentences.
        worker.join(timeout=90)

        orch_ms = (time.perf_counter() - t0) * 1000
        print(f"🧠  [brain] Streaming turn done ({orch_ms:.0f} ms): {full_text_holder['text'][:120]!r}")
        return full_text_holder["text"]

    with _voice_session_lock:
        if _voice_session is None:
            from voice.web import VoiceSession  # heavy import → defer until needed
            _voice_session = VoiceSession(
                brain=_voice_brain_streaming,
                brain_streams_tts=True,
            )
            # If startup pre-warmed Whisper, hand the warm instance to the
            # session so the first listen_once doesn't re-load the model.
            warm_stt = getattr(app.state, "warm_stt", None)
            if warm_stt is not None and hasattr(warm_stt, "_model") and warm_stt._model is not None:
                _voice_session._stt._model = warm_stt._model  # type: ignore[attr-defined]
                print("🎙️  Voice session adopted pre-warmed Whisper model")
    return _voice_session


@app.post("/voice/start")
async def voice_start():
    """Begin a single voice turn (listen → think → speak)."""
    if not VOICE_ENABLED:
        return JSONResponse(status_code=403, content=_VOICE_DISABLED_RESPONSE)
    try:
        session = await _ensure_voice_session()
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(
            status_code=503,
            content={
                "ok": False,
                "error": str(exc),
                "hint": "Check VOICE_SETUP.md — ELEVENLABS_API_KEY and pip install -r voice_requirements.txt",
            },
        )
    started = session.start()
    return {"ok": True, "started": started, "status": session.status()}


@app.get("/voice/status")
async def voice_status():
    """Return current voice session state — UI polls this 4 Hz."""
    if _voice_session is None:
        return {"state": "idle", "transcript": "", "reply": "", "error": "", "active": False}
    return _voice_session.status()


@app.post("/voice/stop")
async def voice_stop():
    """Interrupt the current turn (cancels STT or TTS mid-stream)."""
    if _voice_session is None:
        return {"ok": True, "status": {"state": "idle"}}
    _voice_session.cancel()
    return {"ok": True, "status": _voice_session.status()}


@app.post("/voice/test")
async def voice_test():
    """
    Isolated TTS test. Synthesises a fixed phrase through ElevenLabs and
    plays it via sounddevice. Bypasses STT / orchestrator / persona.

    Also prints the API key fingerprint (last 4 chars) so you can confirm
    the server is using the key you think it is.
    """
    if not VOICE_ENABLED:
        return JSONResponse(status_code=403, content=_VOICE_DISABLED_RESPONSE)
    test_phrase = "Audio test successful, sir. ElevenLabs and the output device are both working."

    from voice.config import load_config
    cfg = load_config()
    fingerprint = (
        f"...{cfg.elevenlabs_api_key[-4:]}"
        if cfg.elevenlabs_api_key else "(EMPTY!)"
    )
    print(f"🔊  [voice/test] Using ElevenLabs key fingerprint: {fingerprint}")
    print(f"🔊  [voice/test] Voice ID: {cfg.elevenlabs_voice_id}  |  Model: {cfg.elevenlabs_model}")

    if not cfg.elevenlabs_api_key:
        return JSONResponse(
            status_code=400,
            content={
                "ok": False,
                "error": "ELEVENLABS_API_KEY is empty in .env",
                "hint": "Paste a key into .env and restart the server.",
            },
        )

    try:
        from voice.tts import StreamingTTS
        tts = StreamingTTS(cfg)
        print(f"🔊  [voice/test] Speaking: {test_phrase!r}")
        tts.speak(test_phrase)
        print("🔊  [voice/test] Done.")
        return {"ok": True, "key_fingerprint": fingerprint, "spoken": test_phrase}
    except Exception as exc:  # noqa: BLE001
        import traceback
        tb = traceback.format_exc()
        msg = str(exc)
        # Surface the ElevenLabs JSON detail if it's a 401 / permission error
        hint = None
        if "missing the permission text_to_speech" in msg:
            hint = (
                "Your API key doesn't have text_to_speech permission. "
                "Recreate the key at elevenlabs.io with Text to Speech → Access enabled."
            )
        elif "401" in msg or "Unauthorized" in msg:
            hint = "API key invalid or revoked. Generate a new key at elevenlabs.io."
        elif "429" in msg:
            hint = "Rate limited. Wait a minute or upgrade tier."
        print(f"❌  [voice/test] Failed: {msg[:300]}")
        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "key_fingerprint": fingerprint,
                "error": msg[:500],
                "hint": hint,
                "traceback_tail": tb.splitlines()[-6:],
            },
        )


@app.get("/voice/devices")
async def voice_devices():
    if not VOICE_ENABLED:
        return JSONResponse(status_code=403, content=_VOICE_DISABLED_RESPONSE)
    """
    List all audio devices visible to sounddevice (both inputs and outputs)
    plus the current defaults. Use this to confirm macOS is capturing from
    the mic you expect *and* playing through the speakers you expect.

    To override either default, set AUDIO_INPUT_DEVICE=<index> or
    AUDIO_OUTPUT_DEVICE=<index> in .env, then restart the server.
    """
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        default_input, default_output = sd.default.device
        in_devices = []
        out_devices = []
        for i, d in enumerate(devices):
            entry_in = {
                "index": i,
                "name": d.get("name"),
                "channels": d.get("max_input_channels"),
                "default_sample_rate": d.get("default_samplerate"),
                "is_default": (i == default_input),
            }
            entry_out = {
                "index": i,
                "name": d.get("name"),
                "channels": d.get("max_output_channels"),
                "default_sample_rate": d.get("default_samplerate"),
                "is_default": (i == default_output),
            }
            if d.get("max_input_channels", 0) > 0:
                in_devices.append(entry_in)
            if d.get("max_output_channels", 0) > 0:
                out_devices.append(entry_out)
        return {
            "ok": True,
            "default_input_index": int(default_input),
            "default_output_index": int(default_output),
            "inputs": in_devices,
            "outputs": out_devices,
            "hint": (
                "If the wrong mic is default, switch in System Settings → "
                "Sound → Input, or set AUDIO_INPUT_DEVICE=<index> in .env "
                "and restart. Same for outputs via AUDIO_OUTPUT_DEVICE."
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": f"{type(exc).__name__}: {exc}"},
        )


@app.post("/voice/mic-test")
async def voice_mic_test(duration: float = 3.0):
    if not VOICE_ENABLED:
        return JSONResponse(status_code=403, content=_VOICE_DISABLED_RESPONSE)
    """
    Definitive microphone diagnostic. Captures `duration` seconds of audio
    from the configured input device and reports:

      - bytes_captured / frames_captured  (is anything coming through?)
      - peak_amplitude_int16              (0..32767 — anything < 100 means
                                           macOS is feeding digital silence,
                                           almost certainly a permission issue)
      - rms_amplitude_int16               (loudness over the window)
      - vad_max_prob                      (did Silero ever think it heard speech?)
      - device                            (which device was used)

    Call this once before trying /voice/start to confirm the mic actually
    works. If peak_amplitude_int16 stays near zero while you speak, the
    problem is upstream of Jarvis: grant mic permission in
    System Settings → Privacy & Security → Microphone (Terminal / VS Code).
    """
    duration = max(0.5, min(float(duration), 10.0))

    def _run_capture() -> dict:
        import numpy as np
        import sounddevice as sd
        from voice.audio import MicStream, CHUNK_SAMPLES
        from voice.config import load_config
        from voice.vad import SileroEndpointer

        cfg = load_config()

        # Resolve which device sounddevice will actually use, so the report
        # is unambiguous (default_input may be -1 on headless setups).
        try:
            idx = cfg.input_device if cfg.input_device is not None else sd.default.device[0]
            dev_info = sd.query_devices(idx)
            device_name = dev_info.get("name", f"device #{idx}")
            device_index = int(idx)
        except Exception as exc:  # noqa: BLE001
            device_name = f"(unknown — {exc})"
            device_index = -1

        mic = MicStream(cfg)
        mic.start()

        chunks: list = []
        vad = SileroEndpointer(cfg)
        vad.reset()
        peak = 0
        import time as _t
        deadline = _t.monotonic() + duration
        try:
            while _t.monotonic() < deadline:
                try:
                    remaining = max(0.05, deadline - _t.monotonic())
                    frame = mic.read(timeout=remaining)
                except Exception:  # noqa: BLE001
                    break
                chunks.append(frame)
                vad.feed(frame)
                if frame.size:
                    p = int(np.max(np.abs(frame)))
                    if p > peak:
                        peak = p
        finally:
            mic.stop()

        if not chunks:
            audio = np.zeros(0, dtype=np.int16)
        else:
            audio = np.concatenate(chunks)

        rms = int(float(np.sqrt(np.mean(audio.astype(np.float32) ** 2))) if audio.size else 0)
        bytes_captured = int(audio.nbytes)
        frames_captured = int(audio.size)

        # Verdict
        if peak < 100:
            verdict = (
                "FAIL — mic appears to be feeding digital silence. Grant mic "
                "permission to the terminal / VS Code in System Settings → "
                "Privacy & Security → Microphone, then restart the server."
            )
            ok = False
        elif peak < 500:
            verdict = (
                "WEAK — mic is capturing but the signal is very quiet. "
                "Speak louder, move closer, or pick a better mic via "
                "AUDIO_INPUT_DEVICE in .env."
            )
            ok = True
        elif vad.max_prob < 0.3:
            verdict = (
                "OK — audio captured but Silero VAD didn't recognise speech. "
                "If you spoke during the test, lower VAD_THRESHOLD in .env "
                "(try 0.35 or 0.3)."
            )
            ok = True
        else:
            verdict = "OK — mic captures audio and Silero detected speech."
            ok = True

        return {
            "ok": ok,
            "verdict": verdict,
            "device": {"index": device_index, "name": device_name},
            "duration_s": duration,
            "frames_captured": frames_captured,
            "bytes_captured": bytes_captured,
            "peak_amplitude_int16": int(peak),
            "rms_amplitude_int16": rms,
            "vad_max_prob": round(float(vad.max_prob), 3),
            "sample_rate": cfg.sample_rate,
        }

    # The mic capture is blocking — run it off the event loop so /voice/status
    # polling stays responsive during the 3 s test.
    try:
        result = await asyncio.to_thread(_run_capture)
        return result
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": f"{type(exc).__name__}: {exc}"},
        )


@app.get("/voice/health")
async def voice_health():
    """
    One-shot diagnostic: which voice dependencies are wired up?

    Visit /voice/health in your browser (or `curl localhost:8000/voice/health`)
    to see exactly which piece of the voice stack is missing.
    """
    if not VOICE_ENABLED:
        return dict(_VOICE_DISABLED_RESPONSE)
    import importlib
    report: dict = {"ok": True, "checks": {}}

    # 1. ElevenLabs API key
    key = os.getenv("ELEVENLABS_API_KEY", "").strip()
    report["checks"]["elevenlabs_api_key"] = {
        "ok": bool(key),
        "detail": f"set ({len(key)} chars)" if key else "missing — add ELEVENLABS_API_KEY to .env",
    }
    if not key:
        report["ok"] = False

    # 2. Python deps (wake-word removed — push-to-talk only)
    for mod, hint in [
        ("elevenlabs",      "pip install elevenlabs"),
        ("faster_whisper",  "pip install faster-whisper"),
        ("silero_vad",      "pip install silero-vad"),
        ("sounddevice",     "pip install sounddevice (and on macOS: brew install portaudio)"),
    ]:
        try:
            importlib.import_module(mod)
            report["checks"][mod] = {"ok": True, "detail": "importable"}
        except Exception as exc:  # noqa: BLE001
            report["checks"][mod] = {"ok": False, "detail": f"{type(exc).__name__}: {exc} — {hint}"}
            report["ok"] = False

    # 3. Voice session state
    if _voice_session is None:
        report["checks"]["voice_session"] = {"ok": True, "detail": "not yet instantiated (lazy)"}
    else:
        report["checks"]["voice_session"] = {"ok": True, "detail": _voice_session.status()}

    # 4. Persona stylizer state
    persona_enabled = os.getenv("VOICE_PERSONA_ENABLED", "true").lower() in ("1","true","yes","on")
    report["checks"]["voice_persona"] = {
        "ok": True,
        "detail": {
            "enabled": persona_enabled,
            "stylizer_instantiated": _voice_stylizer is not None,
        },
    }

    return report


@app.get("/sidebar")
async def sidebar():
    """
    Return all sidebar widget data in one call.
    Called on page load and every 60 seconds.

    Suspended while a voice turn is in flight so we don't fan out 22
    parallel HTTP requests at the same moment the user is talking to
    Jarvis (which used to contend for network + the Ollama socket).
    """
    if _voice_is_active():
        return SafeJSONResponse({"_suspended": True, "reason": "voice_active"})

    # Fetch everything in parallel
    async def safe(coro):
        try:
            return await coro
        except Exception as e:
            return {}

    (weather, markets, calendar, spotify, news, tech_news, sports_news,
     world_news, politics_news, science_news, entertainment_news,
     s_pl, s_ucl, s_nba, s_friendly,
     s_cricket,
     s_tennis, s_tennis_wta, s_ufc, s_boxing, s_f1,
     s_rm, s_mu, s_gsw, prayer, gmail) = await asyncio.gather(
        safe(jarvis.weather.get_current()),
        safe(jarvis.markets.get_all()),
        safe(jarvis.calendar.search_events()),
        safe(jarvis.spotify.get_now_playing()),
        safe(jarvis.news.get_headlines(max_stories=8)),
        safe(jarvis.news.get_headlines(category="technology", max_stories=8, query="artificial intelligence machine learning tech")),
        safe(jarvis.news.get_headlines(category="sports", max_stories=8)),
        safe(jarvis.news.get_headlines(category="world", max_stories=10)),
        safe(jarvis.news.get_headlines(category="world", max_stories=10, query="politics election government parliament minister policy vote")),
        safe(jarvis.news.get_headlines(category="science", max_stories=10)),
        safe(jarvis.news.get_headlines(category="entertainment", max_stories=10)),
        safe(jarvis.sports.get_scores("premier_league", limit=8)),
        safe(jarvis.sports.get_scores("champions_league", limit=6)),
        safe(jarvis.sports.get_scores("nba", limit=6)),
        # Summer tours / pre-season — separate ESPN league, never in the
        # PL/UCL scoreboards. This is what fills football cards in July.
        safe(jarvis.sports.get_scores("club_friendlies", limit=10)),
        # Cricket: ESPNcricinfo current-matches feed — all live/recent/upcoming
        # across every series, consolidated into one `sports_cricket` section.
        safe(jarvis.sports.get_cricket_current(limit=16)),
        # Individual / event sports — best-effort (different ESPN shapes).
        safe(jarvis.sports.get_scores("tennis",       limit=8)),
        safe(jarvis.sports.get_scores("tennis_wta",   limit=8)),
        safe(jarvis.sports.get_scores("ufc",          limit=6)),
        safe(jarvis.sports.get_scores("boxing",       limit=6)),
        safe(jarvis.sports.get_scores("f1",           limit=6)),
        # Favourite-team season schedules (includes preseason via the
        # seasontype fallback) — one per configured favourite with a
        # league ESPN can resolve.
        safe(jarvis.sports.search_team("Real Madrid", "la_liga")),
        safe(jarvis.sports.search_team("Manchester United", "premier_league")),
        safe(jarvis.sports.search_team("Golden State Warriors", "nba")),
        safe(jarvis.prayer.get_times()),
        safe(jarvis.gmail.get_inbox(max_results=8, query="is:inbox")),
    )

    result = {}
    if weather.get("success"): result["weather"] = weather
    if markets.get("success"): result["markets"] = markets
    if calendar.get("success"): result["calendar"] = {"events": calendar.get("events", []), "connected": not jarvis.calendar.is_mock}
    if spotify.get("success"): result["spotify"] = {
        "track": spotify.get("track",""),
        "artist": spotify.get("artist",""),
        "playing": spotify.get("playing", False),
        "image_url": spotify.get("image_url",""),
        "progress_pct": spotify.get("progress_pct", 0),
    }
    if news.get("success"):
        result["news"] = {"stories": [{"title":s.get("title",""),"description":s.get("description",""),"sources":list(s.get("sources",[]) if not isinstance(s.get("sources",[]),set) else s.get("sources",set())),"url":s.get("url","")} for s in news.get("stories",[])]}
    if tech_news.get("success"):
        result["tech_news"] = {"stories": [{"title":s.get("title",""),"description":s.get("description",""),"sources":list(s.get("sources",[]) if not isinstance(s.get("sources",[]),set) else s.get("sources",set())),"url":s.get("url","")} for s in tech_news.get("stories",[])]}
    if sports_news.get("success"):
        result["sports_news"] = {"stories": [{"title":s.get("title",""),"description":s.get("description",""),"sources":list(s.get("sources",[]) if not isinstance(s.get("sources",[]),set) else s.get("sources",set())),"url":s.get("url","")} for s in sports_news.get("stories",[])]}
    if world_news.get("success"):
        result["world_news"] = {"stories": [{"title":s.get("title",""),"description":s.get("description",""),"sources":list(s.get("sources",[]) if not isinstance(s.get("sources",[]),set) else s.get("sources",set())),"url":s.get("url","")} for s in world_news.get("stories",[])]}
    if politics_news.get("success"):
        result["politics_news"] = {"stories": [{"title":s.get("title",""),"description":s.get("description",""),"sources":list(s.get("sources",[]) if not isinstance(s.get("sources",[]),set) else s.get("sources",set())),"url":s.get("url","")} for s in politics_news.get("stories",[])]}
    if science_news.get("success"):
        result["science_news"] = {"stories": [{"title":s.get("title",""),"description":s.get("description",""),"sources":list(s.get("sources",[]) if not isinstance(s.get("sources",[]),set) else s.get("sources",set())),"url":s.get("url","")} for s in science_news.get("stories",[])]}
    if entertainment_news.get("success"):
        result["entertainment_news"] = {"stories": [{"title":s.get("title",""),"description":s.get("description",""),"sources":list(s.get("sources",[]) if not isinstance(s.get("sources",[]),set) else s.get("sources",set())),"url":s.get("url","")} for s in entertainment_news.get("stories",[])]}
    def _ser_games(raw_list):
        """Serialise a list of parsed game dicts, including logo/color/date fields."""
        out = []
        for g in raw_list:
            out.append({
                "home_team":  g.get("home_team", ""),
                "away_team":  g.get("away_team", ""),
                "home_score": g.get("home_score", ""),
                "away_score": g.get("away_score", ""),
                "status":     g.get("status", ""),
                "date_str":   g.get("date_str", ""),
                "date_iso":   g.get("date_iso", ""),
                "clock":      g.get("clock", ""),
                "home_color": g.get("home_color", ""),
                "away_color": g.get("away_color", ""),
                "home_logo":  g.get("home_logo", ""),
                "away_logo":  g.get("away_logo", ""),
            })
        return out

    if s_pl.get("success"):
        result["sports_pl"]  = {"success":True,"league":s_pl.get("league",""),"league_key":s_pl.get("league_key",""),"games":_ser_games(s_pl.get("games",[]))}
    if s_ucl.get("success"):
        result["sports_ucl"] = {"success":True,"league":s_ucl.get("league",""),"league_key":s_ucl.get("league_key",""),"games":_ser_games(s_ucl.get("games",[]))}
    if s_nba.get("success"):
        result["sports_nba"] = {"success":True,"league":s_nba.get("league",""),"league_key":s_nba.get("league_key",""),"games":_ser_games(s_nba.get("games",[]))}
    # ── Consolidated cricket payload (ESPNcricinfo current matches) ──────────
    # get_cricket_current already returns clean, ordered cricket game dicts with
    # innings / note / format fields, so we just pass them through.
    def _ser_cricket_games(raw_list):
        out = []
        for g in raw_list:
            out.append({
                "home_team":  g.get("home_team", ""),
                "away_team":  g.get("away_team", ""),
                "home_score": g.get("home_score", ""),
                "away_score": g.get("away_score", ""),
                "status":     g.get("status", ""),
                "date_str":   g.get("date_str", ""),
                "date_iso":   g.get("date_iso", ""),
                "clock":      g.get("clock", ""),
                "home_color": g.get("home_color", ""),
                "away_color": g.get("away_color", ""),
                "home_logo":  g.get("home_logo", ""),
                "away_logo":  g.get("away_logo", ""),
                "home_innings": g.get("home_innings", []),
                "away_innings": g.get("away_innings", []),
                "note":         g.get("note", ""),
                "format":       g.get("format", ""),
            })
        return out

    if s_cricket.get("success") and s_cricket.get("games"):
        result["sports_cricket"] = {
            "success":    True,
            "league":     "Cricket",
            "league_key": "cricket",
            "games":      _ser_cricket_games(s_cricket.get("games", [])),
        }
    if s_rm.get("success"):
        result["sports_rm"] = {"success":True,"league":s_rm.get("league","La Liga"),"league_key":"la_liga","team":"Real Madrid","games":_ser_games(s_rm.get("games",[]))}
    if s_mu.get("success"):
        result["sports_mu"] = {"success":True,"league":s_mu.get("league","Premier League"),"league_key":"premier_league","team":"Manchester United","games":_ser_games(s_mu.get("games",[]))}
    if s_gsw.get("success"):
        result["sports_gsw"] = {"success":True,"league":s_gsw.get("league","NBA"),"league_key":"nba","team":"Golden State Warriors","games":_ser_games(s_gsw.get("games",[]))}
    if s_friendly.get("success") and s_friendly.get("games"):
        result["sports_friendly"] = {"success":True,"league":"Club Friendlies","league_key":"club_friendlies","games":_ser_games(s_friendly.get("games",[]))}
    # ── Individual / event sports (tennis, combat, F1) ──────────────────────
    # These carry typed fields (type, results, fighter1/2, score1/2, etc.) that
    # the football serialiser would drop, so pass the parsed dicts through whole.
    _INDIV_FIELDS = ("type","status","date_str","date_iso","home_team","away_team",
                     "home_score","away_score","detail",
                     "name","circuit","results",
                     "event_name","fighter1","fighter2","fighter1_logo","fighter2_logo","weight",
                     "fighter1_record","fighter2_record","card_note",
                     "tournament","round","player1","player2","score1","score2")
    def _ser_indiv(raw_list):
        return [{k: g.get(k) for k in _INDIV_FIELDS if k in g} for g in raw_list]

    _tennis_games = _ser_indiv(s_tennis.get("games", []) if s_tennis.get("success") else []) \
                  + _ser_indiv(s_tennis_wta.get("games", []) if s_tennis_wta.get("success") else [])
    if _tennis_games:
        result["sports_tennis"] = {"success":True,"league":"Tennis","league_key":"tennis","games":_tennis_games}
    if s_ufc.get("success") and s_ufc.get("games"):
        result["sports_ufc"] = {"success":True,"league":s_ufc.get("league","UFC"),"league_key":"ufc","games":_ser_indiv(s_ufc.get("games",[]))}
    if s_boxing.get("success") and s_boxing.get("games"):
        result["sports_boxing"] = {"success":True,"league":s_boxing.get("league","Boxing"),"league_key":"boxing","games":_ser_indiv(s_boxing.get("games",[]))}
    if s_f1.get("success") and s_f1.get("games"):
        result["sports_f1"] = {"success":True,"league":s_f1.get("league","Formula 1"),"league_key":"f1","games":_ser_indiv(s_f1.get("games",[]))}
    if prayer.get("success"): result["prayer"] = prayer
    if gmail.get("success"):
        result["emails"] = gmail.get("emails", [])
        # Keep the orchestrator's inbox cache in sync with exactly what the
        # sidebar shows, so "reply to email N" targets the message the user
        # sees here (same is:inbox source and newest-first order).
        jarvis._last_inbox = gmail.get("emails", []) or []

    # Connection diagnostics — the widgets render different empty states
    # depending on whether the agent is genuinely unconfigured, the token
    # expired, or a real auth error occurred.
    result["google"] = {
        "gmail_connected": not jarvis.gmail.is_mock,
        "calendar_connected": not jarvis.calendar.is_mock,
        "gmail_error": getattr(jarvis.gmail, "auth_error", None),
        "calendar_error": getattr(jarvis.calendar, "auth_error", None),
    }

    return SafeJSONResponse(result)


@app.get("/live-tick")
async def live_tick():
    """
    Lightweight endpoint for the UI's live-refresh poll.

    Returns only the fast-changing widgets — markets + all sports payloads —
    so the UI can keep prices and scores up to date every 20-30s without
    re-fetching news, weather, gmail, calendar, prayer times etc. that
    change much less often (those stay on the 60s /sidebar tick).

    Suspended while a voice turn is in flight (same reason as /sidebar).
    """
    if _voice_is_active():
        return SafeJSONResponse({"_suspended": True, "reason": "voice_active"})

    async def safe(coro):
        try:
            return await coro
        except Exception:
            return {}

    (markets, s_pl, s_ucl, s_nba, s_friendly,
     s_cricket,
     s_tennis, s_tennis_wta, s_ufc, s_boxing, s_f1,
     s_rm, s_mu, s_gsw) = await asyncio.gather(
        safe(jarvis.markets.get_all()),
        safe(jarvis.sports.get_scores("premier_league", limit=8)),
        safe(jarvis.sports.get_scores("champions_league", limit=6)),
        safe(jarvis.sports.get_scores("nba", limit=6)),
        safe(jarvis.sports.get_scores("club_friendlies", limit=10)),
        safe(jarvis.sports.get_cricket_current(limit=16)),
        safe(jarvis.sports.get_scores("tennis",     limit=8)),
        safe(jarvis.sports.get_scores("tennis_wta", limit=8)),
        safe(jarvis.sports.get_scores("ufc",        limit=6)),
        safe(jarvis.sports.get_scores("boxing",     limit=6)),
        safe(jarvis.sports.get_scores("f1",         limit=6)),
        safe(jarvis.sports.search_team("Real Madrid", "la_liga")),
        safe(jarvis.sports.search_team("Manchester United", "premier_league")),
        safe(jarvis.sports.search_team("Golden State Warriors", "nba")),
    )

    out = {}
    if markets.get("success"): out["markets"] = markets

    def _ser_games(raw_list):
        return [{
            "home_team":  g.get("home_team", ""),
            "away_team":  g.get("away_team", ""),
            "home_score": g.get("home_score", ""),
            "away_score": g.get("away_score", ""),
            "status":     g.get("status", ""),
            "date_str":   g.get("date_str", ""),
            "date_iso":   g.get("date_iso", ""),
            "clock":      g.get("clock", ""),
            "home_color": g.get("home_color", ""),
            "away_color": g.get("away_color", ""),
            "home_logo":  g.get("home_logo", ""),
            "away_logo":  g.get("away_logo", ""),
        } for g in raw_list]

    if s_pl.get("success"):
        out["sports_pl"]  = {"success":True,"league":s_pl.get("league",""),"league_key":"premier_league","games":_ser_games(s_pl.get("games",[]))}
    if s_ucl.get("success"):
        out["sports_ucl"] = {"success":True,"league":s_ucl.get("league",""),"league_key":"champions_league","games":_ser_games(s_ucl.get("games",[]))}
    if s_nba.get("success"):
        out["sports_nba"] = {"success":True,"league":s_nba.get("league",""),"league_key":"nba","games":_ser_games(s_nba.get("games",[]))}
    if s_rm.get("success"):
        out["sports_rm"]  = {"success":True,"league":s_rm.get("league","La Liga"),"league_key":"la_liga","team":"Real Madrid","games":_ser_games(s_rm.get("games",[]))}
    if s_mu.get("success"):
        out["sports_mu"]  = {"success":True,"league":s_mu.get("league","Premier League"),"league_key":"premier_league","team":"Manchester United","games":_ser_games(s_mu.get("games",[]))}
    if s_gsw.get("success"):
        out["sports_gsw"] = {"success":True,"league":s_gsw.get("league","NBA"),"league_key":"nba","team":"Golden State Warriors","games":_ser_games(s_gsw.get("games",[]))}
    if s_friendly.get("success") and s_friendly.get("games"):
        out["sports_friendly"] = {"success":True,"league":"Club Friendlies","league_key":"club_friendlies","games":_ser_games(s_friendly.get("games",[]))}

    # Consolidated cricket (ESPNcricinfo current matches)
    if s_cricket.get("success") and s_cricket.get("games"):
        out["sports_cricket"] = {
            "success": True, "league": "Cricket", "league_key": "cricket",
            "games": [{
                "home_team": g.get("home_team",""), "away_team": g.get("away_team",""),
                "home_score": g.get("home_score",""), "away_score": g.get("away_score",""),
                "status": g.get("status",""), "date_str": g.get("date_str",""),
                "date_iso": g.get("date_iso",""), "clock": g.get("clock",""),
                "home_color":"", "away_color":"",
                "home_logo": g.get("home_logo",""), "away_logo": g.get("away_logo",""),
                "home_innings": g.get("home_innings",[]), "away_innings": g.get("away_innings",[]),
                "note": g.get("note",""), "format": g.get("format",""),
            } for g in s_cricket.get("games", [])],
        }

    # Individual / event sports (tennis, combat, F1) — pass typed dicts through
    _IF = ("type","status","date_str","date_iso","home_team","away_team","home_score","away_score",
           "detail","name","circuit","results","event_name","fighter1","fighter2",
           "fighter1_logo","fighter2_logo","weight","fighter1_record","fighter2_record","card_note",
           "tournament","round","player1","player2","score1","score2")
    def _si(raw): return [{k: g.get(k) for k in _IF if k in g} for g in raw]
    _tn = (_si(s_tennis.get("games",[]) if s_tennis.get("success") else [])
         + _si(s_tennis_wta.get("games",[]) if s_tennis_wta.get("success") else []))
    if _tn:
        out["sports_tennis"] = {"success":True,"league":"Tennis","league_key":"tennis","games":_tn}
    if s_ufc.get("success") and s_ufc.get("games"):
        out["sports_ufc"] = {"success":True,"league":s_ufc.get("league","UFC"),"league_key":"ufc","games":_si(s_ufc.get("games",[]))}
    if s_boxing.get("success") and s_boxing.get("games"):
        out["sports_boxing"] = {"success":True,"league":s_boxing.get("league","Boxing"),"league_key":"boxing","games":_si(s_boxing.get("games",[]))}
    if s_f1.get("success") and s_f1.get("games"):
        out["sports_f1"] = {"success":True,"league":s_f1.get("league","Formula 1"),"league_key":"f1","games":_si(s_f1.get("games",[]))}

    return SafeJSONResponse(out)


@app.get("/spotify/now-playing")
async def spotify_now_playing():
    """Lightweight endpoint polled by UI every 10s for now playing widget."""
    try:
        data = await jarvis.spotify.get_now_playing()
        return SafeJSONResponse(data)
    except Exception as e:
        return SafeJSONResponse({"success": False, "error": str(e)})


@app.post("/spotify/play")
async def spotify_play():
    try:
        data = await jarvis.spotify.play()
        return SafeJSONResponse(data)
    except Exception as e:
        return SafeJSONResponse({"success": False, "error": str(e)})


@app.post("/spotify/pause")
async def spotify_pause():
    try:
        data = await jarvis.spotify.pause()
        return SafeJSONResponse(data)
    except Exception as e:
        return SafeJSONResponse({"success": False, "error": str(e)})


@app.post("/spotify/next")
async def spotify_next():
    try:
        data = await jarvis.spotify.skip()
        return SafeJSONResponse(data)
    except Exception as e:
        return SafeJSONResponse({"success": False, "error": str(e)})


@app.post("/spotify/previous")
async def spotify_previous():
    try:
        data = await jarvis.spotify.previous()
        return SafeJSONResponse(data)
    except Exception as e:
        return SafeJSONResponse({"success": False, "error": str(e)})


@app.post("/spotify/volume")
async def spotify_volume(level: int = 50):
    try:
        data = await jarvis.spotify.set_volume(level)
        return SafeJSONResponse(data)
    except Exception as e:
        return SafeJSONResponse({"success": False, "error": str(e)})


@app.get("/hardware")
async def hardware():
    """Return system hardware info — CPU, memory, network, thermal, battery, wifi, disk."""
    import subprocess, asyncio, re, time
    loop = asyncio.get_event_loop()

    async def run(cmd):
        try:
            result = await loop.run_in_executor(None,
                lambda: subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5))
            return result.stdout.strip()
        except Exception:
            return ""

    # ── Battery ────────────────────────────────────────────────────────────
    bat_raw = await run("pmset -g batt | grep -Eo '[0-9]+%' | head -1")
    battery = bat_raw.replace('%','') if bat_raw else None

    # ── Wi-Fi ──────────────────────────────────────────────────────────────
    iface_raw = await run("networksetup -listallhardwareports | awk '/Wi-Fi/{getline; print $2}'")
    # Sanitize: this value is interpolated into further shell commands below.
    # It comes from networksetup output (not user input), but validating it
    # keeps the only shell=True path in the codebase injection-proof even if
    # that output is ever weird (localized text, error strings, etc.).
    wifi_iface = iface_raw.strip() or "en0"
    import re as _re_iface
    if not _re_iface.fullmatch(r"[A-Za-z0-9]{1,16}", wifi_iface):
        wifi_iface = "en0"
    wifi_raw = await run(f"networksetup -getairportnetwork {wifi_iface}")
    if ":" in wifi_raw:
        wifi_name = wifi_raw.split(":", 1)[-1].strip()
        wifi = wifi_name if wifi_name else "Connected"
    else:
        wifi = "Connected"

    # ── Disk ───────────────────────────────────────────────────────────────
    disk_raw = await run("df -h / | tail -1 | awk '{print $3, $4, $5}'")
    parts = disk_raw.split() if disk_raw else []
    disk_used = parts[0] if parts else None
    disk_free = parts[1] if len(parts) > 1 else None
    disk_pct  = parts[2].rstrip('%') if len(parts) > 2 else None

    # ── CPU usage (single sample of top) ───────────────────────────────────
    cpu_raw = await run("top -l 1 -n 0 | awk '/CPU usage/ {print $3, $5}'")
    # e.g. "2.41% 5.10%" → user + sys
    cpu_pct = None
    try:
        nums = re.findall(r"([\d.]+)%", cpu_raw)
        if nums:
            cpu_pct = round(sum(float(n) for n in nums), 1)
    except Exception:
        cpu_pct = None

    # ── Memory ─────────────────────────────────────────────────────────────
    # Total RAM in bytes
    mem_total_raw = await run("sysctl -n hw.memsize")
    try:
        mem_total_bytes = int(mem_total_raw) if mem_total_raw else 0
    except Exception:
        mem_total_bytes = 0
    mem_total_gb = round(mem_total_bytes / (1024 ** 3), 1) if mem_total_bytes else None

    # Active + wired memory from vm_stat
    vm_raw = await run("vm_stat | head -20")
    page_size = 4096
    pm = re.search(r"page size of (\d+) bytes", vm_raw)
    if pm:
        page_size = int(pm.group(1))
    def _vm(field):
        m = re.search(rf"{field}:\s+([\d]+)\.", vm_raw)
        return int(m.group(1)) if m else 0
    used_pages = _vm("Pages active") + _vm("Pages wired down") + _vm("Pages occupied by compressor")
    mem_used_bytes = used_pages * page_size
    mem_used_gb = round(mem_used_bytes / (1024 ** 3), 1) if mem_used_bytes else None
    mem_pct = round(mem_used_bytes / mem_total_bytes * 100, 1) if mem_total_bytes else None

    # ── Network throughput (delta over 1s) ─────────────────────────────────
    async def _netbytes():
        out = await run(f"netstat -ibn | awk '$1==\"{wifi_iface}\" {{ib+=$7; ob+=$10}} END {{print ib, ob}}'")
        try:
            ib, ob = out.split()
            return int(ib), int(ob)
        except Exception:
            return 0, 0
    n1_in, n1_out = await _netbytes()
    await asyncio.sleep(1.0)
    n2_in, n2_out = await _netbytes()
    rx_bps = max(0, n2_in - n1_in)   # bytes/sec
    tx_bps = max(0, n2_out - n1_out)

    def _fmt_bps(n):
        if n < 1024: return f"{n} B/s"
        if n < 1024 ** 2: return f"{n/1024:.1f} KB/s"
        if n < 1024 ** 3: return f"{n/(1024**2):.1f} MB/s"
        return f"{n/(1024**3):.2f} GB/s"

    # ── Thermal (CPU temp) ─────────────────────────────────────────────────
    # Apple Silicon doesn't expose temps without sudo/extra tools; we try a
    # few options and fall back gracefully.
    therm_c = None
    therm_raw = await run("osx-cpu-temp 2>/dev/null | head -1")
    if therm_raw:
        m = re.search(r"([\d.]+)", therm_raw)
        if m:
            therm_c = round(float(m.group(1)), 1)
    if therm_c is None:
        # iStats / smc fallback (not always installed)
        alt = await run("istats cpu temp --value-only 2>/dev/null")
        m = re.search(r"([\d.]+)", alt or "")
        if m:
            therm_c = round(float(m.group(1)), 1)
    if therm_c is None:
        # Synthetic fallback: rough mapping from CPU% so the gauge moves
        therm_c = round(38 + (cpu_pct or 0) * 0.4, 1) if cpu_pct is not None else None

    return SafeJSONResponse({
        "battery": int(battery) if battery and battery.isdigit() else None,
        "wifi": wifi,
        "disk_used": disk_used,
        "disk_free": disk_free,
        "disk_pct": int(disk_pct) if disk_pct and disk_pct.isdigit() else None,
        "cpu_pct": cpu_pct,
        "mem_used_gb": mem_used_gb,
        "mem_total_gb": mem_total_gb,
        "mem_pct": mem_pct,
        "net_rx_bps": rx_bps,
        "net_tx_bps": tx_bps,
        "net_rx_human": _fmt_bps(rx_bps),
        "net_tx_human": _fmt_bps(tx_bps),
        "thermal_c": therm_c,
    })


# ── Google OAuth diagnostics ──────────────────────────────────────────────────

@app.get("/google/status")
async def google_status():
    """
    Return the connection state of the Gmail + Calendar agents.

    Used by the UI's Inbox / Calendar widgets to show a meaningful message
    when something's broken — e.g. "token refresh failed: invalid_grant"
    instead of a generic "Connect Gmail via .env".
    """
    return SafeJSONResponse({
        "gmail": {
            "connected": not jarvis.gmail.is_mock,
            "error": getattr(jarvis.gmail, "auth_error", None),
        },
        "calendar": {
            "connected": not jarvis.calendar.is_mock,
            "error": getattr(jarvis.calendar, "auth_error", None),
        },
    })


@app.post("/google/reauth")
async def google_reauth():
    """
    Force a fresh OAuth flow for both Gmail and Calendar.

    Deletes the saved token.json so the next agent initialisation triggers
    the interactive `flow.run_local_server` flow — your browser will open
    to the Google consent screen. Use this when token refresh fails.
    """
    try:
        from config.settings import GOOGLE_TOKEN_PATH
        if GOOGLE_TOKEN_PATH.exists():
            GOOGLE_TOKEN_PATH.unlink()
        # Re-initialise both agents in place so the next /sidebar call
        # picks up the new service. This will block until the user
        # completes the browser-based OAuth consent.
        from agents.calendar_agent import CalendarAgent
        from agents.gmail_agent import GmailAgent

        # This endpoint is an explicit user action, so it is the one place the
        # interactive browser flow is sanctioned. Elsewhere the agents refuse
        # to open it, because doing so from a request handler blocks the whole
        # event loop until someone clicks through the consent screen.
        #
        # Run it in a worker thread for the same reason: run_local_server()
        # binds a port and waits on a human, and awaiting that inline would
        # freeze every other request while the user reads the consent page.
        import os as _os_reauth
        _os_reauth.environ["JARVIS_INTERACTIVE_OAUTH"] = "true"
        try:
            jarvis.calendar = await asyncio.wait_for(
                asyncio.to_thread(CalendarAgent), timeout=300)
            jarvis.gmail = await asyncio.wait_for(
                asyncio.to_thread(GmailAgent), timeout=300)
        except asyncio.TimeoutError:
            return SafeJSONResponse({
                "success": False,
                "error": "OAuth consent not completed within 5 minutes.",
            })
        finally:
            _os_reauth.environ.pop("JARVIS_INTERACTIVE_OAUTH", None)
        return SafeJSONResponse({
            "success": True,
            "gmail_connected": not jarvis.gmail.is_mock,
            "calendar_connected": not jarvis.calendar.is_mock,
            "gmail_error": getattr(jarvis.gmail, "auth_error", None),
            "calendar_error": getattr(jarvis.calendar, "auth_error", None),
        })
    except Exception as e:
        return SafeJSONResponse({"success": False, "error": str(e)})


# ── Reminders ─────────────────────────────────────────────────────────────────
# Backed by the same ReminderStore the orchestrator uses, so anything created
# via voice/chat ("remind me to call mum in 10 minutes") shows up in the UI
# widget on the next refresh, and vice versa.

class ReminderCreate(BaseModel):
    title: str
    body: str = ""
    due_at: str | None = None          # ISO datetime; mutually exclusive with offset_minutes
    offset_minutes: int | None = None  # "in N minutes from now"
    recurring_minutes: int | None = None


@app.get("/reminders")
async def list_reminders():
    """Return all pending reminders (uncompleted), oldest-due first."""
    try:
        pending = jarvis.reminders.list_pending()
        # The DB returns timestamps as raw ISO strings — the UI wants the
        # field names it already uses. Map them once here so the front-end
        # stays simple.
        return SafeJSONResponse({
            "success": True,
            "reminders": [
                {
                    "id": r["id"],
                    "title": r["title"],
                    "body": r.get("body", ""),
                    "due_at": r["due_at"],
                    "recurring_minutes": r.get("recurring_minutes"),
                    "completed": bool(r.get("completed", 0)),
                    "created_at": r.get("created_at", ""),
                }
                for r in pending
            ],
        })
    except Exception as e:
        return SafeJSONResponse({"success": False, "error": str(e), "reminders": []})


@app.post("/reminders")
async def create_reminder(req: ReminderCreate):
    """Create a new reminder. Returns the new reminder ID."""
    try:
        rid = jarvis.reminders.add(
            title=req.title,
            body=req.body or "",
            due_at=req.due_at,
            offset_minutes=req.offset_minutes,
            recurring_minutes=req.recurring_minutes,
        )
        return SafeJSONResponse({"success": True, "id": rid})
    except Exception as e:
        return SafeJSONResponse({"success": False, "error": str(e)})


@app.delete("/reminders/{reminder_id}")
async def delete_reminder(reminder_id: str):
    """Permanently delete a reminder by ID."""
    try:
        ok = jarvis.reminders.delete(reminder_id)
        if not ok:
            return SafeJSONResponse({"success": False, "error": "Reminder not found"})
        return SafeJSONResponse({"success": True})
    except Exception as e:
        return SafeJSONResponse({"success": False, "error": str(e)})


@app.post("/reminders/{reminder_id}/complete")
async def complete_reminder(reminder_id: str):
    """Mark a reminder as complete (it'll stop being returned by /reminders)."""
    try:
        ok = jarvis.reminders.complete(reminder_id)
        if not ok:
            return SafeJSONResponse({"success": False, "error": "Reminder not found"})
        return SafeJSONResponse({"success": True})
    except Exception as e:
        return SafeJSONResponse({"success": False, "error": str(e)})


@app.post("/upload")
async def upload_document(file: UploadFile):
    """Accept document upload and store for analysis."""
    import tempfile, os
    try:
        content_bytes = await file.read()
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content_bytes)
            tmp_path = tmp.name
        # Store path for Jarvis to use
        jarvis._last_uploaded_doc = tmp_path
        jarvis._last_uploaded_name = file.filename
        return SafeJSONResponse({"success": True, "filename": file.filename, "path": tmp_path})
    except Exception as e:
        return SafeJSONResponse({"success": False, "error": str(e)})


# ── FinEx UI ──────────────────────────────────────────────────────────────────

@app.get("/finex")
async def finex_ui():
    """Serve the FinEx financial analysis dashboard."""
    return FileResponse(UI_DIR / "finex.html")


# ── FinEx Financial Statement Endpoints ───────────────────────────────────────

class FinExChatRequest(BaseModel):
    question: str
    company: str = "Bestway Cement"
    history: list = []
    # Optional explicit level override from the UI: 1–6 forces that reasoning
    # level; None / "auto" / 0 leaves it to the deterministic router.
    level: Any = None


@app.post("/finex/chat")
async def finex_chat(req: FinExChatRequest):
    """Financial statement Q&A — powered by the FinEx engine (6 reasoning levels)."""
    result = await finex.chat(req.question, req.company, req.history, req.level)
    return SafeJSONResponse(result)


@app.post("/finex/upload")
async def finex_upload(
    file: UploadFile,
    company: str = Form("Bestway Cement"),
):
    """Upload a PDF financial statement, extract data, and store in Postgres."""
    import tempfile, os
    tmp_path = None
    try:
        content = await file.read()
        if not content:
            return SafeJSONResponse({"success": False, "error": "Uploaded file is empty."})
        suffix = os.path.splitext(file.filename or "upload")[1] or ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        print(f"💹 FinEx upload: company={company!r}  file={file.filename!r}  tmp={tmp_path}")
        result = await finex.upload_pdf(tmp_path, company)
        print(f"💹 FinEx result: success={result.get('success')}  error={result.get('error','—')}")
        return SafeJSONResponse(result)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"💹 FinEx upload exception:\n{tb}")
        return SafeJSONResponse({"success": False, "error": str(e), "traceback": tb})
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


@app.get("/finex/companies")
async def finex_companies():
    """List all companies and periods stored in the FinEx database."""
    result = await finex.list_companies()
    return SafeJSONResponse(result)


# Market symbol groups for the FinEx financial dashboard
_INDICES = {
    "^GSPC":  "S&P 500",
    "^FTSE":  "FTSE 100",
    "^DJI":   "Dow Jones",
    "^IXIC":  "Nasdaq",
    "^N225":  "Nikkei 225",
}
_CRYPTO = {
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "BNB-USD": "BNB",
    "XRP-USD": "XRP",
    "SOL-USD": "Solana",
}
_COMMODITIES = {
    "GLD":      "Gold ETF",
    "USO":      "Oil ETF",
    "GBPUSD=X": "GBP/USD",
    "EURUSD=X": "EUR/USD",
    "JPYUSD=X": "JPY/USD",
}
_TECH = {
    "AAPL":  "Apple",
    "NVDA":  "NVIDIA",
    "MSFT":  "Microsoft",
    "GOOGL": "Alphabet",
    "META":  "Meta",
}
_FINANCE_STOCKS = {
    "JPM": "JPMorgan",
    "GS":  "Goldman Sachs",
    "BAC": "Bank of America",
    "V":   "Visa",
    "MA":  "Mastercard",
}
_ENERGY_HEALTH = {
    "XOM": "Exxon Mobil",
    "CVX": "Chevron",
    "JNJ": "Johnson & Johnson",
    "PFE": "Pfizer",
    "UNH": "UnitedHealth",
}

# Finance-specific RSS feeds for the FinEx news panel
_FINANCE_FEEDS = {
    "reuters_biz":  ("Reuters Business",  "https://feeds.reuters.com/reuters/businessNews"),
    "cnbc":         ("CNBC",              "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
    "yahoo_fin":    ("Yahoo Finance",     "https://finance.yahoo.com/news/rssindex"),
    "marketwatch":  ("MarketWatch",       "https://feeds.marketwatch.com/marketwatch/topstories"),
    "ft":           ("Financial Times",   "https://www.ft.com/rss/home"),
    "investopedia": ("Investopedia",      "https://www.investopedia.com/feeds/news.xml"),
}


async def _fetch_finance_news() -> dict:
    """Fetch headlines from finance-specific RSS feeds."""
    import xml.etree.ElementTree as ET
    TIMEOUT = aiohttp.ClientTimeout(total=8)
    HEADERS = {"User-Agent": "Jarvis/1.0 Python/aiohttp"}
    stories = []

    async def _one(key, name, url):
        try:
            async with aiohttp.ClientSession(timeout=TIMEOUT, headers=HEADERS) as s:
                async with s.get(url) as r:
                    if r.status != 200:
                        return []
                    text = await r.text()
            root = ET.fromstring(text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            items = root.findall(".//item") or root.findall(".//atom:entry", ns)
            out = []
            for item in items[:6]:
                title = (item.findtext("title") or item.findtext("atom:title", namespaces=ns) or "").strip()
                desc  = (item.findtext("description") or item.findtext("atom:summary", namespaces=ns) or "").strip()
                link  = (item.findtext("link") or item.findtext("atom:link", namespaces=ns) or "").strip()
                if title:
                    out.append({"title": title, "description": desc[:200], "source": name, "url": link})
            return out
        except Exception:
            return []

    tasks = [_one(k, n, u) for k, (n, u) in _FINANCE_FEEDS.items()]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for r in results:
        if isinstance(r, list):
            stories.extend(r)
    return {"success": True, "stories": stories[:30]}


@app.get("/finex/sidebar")
async def finex_sidebar():
    """
    All financial widget data for the FinEx dashboard — fetched in parallel.
    Returns: indices, crypto, commodities, tech stocks, finance stocks,
             energy/health stocks, financial news, and company list.
    """
    async def safe(coro):
        try:
            return await coro
        except Exception:
            return {"success": False, "prices": []}

    (indices, crypto, commodities, tech, fin_stocks, energy_health,
     fin_news, companies) = await asyncio.gather(
        safe(jarvis.markets.get_all(_INDICES)),
        safe(jarvis.markets.get_all(_CRYPTO)),
        safe(jarvis.markets.get_all(_COMMODITIES)),
        safe(jarvis.markets.get_all(_TECH)),
        safe(jarvis.markets.get_all(_FINANCE_STOCKS)),
        safe(jarvis.markets.get_all(_ENERGY_HEALTH)),
        safe(_fetch_finance_news()),
        safe(finex.list_companies()),
    )

    return SafeJSONResponse({
        "indices":       indices,
        "crypto":        crypto,
        "commodities":   commodities,
        "tech":          tech,
        "fin_stocks":    fin_stocks,
        "energy_health": energy_health,
        "fin_news":      fin_news,
        "companies":     companies,
    })


@app.get("/finex/markets-tick")
async def finex_markets_tick():
    """
    Lightweight live-refresh endpoint for the FinEx market widgets.

    Returns only the six price baskets — indices, crypto, commodities, tech,
    finance stocks, energy/health. Skips finance news and the company list
    which change much less often and add unnecessary latency to a poll
    that runs every 20 seconds.

    The FinEx UI calls this aggressively to keep tickers live while the
    full /finex/sidebar payload is fetched once on page load and then
    only when news/companies actually need to refresh.
    """
    async def safe(coro):
        try:
            return await coro
        except Exception:
            return {"success": False, "prices": []}

    indices, crypto, commodities, tech, fin_stocks, energy_health = await asyncio.gather(
        safe(jarvis.markets.get_all(_INDICES)),
        safe(jarvis.markets.get_all(_CRYPTO)),
        safe(jarvis.markets.get_all(_COMMODITIES)),
        safe(jarvis.markets.get_all(_TECH)),
        safe(jarvis.markets.get_all(_FINANCE_STOCKS)),
        safe(jarvis.markets.get_all(_ENERGY_HEALTH)),
    )

    return SafeJSONResponse({
        "indices":       indices,
        "crypto":        crypto,
        "commodities":   commodities,
        "tech":          tech,
        "fin_stocks":    fin_stocks,
        "energy_health": energy_health,
    })


# ── JAMS (Job Application Management System) — n8n Job HUD bridge ────────
# JARVIS is the control plane; JAMS keeps running as its own n8n process. These
# routes serve the real Job HUD UI inside Jarvis (blue theme) and proxy its data
# + action webhooks to n8n so every control keeps working. Server-to-server (no
# browser CORS). GET /jams/data is read-only; the action endpoints simply relay
# the same POSTs the native HUD makes to n8n (they never write the Sheet here).
JAMS_N8N_BASE = os.getenv("JAMS_N8N_BASE_URL", "http://localhost:5678").rstrip("/")
JAMS_WEBHOOK_PREFIX = os.getenv("JAMS_WEBHOOK_PREFIX", "/webhook")


async def _jams_get(path: str, total: float = 8.0):
    url = f"{JAMS_N8N_BASE}{JAMS_WEBHOOK_PREFIX}/{path}"
    timeout = aiohttp.ClientTimeout(total=total)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        async with s.get(url, headers={"cache-control": "no-cache"}) as r:
            return r.status, await r.text()


async def _jams_post(path: str, payload, total: float = 200.0):
    url = f"{JAMS_N8N_BASE}{JAMS_WEBHOOK_PREFIX}/{path}"
    timeout = aiohttp.ClientTimeout(total=total)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        async with s.post(url, json=payload) as r:
            return r.status, await r.text()


# ── Reading from JAMS ───────────────────────────────────────────────────────
#
# Every read below goes through _jams_read, because the interesting failure is
# one that doesn't look like a failure. A webhook node set to `responseMode:
# responseNode` only produces a body if the run reaches its Respond node; when
# a node throws first, n8n closes the request with **200 OK and zero bytes**.
# Parsed with a bare json.loads inside a try/except, that becomes `{"jobs":
# []}` — indistinguishable from a genuinely empty job board.
#
# That is not hypothetical: a revoked Google Sheets credential produced exactly
# this for four days, across 330 failed runs, while /jams/workflows reported
# every webhook healthy (it was — the *workflow* wasn't) and the UI showed a
# green "✓ Updated" over an empty board. An empty body is now its own case,
# and n8n's own execution log supplies the reason.


async def _jams_explain_empty(path: str) -> str:
    """Ask n8n why the workflow behind `path` produced no response body."""
    generic = ("n8n accepted the request but returned an empty response, which "
               "means a node failed before the workflow could reply. Open the "
               "workflow's Executions tab in n8n for the failing step.")
    try:
        from tools.n8n_trigger import last_failure, webhook_workflow_ids
        ids = await asyncio.to_thread(webhook_workflow_ids, _N8N_DB)
        if not ids or path not in ids:
            return generic
        fail = await asyncio.to_thread(last_failure, _N8N_DB, ids[path])
    except Exception:  # noqa: BLE001
        return generic
    if not fail or not fail.get("message"):
        return generic
    where = f" at '{fail['node']}'" if fail.get("node") else ""
    detail = f" {fail['description']}" if fail.get("description") else ""
    return f"n8n workflow failed{where}: {fail['message']}{detail}"


async def _jams_read(path: str, total: float = 8.0):
    """GET a JAMS read webhook. Returns (payload, error) — exactly one is None.

    Callers must not treat a None payload as empty data; that conflation is
    the whole bug this function exists to prevent.
    """
    try:
        status, body = await _jams_get(path, total=total)
    except asyncio.TimeoutError:
        return None, (f"n8n did not respond within {total:.0f}s. The workflow "
                      f"may be running slowly, or Google API calls are hanging.")
    except Exception as exc:  # noqa: BLE001
        return None, (f"Can't reach n8n at {JAMS_N8N_BASE} "
                      f"({type(exc).__name__}). Is it running?")

    if status != 200:
        hint = (" The workflow exists but may not be Active."
                if status == 404 else "")
        return None, f"n8n returned HTTP {status} for '{path}'.{hint}"

    if not (body or "").strip():
        return None, await _jams_explain_empty(path)

    try:
        data = _json.loads(body)
    except ValueError:
        return None, (f"n8n returned 200 for '{path}' but the body isn't JSON "
                      f"({(body or '')[:120]!r}).")
    if not isinstance(data, dict):
        return None, f"n8n returned JSON for '{path}' that isn't an object."
    return data, None


# ── Rationing the Google Sheets quota ───────────────────────────────────────
#
# One hud-data run reads three sheets (Jobs, Pipeline, Inbox), and Google
# allows 60 read requests per minute per user. That ceiling is ~20 hud-data
# calls a minute, which sounds generous until two "Run workflow" buttons are
# each polling for their own results every 6 seconds and the page is open in
# two tabs. Observed: a jump from ~1 run/hour to 16 runs/minute, and a board
# that had just been fixed started reporting "Quota exceeded".
#
# Nothing upstream deduplicates this, so Jarvis does: one in-flight fetch at a
# time (callers that arrive during it await the same result rather than
# starting their own), a short TTL so a burst of pollers costs one read, and
# last-good data served if a fetch fails. The whole point is that reading more
# often must not read Google more often.

_JAMS_CACHE_TTL = float(os.getenv("JAMS_CACHE_TTL_S", "10"))
# How long last-good data may still be shown after a failed refresh. Beyond
# this the data is too old to pass off as current, and the error wins.
_JAMS_STALE_CEILING = float(os.getenv("JAMS_STALE_MAX_S", "300"))

_QUOTA_MARKERS = ("quota exceeded", "too many requests", "rate limit",
                  "ratelimitexceeded", "resource_exhausted")


def _is_transient(error: str) -> bool:
    """Quota and timeout failures pass on their own; credential ones don't.

    Worth distinguishing because the right response differs: back off and keep
    showing the last good board for the first, say so loudly for the second.
    """
    low = (error or "").lower()
    return (any(m in low for m in _QUOTA_MARKERS)
            or "did not respond within" in low)


class _JamsHudCache:
    """TTL + single-flight cache over the hud-data read."""

    def __init__(self) -> None:
        self._data: Optional[Dict[str, Any]] = None
        self._fetched_at = 0.0
        self._lock = asyncio.Lock()

    def _fresh(self, ttl: float) -> bool:
        return (self._data is not None
                and (_time.monotonic() - self._fetched_at) < ttl)

    def invalidate(self) -> None:
        """Drop the cached board.

        Called after firing a workflow: discovery rewrites the Jobs sheet, and
        without this the next poll can be served a copy from before the run
        started — which reads as "nothing happened".
        """
        self._data = None
        self._fetched_at = 0.0

    async def get(self, *, ttl: Optional[float] = None, total: float = 8.0):
        """Returns (data, error, meta). `meta` carries staleness info."""
        ttl = _JAMS_CACHE_TTL if ttl is None else ttl
        if self._fresh(ttl):
            return self._data, None, {"cached": True, "stale": False}

        async with self._lock:
            # Another caller may have refreshed while we waited on the lock —
            # that is the entire saving, so re-check before spending a read.
            if self._fresh(ttl):
                return self._data, None, {"cached": True, "stale": False}

            data, error = await _jams_read("hud-data", total=total)
            if data is not None:
                self._data = data
                self._fetched_at = _time.monotonic()
                return data, None, {"cached": False, "stale": False}

            age = _time.monotonic() - self._fetched_at
            if (self._data is not None and _is_transient(error)
                    and age < _JAMS_STALE_CEILING):
                # A rate-limit blip must not blank a board that was fine a
                # moment ago; show what we had and name the reason.
                return self._data, None, {"cached": True, "stale": True,
                                          "stale_age_s": round(age),
                                          "stale_reason": error}
            return None, error, {"cached": False, "stale": False}


_jams_hud_cache = _JamsHudCache()


@app.get("/jams")
async def jams_ui():
    """Serve the Job HUD UI inside Jarvis, with live pipeline data injected the
    same way the n8n 'Build HTML' node does (so the first paint is populated)."""
    from fastapi.responses import HTMLResponse
    tpl = (UI_DIR / "jams.html").read_text(encoding="utf-8")
    jobs, pipe, inbox = [], [], []
    data, error, meta = await _jams_hud_cache.get(total=8.0)
    if data is not None and meta.get("stale"):
        error = meta["stale_reason"]
    if data is not None:
        jobs = data.get("jobs", []) or []
        pipe = data.get("pipe", []) or []
        inbox = data.get("inbox", []) or []
    else:
        # Log the real reason rather than the exception type — "JSONDecodeError"
        # sent four days of debugging at a webhook that was working fine.
        print(f"🧭 JAMS: {error}")
    html = (tpl.replace("__JOBS__", _json.dumps(jobs))
               .replace("__PIPE__", _json.dumps(pipe))
               .replace("__INBOX__", _json.dumps(inbox))
               .replace("__ERROR__", _json.dumps(error)))
    return HTMLResponse(content=html)


@app.get("/jams/data")
async def jams_data():
    """Proxy JAMS's n8n hud-data webhook (live Jobs + Pipeline + Inbox).

    On failure this returns 502 with an `error` and no `jobs` key at all —
    deliberately not `{"jobs": []}`, so a caller that ignores `error` breaks
    loudly instead of quietly rendering an empty board.
    """
    data, error, meta = await _jams_hud_cache.get(total=8.0)
    if data is None:
        return SafeJSONResponse({"ok": False, "error": error}, status_code=502)
    payload = dict(data)
    payload.setdefault("jobs", [])
    payload.setdefault("pipe", [])
    payload.setdefault("inbox", [])
    payload["ok"] = True
    payload["cached"] = bool(meta.get("cached"))
    if meta.get("stale"):
        # 200 with real (if slightly old) rows, plus the reason — the board
        # stays usable through a rate-limit blip instead of going blank.
        payload["stale"] = True
        payload["stale_age_s"] = meta.get("stale_age_s")
        payload["error"] = meta.get("stale_reason")
    return SafeJSONResponse(payload)


async def _jams_action(request: _Request, path: str, total: float):
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    try:
        status, body = await _jams_post(path, payload, total=total)
    except Exception as e:  # noqa: BLE001
        return SafeJSONResponse(
            {"ok": False, "error": f"Can't reach JAMS n8n ({type(e).__name__}: {e})."},
            status_code=502,
        )
    try:
        data = _json.loads(body)
    except Exception:
        data = {"ok": status == 200, "raw": body}
    return SafeJSONResponse(data, status_code=status)


# ── Triggering JAMS workflows ───────────────────────────────────────────────
#
# The Refresh button only ever called hud-data, which *reads* the current Jobs
# and Inbox. Nothing in Jarvis could START a workflow, so discovery and
# response-checking had to be run by hand in the n8n UI.
#
# Which webhook paths exist depends on how the workflows were imported, and
# guessing wrong fails silently (n8n answers 404 for an unregistered webhook).
# So rather than hardcode a list, /jams/workflows probes the candidates and
# reports what is actually live — the page then only offers buttons that work.

# name -> (webhook path, human label, description, seconds to allow)
# Only workflows with their own webhook trigger belong here. n8n_07_actions
# has webhooks (mark-applied, pipeline-status) but they're REACTIVE — the UI
# calls them per job — so there is nothing there to "run".
JAMS_WORKFLOWS = {
    "discovery":  ("discovery", "Find jobs",
                   "Scrape the boards for new roles", 300.0),
    "responses":  ("responses", "Check responses",
                   "Read replies and update the pipeline", 180.0),
    "hud-data":   ("hud-data", "Refresh data",
                   "Re-read jobs and inbox", 15.0),
}
# Override or extend with JAMS_EXTRA_WEBHOOKS="name:path,other:path2"
for _pair in os.getenv("JAMS_EXTRA_WEBHOOKS", "").split(","):
    if ":" in _pair:
        _n, _p = _pair.split(":", 1)
        _n, _p = _n.strip(), _p.strip()
        if _n and _p:
            JAMS_WORKFLOWS[_n] = (_p, _n.replace("-", " ").title(), "Custom workflow", 300.0)


# `Path`/`os`, not the `_Path`/`_os` aliases — those are imported ~600 lines
# further down, so at this point in module execution they don't exist yet.
_N8N_DB = Path(os.getenv("N8N_DB_PATH",
                         os.path.expanduser("~/.n8n/database.sqlite")))


async def _jams_probe(path: str) -> Dict[str, Any]:
    """Is this webhook registered in n8n?

    Prefers n8n's own database over HTTP probing: it's authoritative, and it
    can tell "no such workflow" apart from "workflow exists but is inactive",
    which is the difference between "you never imported it" and "you forgot to
    flip the toggle". Falls back to an HTTP probe when the DB isn't readable
    (different host, permissions, n8n on Postgres).
    """
    from tools.n8n_trigger import classify_probe, registered_webhooks

    reg = await asyncio.to_thread(registered_webhooks, _N8N_DB)
    if reg is not None:
        if path in reg:
            return {"live": True, "method": reg[path], "via": "n8n database"}
        return {"live": False, "via": "n8n database",
                "reason": "no active workflow registers this webhook"}

    try:
        status, body = await _jams_get(path, total=4.0)
    except Exception as exc:  # noqa: BLE001
        return {"live": False, "reason": f"{type(exc).__name__}: {exc}"}
    return {**classify_probe(status, body), "via": "http probe"}


async def _jams_last_failure(path: str) -> Optional[Dict[str, Any]]:
    """The most recent failed run of the workflow behind `path`, if any."""
    try:
        from tools.n8n_trigger import last_failure, webhook_workflow_ids
        ids = await asyncio.to_thread(webhook_workflow_ids, _N8N_DB)
        if not ids or path not in ids:
            return None
        return await asyncio.to_thread(last_failure, _N8N_DB, ids[path])
    except Exception:  # noqa: BLE001
        return None


@app.get("/jams/workflows")
async def jams_workflows():
    """Which JAMS workflows are reachable — and whether they actually work.

    `live` answers "is a webhook registered", which is not the same question as
    "does this workflow succeed". Every webhook here stayed `live: True`
    through 330 consecutive failed runs, because registration is unaffected by
    a credential expiring. `last_failure` is the missing half.
    """
    probes = await asyncio.gather(
        *[_jams_probe(spec[0]) for spec in JAMS_WORKFLOWS.values()],
        return_exceptions=True)
    failures = await asyncio.gather(
        *[_jams_last_failure(spec[0]) for spec in JAMS_WORKFLOWS.values()],
        return_exceptions=True)
    out = []
    for (name, (path, label, desc, _t)), probe, fail in zip(
            JAMS_WORKFLOWS.items(), probes, failures):
        if isinstance(probe, Exception):
            probe = {"live": False, "reason": str(probe)}
        entry = {"name": name, "path": path, "label": label,
                 "description": desc, **probe}
        if isinstance(fail, dict) and fail.get("message"):
            entry["last_failure"] = fail
        out.append(entry)
    reachable = any(w["live"] for w in out)

    # A registered webhook in front of a failing workflow is the confusing
    # case — surface it ahead of the "is n8n even up" hint, which is the one
    # people already know how to check.
    broken = [w for w in out if w.get("live") and w.get("last_failure")]
    if broken:
        f = broken[0]["last_failure"]
        where = f" at '{f['node']}'" if f.get("node") else ""
        hint = (f"Webhooks are registered, but {broken[0]['label']} last failed"
                f"{where}: {f['message']} "
                f"(execution {f.get('execution_id')}, {f.get('started_at')})")
    elif reachable:
        hint = ""
    else:
        hint = (f"Nothing answered at {JAMS_N8N_BASE}{JAMS_WEBHOOK_PREFIX}/… — "
                f"is n8n running, and are the workflows Active? A workflow "
                f"saved but not activated has no registered webhook.")

    return SafeJSONResponse({
        "base": JAMS_N8N_BASE, "prefix": JAMS_WEBHOOK_PREFIX,
        "workflows": out, "n8n_reachable": reachable, "hint": hint,
    })


@app.post("/jams/trigger/{name}")
async def jams_trigger(name: str, request: _Request):
    """Fire a JAMS workflow by name."""
    spec = JAMS_WORKFLOWS.get(name)
    if not spec:
        return SafeJSONResponse(
            {"ok": False, "error": f"unknown workflow {name!r}",
             "known": sorted(JAMS_WORKFLOWS)}, status_code=404)
    path, label, _desc, timeout = spec
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    payload.setdefault("source", "jarvis")

    # Fire with the method n8n actually registered. A Webhook node bound to
    # GET answers a POST with the same 404 body as a path that doesn't exist
    # at all, so blindly POSTing turns a working workflow into "not found" —
    # which is precisely what made hud-data look unrunnable.
    probe = await _jams_probe(path)
    method = (probe.get("method") or "POST").upper()

    started = _time.time()
    try:
        if method == "GET":
            status, body = await _jams_get(path, total=timeout)
        else:
            status, body = await _jams_post(path, payload, total=timeout)
    except asyncio.TimeoutError:
        # A long discovery run legitimately outlives the HTTP request. It is
        # still running in n8n, so saying "failed" would be wrong.
        return SafeJSONResponse(
            {"ok": True, "running": True, "workflow": name,
             "message": f"{label} is still running in n8n after "
                        f"{timeout:.0f}s — it will finish in the background."})
    except Exception as exc:  # noqa: BLE001
        return SafeJSONResponse(
            {"ok": False, "workflow": name,
             "error": f"Can't reach n8n at {JAMS_N8N_BASE} "
                      f"({type(exc).__name__}). Is it running?"},
            status_code=502)

    try:
        data = _json.loads(body)
        if not isinstance(data, dict):
            data = {"result": data}
    except Exception:
        data = {"raw": (body or "")[:2000]}

    if status == 404 and "not registered" in (body or "").lower():
        return SafeJSONResponse(
            {"ok": False, "workflow": name,
             "error": f"n8n has no active webhook at '{path}'. The workflow "
                      f"exists but may not be Active — open it in n8n and "
                      f"toggle Active on."}, status_code=404)

    # The run may have rewritten the sheets, so the cached board is now a
    # picture of "before". Drop it rather than let a poller read it back and
    # conclude the run changed nothing.
    _jams_hud_cache.invalidate()

    data.update({"ok": status == 200, "workflow": name, "label": label,
                 "elapsed_ms": round((_time.time() - started) * 1000)})
    return SafeJSONResponse(data, status_code=200 if status == 200 else status)


@app.post("/jams/apply-job")
async def jams_apply_job(request: _Request):
    """Relay Tailor & Apply to JAMS's n8n Apply workflow (long-running: Ollama/Claude)."""
    return await _jams_action(request, "apply-job", total=200.0)


@app.post("/jams/pipeline-status")
async def jams_pipeline_status(request: _Request):
    """Relay a pipeline status upsert (viewed / submitted / dismissed) to n8n."""
    return await _jams_action(request, "pipeline-status", total=20.0)


@app.post("/jams/mark-applied")
async def jams_mark_applied(request: _Request):
    """Relay a mark-applied event to n8n."""
    return await _jams_action(request, "mark-applied", total=20.0)


# ── JARVIS skill endpoints (control plane for JAMS / OpenClaw) ───────────────
_SKILL_TOKEN = os.getenv("JARVIS_SKILL_TOKEN", "").strip()


def _skill_ok(request: _Request) -> bool:
    if not _SKILL_TOKEN:
        return True
    return request.headers.get("x-jarvis-skill-token", "") == _SKILL_TOKEN


_CV_CRITIC_SYSTEM = """You are the Critic for Jarvis, reviewing a TAILORED CV against a specific job description before it is sent. Judge only on evidence in the CV and JD.

Return VALID JSON only: {"approved": true, "score": 82, "jd_match": 0.0, "missing_keywords": ["..."], "issues": ["..."], "suggestions": ["..."], "replan_needed": false}

Rules: score 0-100; approved = score >= 70; replan_needed = score < 60; be specific."""


@app.post("/skills/review-cv")
async def skills_review_cv(request: _Request):
    """Quality gate for JAMS: score a tailored CV against a JD. Fails OPEN."""
    if not _skill_ok(request):
        return SafeJSONResponse({"error": "unauthorised"}, status_code=401)
    try:
        body = await request.json()
    except Exception:
        body = {}
    cv = (body.get("cv_md") or body.get("cv") or body.get("tailored_cv") or "").strip()
    jd = (body.get("job_description") or body.get("jd") or body.get("description") or "").strip()
    title = body.get("job_title") or body.get("title") or ""
    company = body.get("company") or ""
    if not cv:
        return SafeJSONResponse({"approved": False, "score": 0, "replan_needed": True,
                                 "issues": ["No CV text provided."], "error": "missing cv"})
    user = (f"ROLE: {title} at {company}\n\nJOB DESCRIPTION:\n{(jd[:6000] or '(none)')}\n\n"
            f"TAILORED CV:\n{cv[:9000]}\n\nReturn the JSON verdict.")
    messages = [{"role": "system", "content": _CV_CRITIC_SYSTEM}, {"role": "user", "content": user}]
    try:
        data = await jarvis.llm.chat_json(messages, max_tokens=800)
    except Exception as e:  # noqa: BLE001
        return SafeJSONResponse({"approved": True, "score": None, "issues": [], "suggestions": [],
                                 "replan_needed": False, "error": f"critic unavailable: {type(e).__name__}: {e}"})
    score = data.get("score")
    try:
        score = float(score)
    except (TypeError, ValueError):
        score = None
    approved = data.get("approved")
    if approved is None and score is not None:
        approved = score >= 70
    replan = data.get("replan_needed")
    if replan is None and score is not None:
        replan = score < 60
    return SafeJSONResponse({"approved": bool(approved) if approved is not None else True, "score": score,
                             "jd_match": data.get("jd_match"), "missing_keywords": data.get("missing_keywords", []),
                             "issues": data.get("issues", []), "suggestions": data.get("suggestions", []),
                             "replan_needed": bool(replan) if replan is not None else False,
                             "role": {"title": title, "company": company}})


@app.post("/skills/critic")
async def skills_critic(request: _Request):
    """Generic Critic pass over arbitrary content. Body: {text}."""
    if not _skill_ok(request):
        return SafeJSONResponse({"error": "unauthorised"}, status_code=401)
    try:
        body = await request.json()
    except Exception:
        body = {}
    text = (body.get("text") or body.get("content") or "").strip()
    if not text:
        return SafeJSONResponse({"error": "provide {text}"}, status_code=400)
    messages = [{"role": "system", "content": "You are the Critic. Return VALID JSON only: {\"approved\": bool, \"score\": 0-100, \"issues\": [], \"suggestions\": [], \"replan_needed\": bool}."},
                {"role": "user", "content": text[:12000]}]
    try:
        return SafeJSONResponse(await jarvis.llm.chat_json(messages, max_tokens=600))
    except Exception as e:  # noqa: BLE001
        return SafeJSONResponse({"approved": True, "score": None, "error": str(e)})


@app.post("/skills/planner")
async def skills_planner(request: _Request):
    """Expose the Planner as a skill. Body: {request}."""
    if not _skill_ok(request):
        return SafeJSONResponse({"error": "unauthorised"}, status_code=401)
    try:
        body = await request.json()
    except Exception:
        body = {}
    req = (body.get("request") or body.get("goal") or body.get("text") or "").strip()
    if not req:
        return SafeJSONResponse({"error": "provide {request}"}, status_code=400)
    try:
        plan = await jarvis.planner.plan(req)
        return SafeJSONResponse({"task_id": getattr(plan, "task_id", None), "intent": getattr(plan, "intent", None),
                                 "reasoning": getattr(plan, "reasoning", ""),
                                 "subtasks": [{"id": st.id, "action": st.action, "agent": st.agent,
                                               "params": st.params, "depends_on": st.depends_on}
                                              for st in getattr(plan, "subtasks", [])]})
    except Exception as e:  # noqa: BLE001
        return SafeJSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)


# ── Connections & Health console ─────────────────────────────────────────────
@app.get("/health")
async def health_ui():
    return FileResponse(UI_DIR / "health.html")


@app.get("/health/status")
async def health_status():
    """Aggregate live status of every JARVIS connection, agent and tool."""
    import time as _t

    async def _check_ollama():
        base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        try:
            timeout = aiohttp.ClientTimeout(total=4)
            async with aiohttp.ClientSession(timeout=timeout) as sess:
                async with sess.get(f"{base}/api/tags") as r:
                    if r.status != 200:
                        return {"connected": False, "base_url": base, "detail": f"HTTP {r.status}"}
                    d = await r.json()
                    models = [m.get("name", "") for m in d.get("models", [])]
                    return {"connected": True, "base_url": base, "models": models, "count": len(models)}
        except Exception as e:  # noqa: BLE001
            return {"connected": False, "base_url": base, "detail": f"{type(e).__name__}: {e}"}

    async def _check_jams():
        base = os.getenv("JAMS_N8N_BASE_URL", "http://localhost:5678").rstrip("/")
        d, error, meta = await _jams_hud_cache.get(total=5.0)
        if d is None:
            return {"connected": False, "base_url": base, "detail": error}
        out = {"connected": True, "base_url": base,
               "jobs": len(d.get("jobs", []) or []),
               "pipeline": len(d.get("pipe", []) or []),
               "inbox": len(d.get("inbox", []) or [])}
        if meta.get("stale"):
            out["detail"] = f"showing data from {meta['stale_age_s']}s ago — " \
                            f"{meta['stale_reason']}"
        return out

    async def _check_spotify():
        if not os.getenv("SPOTIFY_CLIENT_ID") or not os.getenv("SPOTIFY_CLIENT_SECRET"):
            return {"connected": False, "detail": "not configured (no SPOTIFY_CLIENT_ID / SECRET in .env)"}
        try:
            data = await jarvis.spotify.get_now_playing()
            if getattr(jarvis.spotify, "_refresh_failed_until", 0) > _t.time():
                return {"connected": False, "detail": "invalid_client — SPOTIFY_CLIENT_SECRET no longer matches the Spotify app"}
            if data.get("success"):
                return {"connected": True, "detail": "playing" if data.get("playing") else "connected (idle)"}
            return {"connected": False, "detail": data.get("error", "not authenticated")}
        except Exception as e:  # noqa: BLE001
            return {"connected": False, "detail": f"{type(e).__name__}: {e}"}

    def _google_token_info():
        try:
            from config.settings import GOOGLE_TOKEN_PATH
            if not GOOGLE_TOKEN_PATH.exists():
                return {"path": str(GOOGLE_TOKEN_PATH), "present": False}
            t = _json.loads(GOOGLE_TOKEN_PATH.read_text())
            return {"path": str(GOOGLE_TOKEN_PATH), "present": True, "has_refresh_token": bool(t.get("refresh_token")),
                    "scopes": t.get("scopes"), "expiry": t.get("expiry")}
        except Exception as e:  # noqa: BLE001
            return {"present": None, "detail": f"{type(e).__name__}: {e}"}

    ollama, jams, spotify = await asyncio.gather(_check_ollama(), _check_jams(), _check_spotify())

    def _conn(agent_attr):
        try:
            a = getattr(jarvis, agent_attr)
            return {"connected": not a.is_mock, "error": getattr(a, "auth_error", None)}
        except Exception as e:  # noqa: BLE001
            return {"connected": False, "error": f"{type(e).__name__}: {e}"}

    gmail_conn = _conn("gmail")
    cal_conn = _conn("calendar")

    def _ready(attr):
        try:
            return getattr(jarvis, attr, None) is not None
        except Exception:  # noqa: BLE001
            return False

    try:
        # Deliberately reads the instance rather than the proxy: asking the
        # proxy would construct FinEx, so a health check would start a
        # Postgres connection as a side effect of being looked at.
        finex_ready = bool(getattr(_finex_instance, "_ready", False))
    except Exception:  # noqa: BLE001
        finex_ready = False

    _AGENTS = [
        ("router", "Router", "Classifies intent and routes each request", _ready("router")),
        ("memory", "Memory", "ChromaDB long-term memory (HNSW / cosine)", _ready("memory")),
        ("planner", "Planner", "Decomposes requests into task DAGs", _ready("planner")),
        ("critic", "Critic", "Reviews plans / outputs, triggers replans", _ready("critic")),
        ("evaluator", "Evaluator", "Scores results, stores benchmarks", _ready("evaluator")),
        ("summariser", "Summariser", "Condenses text and long threads", _ready("summariser")),
        ("calendar", "Calendar", "Google Calendar events", cal_conn["connected"]),
        ("gmail", "Gmail", "Gmail read / send / draft / label", gmail_conn["connected"]),
        ("finex", "FinEx", "Financial-statement analysis (6 levels)", finex_ready),
    ]
    agents = [{"key": k, "name": n, "desc": d, "ready": bool(r)} for (k, n, d, r) in _AGENTS]

    _TOOLS = [
        ("weather", "Weather", "Open-Meteo forecast"), ("websearch", "Web Search", "DuckDuckGo + scrape"),
        ("news", "News", "RSS aggregation"), ("markets", "Markets", "yfinance quotes"),
        ("sports", "Sports", "ESPN + Cricinfo"), ("prayer", "Prayer Times", "AlAdhan"),
        ("spotify", "Spotify", "Playback control"), ("mac", "Mac Control", "Local macOS control"),
        ("document", "Documents", "Extract + summarise"), ("files", "File Manager", "Desktop / Docs / Downloads"),
        ("contacts", "Contacts", "Contact book"), ("composer", "Email Composer", "Drafts emails"),
        ("briefing", "Briefing", "Daily brief builder"), ("reminders", "Reminders", "SQLite reminder store"),
    ]
    tools = []
    for (k, n, d) in _TOOLS:
        entry = {"key": k, "name": n, "desc": d, "ready": _ready(k)}
        if k == "spotify":
            entry["ready"] = bool(spotify.get("connected"))
            entry["attention"] = not spotify.get("connected")
        tools.append(entry)

    return SafeJSONResponse({
        "connections": {"gmail": gmail_conn, "calendar": cal_conn, "google_token": _google_token_info(),
                        "spotify": spotify, "ollama": ollama, "jams": jams},
        "agents": agents, "tools": tools, "llm_model": getattr(jarvis.llm, "model", None),
    })


# ── Morning Brief ────────────────────────────────────────────────────────────
@app.get("/brief")
async def brief_ui():
    """Serve the revamped morning brief."""
    return FileResponse(UI_DIR / "brief.html")


@app.get("/brief/data")
async def brief_data():
    """Assemble the morning brief.

    Two rules shape this, both learned from the version it replaces:

    1. ACTION FIRST. The old brief showed everything it could find, which made
       a quiet day look identical to a day with an offer waiting. Everything
       that needs a decision is now ranked into a single `needs_you` list;
       everything else is reference material below it.

    2. NEVER STATE A NUMBER YOU DIDN'T COUNT. The summary used to be written
       by the local model from a stats blob, and it drifted — it once reported
       "ten interview threads" and "five applications in the pipeline" when
       the real figures were 0 and 51. A brief that misreports is worse than
       no brief, so the summary is now composed from the same counters that
       feed the tiles. It cannot disagree with them.
    """
    from datetime import datetime
    import re as _re

    async def _safe(coro, default):
        try:
            return await coro
        except Exception:
            return default

    async def _jams_brief():
        d, error, _meta = await _jams_hud_cache.get(total=6.0)
        if d is None:
            # The brief degrades to "no job data" either way, but carrying the
            # reason means the page can say why instead of implying zero jobs.
            return {"jobs": [], "pipe": [], "inbox": [], "ok": False,
                    "error": error}
        return {"jobs": d.get("jobs", []) or [], "pipe": d.get("pipe", []) or [],
                "inbox": d.get("inbox", []) or [], "ok": True, "error": None}

    weather, events_r, inbox_r, news_r, markets_r, prayer_r, jams = await asyncio.gather(
        _safe(jarvis.weather.get_current(), {}),
        _safe(jarvis.calendar.search_events(), {}),
        _safe(jarvis.gmail.get_inbox(max_results=15, query="is:inbox"), {}),
        _safe(jarvis.news.get_headlines(category="technology", max_stories=8,
                                        query="AI artificial intelligence technology"), {}),
        _safe(jarvis.markets.get_all(), {}),
        _safe(jarvis.prayer.get_times(), {}),
        _jams_brief(),
    )

    now = datetime.now()
    part = "morning" if now.hour < 12 else ("afternoon" if now.hour < 18 else "evening")
    name = os.getenv("JARVIS_USER_NAME", "Abdullah")
    greeting = f"Good {part}, {name}."
    try:
        date_line = now.strftime("%A, %-d %B")
    except Exception:
        date_line = now.strftime("%A, %d %B")

    wx = weather if (isinstance(weather, dict) and weather.get("temperature_c") is not None) else {}

    # ── Prayer ──────────────────────────────────────────────────────────────
    prayer = {}
    try:
        pr = prayer_r if isinstance(prayer_r, dict) else {}
        order = ("fajr", "dhuhr", "asr", "maghrib", "isha")
        times = {k: pr.get(k) for k in order if pr.get(k)}

        def _mins(sval):
            m = _re.search(r"(\d{1,2}):(\d{2})", str(sval))
            if not m:
                return None
            h, mm = int(m.group(1)), int(m.group(2))
            low = str(sval).lower()
            if "pm" in low and h < 12:
                h += 12
            if "am" in low and h == 12:
                h = 0
            return h * 60 + mm

        nowmin = now.hour * 60 + now.minute
        nxt = None
        for k in order:
            tv = _mins(times.get(k))
            if tv is not None and tv >= nowmin:
                nxt = {"name": k.capitalize(), "time": times.get(k)}
                break
        if nxt is None and times.get("fajr"):
            nxt = {"name": "Fajr", "time": times.get("fajr")}
        # Labelled and flagged, so the UI can render a schedule instead of the
        # unlabelled "03:14 · 13:10 · 17:23 · 21:00 · 23:04" run-on it had.
        schedule = [{"name": k.capitalize(), "time": times[k],
                     "past": (_mins(times[k]) is not None and _mins(times[k]) < nowmin),
                     "next": bool(nxt and nxt["name"].lower() == k)}
                    for k in order if k in times]
        prayer = {"next": nxt, "times": times, "schedule": schedule}
    except Exception:
        prayer = {}

    # ── Calendar ────────────────────────────────────────────────────────────
    today = now.date()
    events = (events_r.get("events") if isinstance(events_r, dict) else None) or []
    today_events, cal_interviews = [], []
    for e in events:
        st = e.get("start")
        try:
            sd = datetime.fromisoformat(str(st).replace("Z", "+00:00")) if st else None
        except Exception:
            sd = None
        if sd and sd.date() == today:
            today_events.append({"time": sd.strftime("%H:%M"), "title": e.get("title") or "Untitled",
                                 "location": e.get("location") or ""})
        if sd and "interview" in (e.get("title") or "").lower() and 0 <= (sd.date() - today).days <= 1:
            cal_interviews.append({"text": e.get("title"), "sub": sd.strftime("%a %H:%M"),
                                   "source": "calendar"})
    today_events.sort(key=lambda x: x["time"] or "99:99")

    cal_connected = not getattr(jarvis.calendar, "is_mock", True)
    gmail_connected = not getattr(jarvis.gmail, "is_mock", True)

    # ── Inbox triage ────────────────────────────────────────────────────────
    #
    # JAMS already drops digests and networking spam, but its bar is "might be
    # job-related", which is deliberately generous — a missed interview invite
    # costs more than a stray newsletter. The brief needs a stricter bar,
    # because anything shown here is claiming to need a decision. Marketing
    # from CV-writing services was clearing JAMS's filter on the strength of
    # words like "interviewed" and landing in Interviews & follow-ups.
    BRIEF_NOISE_FROM = ("topcv", "cvlibrary", "resume", "@testpartnership",
                        "fourthrev", "newsletter", "marketing@", "digest@",
                        "@medium.com", "@substack.com")
    BRIEF_NOISE_SUBJECT = ("3x more likely", "webinar", "masterclass", "free cv",
                           "cv review", "upgrade your", "% off", "career accelerator",
                           "i'd love for you to meet", "unsubscribe")
    # An automated confirmation that an application was forwarded is an
    # acknowledgement, whatever the classifier said. This guard exists because
    # a reed.co.uk no-reply reading "We've sent your application to …" was
    # labelled `offer` by the local model and rendered with a green OFFER
    # badge — the single most misleading thing the brief could do.
    ACK_PHRASES = ("we've sent your application", "weve sent your application",
                   "we have sent your application", "your application has been sent",
                   "application submitted", "we have received your application",
                   "thank you for applying", "application received")
    AUTOMATED_SENDER = ("no-reply", "noreply", "do-not-reply", "donotreply", "mailer-daemon")

    def _is_brief_noise(frm: str, subj: str) -> bool:
        f, s = frm.lower(), subj.lower()
        return (any(n in f for n in BRIEF_NOISE_FROM)
                or any(n in s for n in BRIEF_NOISE_SUBJECT))

    def _category(m) -> str:
        """Trust JAMS's label, but never let a wrong one become a green badge."""
        raw = str(m.get("category") or m.get("label") or m.get("reply_status") or "").lower().strip()
        subj = str(m.get("subject") or "").lower()
        frm = str(m.get("from") or "").lower()
        cat = next((k for k in ("offer", "interview", "rejected", "acknowledged",
                                "recruiter", "personal", "update") if k in raw), "")
        if cat in ("offer", "interview") and any(p in subj for p in ACK_PHRASES):
            return "acknowledged"
        if cat == "offer" and any(a in frm for a in AUTOMATED_SENDER):
            # Real offers come from a person. An automated sender claiming one
            # is a misclassification often enough that demoting it is right.
            return "acknowledged"
        return cat

    jinbox = jams["inbox"]
    inbox_rows = []
    for m in jinbox:
        subj = (m.get("subject") or "(no subject)").strip()
        frm = (m.get("from") or "").strip()
        if str(m.get("is_job", "yes")).lower() == "no":
            continue
        if _is_brief_noise(frm, subj):
            continue
        inbox_rows.append({"cat": _category(m), "subject": subj, "from": frm,
                           "link": m.get("link") or "", "company": m.get("company") or "",
                           "date": m.get("date") or ""})

    offers = [r for r in inbox_rows if r["cat"] == "offer"]
    interviews = [r for r in inbox_rows if r["cat"] == "interview"]
    recruiters = [r for r in inbox_rows if r["cat"] == "recruiter"]

    # ── Pipeline ────────────────────────────────────────────────────────────
    pipe = jams["pipe"]
    jobs = jams["jobs"]

    def _norm(r):
        rs = str(r.get("reply_status", "")).lower()
        for k in ("offer", "interview", "rejected", "acknowledged"):
            if k in rs:
                return k
        st = str(r.get("status", "")).lower().strip()
        if "submitted" in st or "applied" in st:
            return "applied"
        if st == "dismissed":
            return "dismissed"
        return st or "applied"

    counts = {}
    for r in pipe:
        counts[_norm(r)] = counts.get(_norm(r), 0) + 1
    pipeline_active = [r for r in pipe if _norm(r) != "dismissed"]

    # The tiles used to draw "interviews" from reply_status while the list
    # beside them drew from the inbox, so the page showed "0 INTERVIEWS" above
    # a list of interview invitations. One source now feeds both.
    interview_total = counts.get("interview", 0) + len(interviews) + len(cal_interviews)
    offer_total = counts.get("offer", 0) + len(offers)

    applied_keys = set()
    for r in pipe:
        if r.get("dedupe_key") and _norm(r) in ("applied", "interview", "offer", "acknowledged", "rejected"):
            applied_keys.add(r.get("dedupe_key"))
    new_high_fit = []
    for g in sorted(jobs, key=lambda x: float(x.get("match_score") or 0), reverse=True):
        if g.get("dedupe_key") in applied_keys:
            continue
        sc = g.get("match_score")
        try:
            sc = int(round(float(sc)))
        except Exception:
            sc = None
        if sc is not None and sc < 60:
            break
        new_high_fit.append({"title": g.get("title") or "Role", "company": g.get("company") or "",
                             "location": g.get("location_bucket") or g.get("location") or "",
                             "score": sc, "url": g.get("url") or "#"})
        if len(new_high_fit) >= 5:
            break

    follow_ups = []
    for r in pipe:
        if _norm(r) != "applied" or str(r.get("reply_status") or "").strip():
            continue
        ad = str(r.get("applied_date") or "")
        try:
            days = (today - datetime.fromisoformat(ad[:10]).date()).days
        except Exception:
            days = None
        if days is not None and days >= 5:
            follow_ups.append({"text": (str(r.get("company", "")) + " — " + str(r.get("title", ""))).strip(" —"),
                               "sub": f"applied {days}d ago, no reply", "days": days})
    follow_ups.sort(key=lambda x: -(x["days"] or 0))
    follow_ups = follow_ups[:6]

    jobs_block = {
        "discovered": len(jobs),
        "total_pipeline": len(pipeline_active),
        "counts": {
            "applied": sum(1 for r in pipe if _norm(r) in ("applied", "acknowledged")),
            "interview": interview_total,
            "offer": offer_total,
            "rejected": counts.get("rejected", 0),
        },
        "new_high_fit": new_high_fit,
        "follow_ups": follow_ups,
    }

    # ── Needs you: the only section that claims urgency ──────────────────────
    #
    # Ranked hardest-consequence first. Everything here is something you can
    # act on today; anything that is merely interesting lives further down.
    needs_you = []
    for r in offers:
        needs_you.append({"icon": "🎉", "text": r["subject"], "sub": r["from"],
                          "tag": "offer", "tagClass": "green", "href": r["link"], "rank": 0})
    for r in interviews:
        needs_you.append({"icon": "🗣️", "text": r["subject"], "sub": r["from"],
                          "tag": "interview", "tagClass": "cyan", "href": r["link"], "rank": 1})
    for i in cal_interviews:
        needs_you.append({"icon": "📅", "text": i["text"], "sub": i["sub"],
                          "tag": "scheduled", "tagClass": "cyan", "rank": 1})
    try:
        for r in jarvis.reminders.list_pending():
            if str(r.get("due_at") or "")[:10] == today.isoformat():
                needs_you.append({"icon": "⏰", "text": r.get("title") or "Reminder",
                                  "sub": (r.get("body") or "")[:70], "tag": "due",
                                  "tagClass": "amber", "rank": 2})
    except Exception:
        pass
    for r in recruiters:
        needs_you.append({"icon": "✉️", "text": r["subject"], "sub": r["from"],
                          "tag": "recruiter", "tagClass": "violet", "href": r["link"], "rank": 3})
    needs_you.sort(key=lambda x: x["rank"])
    for item in needs_you:
        item.pop("rank", None)

    # ── Digest ──────────────────────────────────────────────────────────────
    stories = (news_r.get("stories") if isinstance(news_r, dict) else None) or []
    news, seen_sources = [], {}
    for s in stories:
        title = s.get("title") or ""
        if not title:
            continue
        src = (list(s.get("sources") or [])[:1] or [""])[0]
        # At most two per outlet: the old brief regularly showed four
        # TechCrunch headlines and called it a digest.
        if seen_sources.get(src, 0) >= 2:
            continue
        seen_sources[src] = seen_sources.get(src, 0) + 1
        news.append({"title": title, "source": src, "url": s.get("url") or "#"})
        if len(news) >= 4:
            break

    mprices = (markets_r.get("prices") if isinstance(markets_r, dict) else None) or []
    markets = []
    for m in mprices[:4]:
        raw = m.get("change_pct")
        # `or 0` turned "no change data" into a green +0.00%, which reads as a
        # flat market rather than a missing field. None now means unknown and
        # the UI renders a dash.
        try:
            chg = None if raw is None else float(raw)
        except (TypeError, ValueError):
            chg = None
        markets.append({"name": m.get("name") or m.get("symbol"), "symbol": m.get("symbol"),
                        "price": m.get("price"), "change_pct": chg})
    digest = {"news": news, "markets": markets}

    # ── Summary: composed, never generated ──────────────────────────────────
    def _plural(n, word, suffix="s"):
        return f"{n} {word}{'' if n == 1 else suffix}"

    lead = []
    if offer_total:
        lead.append(_plural(offer_total, "offer") + " to respond to")
    if interview_total:
        lead.append(_plural(interview_total, "interview") + " in play")
    if len(needs_you) and not lead:
        lead.append(_plural(len(needs_you), "item") + " needing a reply")

    rest = []
    if today_events:
        rest.append(_plural(len(today_events), "event") + f" today, first at {today_events[0]['time']}")
    else:
        rest.append("a clear calendar")
    if new_high_fit:
        rest.append(_plural(len(new_high_fit), "new high-fit role"))
    if follow_ups:
        rest.append(_plural(len(follow_ups), "follow-up") + " due")

    def _join(parts):
        if not parts:
            return ""
        if len(parts) == 1:
            return parts[0]
        return ", ".join(parts[:-1]) + f" and {parts[-1]}"

    if lead:
        summary = f"{_join(lead).capitalize()}. Beyond that: {_join(rest)}."
    else:
        summary = f"Nothing needs a decision today. You have {_join(rest)}."

    # ── Focus ───────────────────────────────────────────────────────────────
    if offer_total:
        focus = "You have an offer on the table — review it and respond today."
    elif interview_total:
        focus = "Prep for your interview: research the company and rehearse two stories that match the role."
    elif recruiters:
        focus = "Clear the recruiter replies waiting in your inbox — they move fast."
    elif new_high_fit:
        top = new_high_fit[0]
        focus = f"Apply to {top['company']} — {top['title']} ({top['score']} fit) while it's fresh."
    elif follow_ups:
        focus = f"Send follow-ups on your {len(follow_ups)} stale application(s)."
    else:
        focus = "Keep the pipeline moving — discover and apply to a few new roles."

    return SafeJSONResponse({
        "greeting": greeting, "date_line": date_line, "weather": wx, "prayer": prayer,
        "summary": summary,
        "needs_you": needs_you,
        # Kept so anything still reading the old key doesn't break.
        "action_queue": needs_you,
        "today": {"events": today_events, "count": len(today_events)},
        "jobs": jobs_block, "digest": digest, "focus": focus,
        "connected": {"gmail": gmail_connected, "calendar": cal_connected, "jams": jams["ok"]},
    })


# ── Document Vault ───────────────────────────────────────────────────────────
# Versioned, searchable store for CVs, cover letters, certificates, IDs. Files
# live under data/vault/ with a JSON manifest; text is extracted for search.
# Additive and non-destructive (no deletes — only archive/re-categorise).
from datetime import datetime as _dt

import os as _os
from pathlib import Path as _Path

# Document vault (CVs, transcripts, certificates). Distinct from CODEX_VAULT_DIR
# below, which is the Obsidian notes vault — these two shared the name
# `VAULT_DIR` until now, and because the later binding won at call time the
# uploads landed in the notes vault and were embedded into ChromaDB.
DOC_VAULT_DIR = (UI_DIR.parent / "data" / "vault")
try:
    DOC_VAULT_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass
_VAULT_MANIFEST = DOC_VAULT_DIR / "manifest.json"
VAULT_CATEGORIES = ["CV", "Cover Letter", "Certificate", "Transcript", "Visa / ID",
                    "Portfolio", "Reference", "Other"]


def _vault_load():
    try:
        return _json.loads(_VAULT_MANIFEST.read_text())
    except Exception:
        return []


def _vault_save(items):
    try:
        _VAULT_MANIFEST.write_text(_json.dumps(items, indent=2, ensure_ascii=False))
    except Exception as exc:  # noqa: BLE001
        print(f"Vault: manifest save failed: {exc}")


def _vault_extract_text(path, suffix):
    suffix = (suffix or "").lower()
    try:
        if suffix in (".txt", ".md"):
            return path.read_text(errors="ignore")[:20000]
        if suffix == ".pdf":
            import pdfplumber
            out = []
            with pdfplumber.open(str(path)) as pdf:
                for pg in pdf.pages[:20]:
                    out.append(pg.extract_text() or "")
            return "\n".join(out)[:20000]
        if suffix == ".docx":
            import docx
            d = docx.Document(str(path))
            return "\n".join(p.text for p in d.paragraphs)[:20000]
    except Exception:
        return ""
    return ""


@app.get("/vault")
async def vault_ui():
    """Serve the Document Vault."""
    return FileResponse(UI_DIR / "vault.html")


@app.get("/vault/list")
async def vault_list(q: str = "", category: str = ""):
    items = _vault_load()
    ql = (q or "").strip().lower()
    out = []
    for it in items:
        if it.get("archived"):
            continue
        if category and it.get("category") != category:
            continue
        if ql:
            hay = " ".join(str(it.get(k, "")) for k in ("filename", "category", "label", "text")).lower()
            if ql not in hay:
                continue
        row = {k: v for k, v in it.items() if k != "text"}
        row["excerpt"] = (it.get("text", "") or "").strip().replace("\n", " ")[:220]
        out.append(row)
    out.sort(key=lambda x: x.get("uploaded_at", ""), reverse=True)
    counts = {}
    for it in items:
        if it.get("archived"):
            continue
        counts[it.get("category", "Other")] = counts.get(it.get("category", "Other"), 0) + 1
    return SafeJSONResponse({"items": out, "counts": counts,
                             "total": sum(1 for it in items if not it.get("archived")),
                             "categories": VAULT_CATEGORIES})


@app.post("/vault/upload")
async def vault_upload(file: UploadFile, category: str = Form("Other"), label: str = Form("")):
    import uuid as _uuid
    import os as _os
    content = await file.read()
    if not content:
        return SafeJSONResponse({"ok": False, "error": "empty file"})
    fname = file.filename or "document"
    base, suffix = _os.path.splitext(fname)
    items = _vault_load()
    version = 1 + sum(1 for it in items if it.get("base") == base and not it.get("archived"))
    fid = _uuid.uuid4().hex[:12]
    stored = f"{fid}{suffix.lower()}"
    try:
        (DOC_VAULT_DIR / stored).write_bytes(content)
    except Exception as exc:  # noqa: BLE001
        return SafeJSONResponse({"ok": False, "error": f"could not store file: {exc}"})
    text = _vault_extract_text(DOC_VAULT_DIR / stored, suffix)
    items.append({
        "id": fid, "filename": fname, "base": base, "suffix": suffix.lower(), "stored": stored,
        "category": category if category in VAULT_CATEGORIES else "Other", "label": (label or "").strip()[:120],
        "size": len(content), "version": version, "uploaded_at": _dt.now().isoformat(), "text": text,
    })
    _vault_save(items)
    return SafeJSONResponse({"ok": True, "id": fid, "version": version})


def _vault_stored_path(stored: str):
    """Locate a stored document.

    While DOC_VAULT_DIR and the Obsidian CODEX_VAULT_DIR shared the name
    `VAULT_DIR`, uploads were written into ~/Documents/JarvisVault. Anything
    uploaded in that window still lives there, so fall back to the old
    location rather than 404ing on files the user can see on disk.
    """
    primary = DOC_VAULT_DIR / stored
    if primary.exists():
        return primary
    legacy = _Path(_os.path.expanduser("~/Documents/JarvisVault")) / stored
    if legacy.exists():
        print(f"Vault: serving {stored} from the pre-fix location; "
              f"move it into {DOC_VAULT_DIR} to tidy up")
        return legacy
    return None


@app.get("/vault/file/{fid}")
async def vault_file(fid: str):
    it = next((x for x in _vault_load() if x.get("id") == fid), None)
    if not it:
        return SafeJSONResponse({"error": "not found"}, status_code=404)
    p = _vault_stored_path(it["stored"])
    if p is None:
        return SafeJSONResponse({"error": "missing on disk"}, status_code=404)
    return FileResponse(str(p), filename=it.get("filename"))


@app.post("/vault/update/{fid}")
async def vault_update(fid: str, request: _Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    items = _vault_load()
    for it in items:
        if it.get("id") == fid:
            if body.get("category") in VAULT_CATEGORIES:
                it["category"] = body["category"]
            if "label" in body:
                it["label"] = str(body["label"])[:120]
            if "archived" in body:
                it["archived"] = bool(body["archived"])
            _vault_save(items)
            return SafeJSONResponse({"ok": True})
    return SafeJSONResponse({"ok": False, "error": "not found"}, status_code=404)


# ── Atlas (file brain) + Codex (Obsidian vault) ──────────────────────────────
# Atlas indexes local files (Desktop + Documents) into a searchable brain; Codex
# is a plain-markdown Obsidian vault that agents read from and write back to.
# Both share the Jarvis Brain (ChromaDB + Ollama embeddings). Fully local.
import os as _os
import subprocess as _subprocess
from pathlib import Path as _Path

ATLAS_ROOTS = [
    _os.path.expanduser("~/Desktop"),
    _os.path.expanduser("~/Documents"),
]
CODEX_VAULT_DIR = _Path(_os.path.expanduser("~/Documents/JarvisVault"))
_VAULT_FOLDERS = ["00 Inbox", "10 Notes", "20 Daily", "90 Agents"]


def _ensure_vault():
    """Create the Obsidian vault skeleton if it doesn't exist yet."""
    try:
        for sub in _VAULT_FOLDERS:
            (CODEX_VAULT_DIR / sub).mkdir(parents=True, exist_ok=True)
        readme = CODEX_VAULT_DIR / "README.md"
        if not readme.exists():
            readme.write_text(
                "# Jarvis Vault\n\n"
                "This is your shared brain. Open this folder as a vault in "
                "Obsidian.\n\n"
                "- **00 Inbox** — quick captures\n"
                "- **10 Notes** — your permanent notes\n"
                "- **20 Daily** — daily notes\n"
                "- **90 Agents** — notes Jarvis writes for you\n\n"
                "Everything here is indexed into Jarvis (Codex) so any agent can "
                "search it, and Jarvis writes notes back into *90 Agents*.\n"
            )
        welcome = CODEX_VAULT_DIR / "90 Agents" / "Welcome.md"
        if not welcome.exists():
            welcome.write_text(
                "# Welcome to your Jarvis Vault\n\n"
                "Jarvis created this vault. Notes you write in Obsidian become "
                "searchable across every Jarvis agent, and Jarvis will file "
                "research, job logs and briefings here automatically.\n"
            )
    except Exception as exc:  # noqa: BLE001
        print(f"Codex: vault ensure failed: {exc}")
    return CODEX_VAULT_DIR


def _safe_under(path: str, roots) -> bool:
    try:
        rp = _Path(path).resolve()
    except Exception:
        return False
    for r in list(roots) + [str(CODEX_VAULT_DIR)]:
        try:
            rp.relative_to(_Path(r).resolve())
            return True
        except Exception:
            continue
    return False


@app.get("/atlas")
async def atlas_ui():
    """Serve the Atlas file-brain page."""
    return FileResponse(UI_DIR / "atlas.html")


@app.get("/atlas/status")
async def atlas_status():
    from tools.brain import get_brain
    b = get_brain()
    out = {}
    # recall_activity included so Recall's indexing progress has somewhere to
    # read from — it shares the Brain's status dict with the other sources.
    for src in ("atlas_files", "codex_notes", "recall_activity"):
        st = dict(b.status.get(src, {}))
        st["count"] = b._safe_count(src)
        out[src] = st
    out["roots"] = ATLAS_ROOTS
    out["vault"] = str(CODEX_VAULT_DIR)
    return SafeJSONResponse(out)


@app.post("/atlas/reindex")
async def atlas_reindex(request: _Request):
    """Kick a background (re)index of files + the vault. Non-blocking."""
    from tools.brain import get_brain, TEXT_EXTS
    try:
        body = await request.json()
    except Exception:
        body = {}
    which = body.get("which", "all")
    # Incremental indexing only ever adds and updates, so anything embedded
    # before an exclusion was added stays in the collection forever. After a
    # skip-list change, a rebuild is the only way to drop it.
    rebuild = bool(body.get("rebuild", False))
    b = get_brain()
    _ensure_vault()
    started = []
    if which in ("all", "files"):
        if not b.status.get("atlas_files", {}).get("running"):
            asyncio.create_task(
                b.index_paths("atlas_files", ATLAS_ROOTS, rebuild=rebuild))
            started.append("atlas_files")
    if which in ("all", "notes"):
        if not b.status.get("codex_notes", {}).get("running"):
            asyncio.create_task(
                b.index_paths("codex_notes", [str(CODEX_VAULT_DIR)],
                              rebuild=rebuild))
            started.append("codex_notes")
    if started:
        _GRAPH_CACHE.clear()          # stale the moment the index changes
    return SafeJSONResponse({"ok": True, "started": started, "rebuild": rebuild})


# Built graphs, keyed by (source, threshold, neighbours, max_nodes). Building
# is a few hundred ms on a real corpus and the result only changes when the
# index does, so the cache is cleared by /atlas/reindex rather than timed out.
_GRAPH_CACHE: dict = {}


@app.get("/atlas/graph")
async def atlas_graph(
    source: str = "atlas_files",
    threshold: str = "auto",
    neighbours: int = 6,
    max_nodes: int = 4000,
):
    """Similarity graph over the indexed files.

    Obsidian draws edges from [[wikilinks]]; Atlas has none, so edges are
    inferred from the embeddings already in ChromaDB — see core/graph.py.
    """
    from core.graph import build_graph
    from tools.brain import get_brain

    # "auto" lets core.graph infer the floor from the corpus. A hardcoded
    # number is a guess about the embedding model, and the wrong guess renders
    # an empty map that reads as a broken feature.
    if str(threshold).lower() == "auto":
        threshold = "auto"
    else:
        try:
            threshold = max(0.0, min(0.99, float(threshold)))
        except (TypeError, ValueError):
            threshold = "auto"
    neighbours = max(1, min(20, int(neighbours)))
    max_nodes = max(10, min(20_000, int(max_nodes)))
    key = (source, threshold if isinstance(threshold, str) else round(threshold, 3),
           neighbours, max_nodes)
    if key in _GRAPH_CACHE:
        return SafeJSONResponse({**_GRAPH_CACHE[key], "cached": True})

    b = get_brain()
    try:
        # Off the event loop: this reads every vector out of Chroma and runs a
        # blocked matmul over them, which is hundreds of ms of pure CPU.
        data = await asyncio.to_thread(b.vectors, source)
        graph = await asyncio.to_thread(
            build_graph,
            data["embeddings"], data["metadatas"],
            roots=ATLAS_ROOTS if source == "atlas_files" else [str(CODEX_VAULT_DIR)],
            threshold=threshold, max_neighbours=neighbours, max_nodes=max_nodes,
        )
    except Exception as exc:  # noqa: BLE001
        return SafeJSONResponse(
            {"error": f"{type(exc).__name__}: {exc}", "nodes": [], "edges": []},
            status_code=500)

    payload = {**graph.to_json(), "source": source,
               "params": {"threshold": threshold, "neighbours": neighbours,
                          "max_nodes": max_nodes}}
    _GRAPH_CACHE[key] = payload
    return SafeJSONResponse({**payload, "cached": False})


@app.get("/atlas/search")
async def atlas_search(q: str = "", k: int = 15):
    """Unified search across files (Atlas) and notes (Codex)."""
    from tools.brain import get_brain
    b = get_brain()
    if not q.strip():
        return SafeJSONResponse({"query": q, "results": []})
    files = await b.search("atlas_files", q, k=k)
    notes = await b.search("codex_notes", q, k=max(4, k // 2))
    for n in notes:
        n["is_note"] = True
    merged = notes + files
    # de-dupe by path, keep best score first
    seen, out = set(), []
    for r in sorted(merged, key=lambda x: (x.get("score") is None, -(x.get("score") or 0))):
        if r["path"] in seen:
            continue
        seen.add(r["path"])
        out.append(r)
    return SafeJSONResponse({"query": q, "results": out[:k]})


@app.get("/atlas/open")
async def atlas_open(path: str = ""):
    """Reveal a file in Finder on the local machine (Jarvis runs locally)."""
    if not _safe_under(path, ATLAS_ROOTS):
        return SafeJSONResponse({"ok": False, "error": "path not allowed"}, status_code=400)
    try:
        _subprocess.Popen(["open", "-R", path])
        return SafeJSONResponse({"ok": True})
    except Exception as exc:  # noqa: BLE001
        return SafeJSONResponse({"ok": False, "error": str(exc)}, status_code=500)


@app.get("/codex/list")
async def codex_list():
    """Recent notes in the vault."""
    _ensure_vault()
    rows = []
    try:
        for p in CODEX_VAULT_DIR.rglob("*.md"):
            try:
                stt = p.stat()
                rows.append({
                    "name": p.stem, "rel": str(p.relative_to(CODEX_VAULT_DIR)),
                    "folder": str(p.parent.relative_to(CODEX_VAULT_DIR)),
                    "mtime": stt.st_mtime, "size": stt.st_size,
                })
            except Exception:
                continue
    except Exception:
        pass
    rows.sort(key=lambda x: x["mtime"], reverse=True)
    return SafeJSONResponse({"vault": str(CODEX_VAULT_DIR), "notes": rows[:60],
                             "total": len(rows), "folders": _VAULT_FOLDERS})


@app.post("/codex/note")
async def codex_note(request: _Request):
    """Write a markdown note into the vault (default: 90 Agents) and index it."""
    from datetime import datetime as _dt
    from tools.brain import get_brain
    try:
        body = await request.json()
    except Exception:
        body = {}
    title = (body.get("title") or "").strip() or f"Note {_dt.now():%Y-%m-%d %H%M}"
    folder = body.get("folder") or "90 Agents"
    if folder not in _VAULT_FOLDERS:
        folder = "90 Agents"
    content = body.get("body") or ""
    _ensure_vault()
    safe = "".join(c for c in title if c.isalnum() or c in " -_").strip()[:80] or "note"
    fname = f"{_dt.now():%Y-%m-%d} {safe}.md"
    dest = CODEX_VAULT_DIR / folder / fname
    header = f"# {title}\n\n_Filed by Jarvis · {_dt.now():%Y-%m-%d %H:%M}_\n\n"
    try:
        dest.write_text(header + content)
    except Exception as exc:  # noqa: BLE001
        return SafeJSONResponse({"ok": False, "error": str(exc)}, status_code=500)
    # index the vault in the background so the note becomes searchable
    b = get_brain()
    if not b.status.get("codex_notes", {}).get("running"):
        asyncio.create_task(b.index_paths("codex_notes", [str(CODEX_VAULT_DIR)]))
    return SafeJSONResponse({"ok": True, "path": str(dest),
                             "rel": str(dest.relative_to(CODEX_VAULT_DIR))})


# ── Recall (activity timeline from Chrome history) ───────────────────────────
# Reads Chrome's local history (read-only copy, so it works while Chrome runs)
# and builds a timeline of what you've searched and read. Optionally embeds page
# titles into the Brain (recall_activity) for semantic recall. Fully local.
import os as _os
import sqlite3 as _sqlite3
import shutil as _shutil
import tempfile as _tempfile
import time as _time
from pathlib import Path as _Path
from urllib.parse import urlparse as _urlparse, parse_qs as _parse_qs

_CHROME_BASE = _Path(_os.path.expanduser("~/Library/Application Support/Google/Chrome"))
_SEARCH_HOSTS = {
    "www.google.com": "q", "google.com": "q",
    "www.bing.com": "q", "bing.com": "q",
    "duckduckgo.com": "q", "search.brave.com": "q",
    "www.youtube.com": "search_query", "youtube.com": "search_query",
}
# Chrome epoch: microseconds since 1601-01-01.
_CHROME_EPOCH_OFFSET = 11644473600


def _chrome_to_unix(ts):
    try:
        return ts / 1_000_000 - _CHROME_EPOCH_OFFSET
    except Exception:
        return 0


def _recall_history_dbs():
    dbs = []
    if not _CHROME_BASE.exists():
        return dbs
    for prof in ["Default"] + [f"Profile {i}" for i in range(1, 6)]:
        p = _CHROME_BASE / prof / "History"
        if p.exists():
            dbs.append(p)
    return dbs


def _extract_query(url, host, path):
    """Pull the search term out of a search-engine URL, else ''."""
    key = _SEARCH_HOSTS.get(host)
    if not key:
        return ""
    try:
        if host.endswith("youtube.com") and "/results" not in path:
            return ""
        if "google." in host and not path.startswith("/search"):
            return ""
        qs = _parse_qs(_urlparse(url).query)
        return (qs.get(key, [""])[0] or "").strip()
    except Exception:
        return ""


def _recall_read(days=7, limit=4000):
    """Return recent visits across all Chrome profiles."""
    since = _time.time() - days * 86400
    rows = []
    for db in _recall_history_dbs():
        tmp = None
        try:
            tmp = _Path(_tempfile.gettempdir()) / f"_jarvis_hist_{_os.getpid()}_{db.parent.name}.db"
            _shutil.copy2(db, tmp)
            uri = f"file:{tmp}?mode=ro&immutable=1"
            con = _sqlite3.connect(uri, uri=True)
            cur = con.cursor()
            cur.execute(
                "SELECT urls.url, urls.title, visits.visit_time "
                "FROM visits JOIN urls ON urls.id = visits.url "
                "ORDER BY visits.visit_time DESC LIMIT ?", (limit,))
            for url, title, vt in cur.fetchall():
                ts = _chrome_to_unix(vt)
                if ts < since:
                    continue
                try:
                    pr = _urlparse(url)
                    host = pr.netloc.lower()
                    query = _extract_query(url, host, pr.path)
                except Exception:
                    host, query = "", ""
                rows.append({"url": url, "title": title or "", "ts": ts,
                             "host": host, "query": query})
            con.close()
        except Exception as exc:  # noqa: BLE001
            print(f"Recall: read failed for {db.parent.name}: {exc}")
        finally:
            try:
                if tmp and tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
    rows.sort(key=lambda r: r["ts"], reverse=True)
    return rows


@app.get("/recall")
async def recall_ui():
    return FileResponse(UI_DIR / "recall.html")


@app.get("/recall/data")
async def recall_data(days: int = 7):
    from datetime import datetime as _dt
    from collections import Counter
    days = max(1, min(days, 90))
    rows = _recall_read(days=days)
    # timeline grouped by calendar day
    by_day = {}
    searches, hosts = Counter(), Counter()
    # Two separate things: the full per-day COUNT, and a capped list of items
    # to render. Conflating them made the activity chart useless — every day
    # busier than 40 visits reported exactly 40, so all the bars came out the
    # same height and the chart showed no variation at all.
    day_counts = {}
    for r in rows:
        day = _dt.fromtimestamp(r["ts"]).strftime("%Y-%m-%d")
        day_counts[day] = day_counts.get(day, 0) + 1
        by_day.setdefault(day, [])
        if len(by_day[day]) < 40:
            by_day[day].append({
                "title": r["title"][:120] or r["host"], "url": r["url"],
                "host": r["host"], "query": r["query"],
                "time": _dt.fromtimestamp(r["ts"]).strftime("%H:%M"),
            })
        if r["query"]:
            searches[r["query"]] += 1
        if r["host"]:
            hosts[r["host"].replace("www.", "")] += 1
    timeline = [{"day": d, "items": by_day[d], "count": day_counts.get(d, 0)}
                for d in sorted(by_day, reverse=True)]
    return SafeJSONResponse({
        "days": days, "total_visits": len(rows),
        "timeline": timeline,
        "top_searches": [{"q": q, "n": n} for q, n in searches.most_common(12)],
        "top_sites": [{"host": h, "n": n} for h, n in hosts.most_common(12)],
        "chrome_found": bool(_recall_history_dbs()),
    })


@app.post("/recall/reindex")
async def recall_reindex(request: _Request):
    """Embed recent page titles/searches into the Brain for semantic recall."""
    from tools.brain import get_brain
    async def _job():
        b = get_brain()
        b.status["recall_activity"] = {"running": True, "indexed": 0, "started": _time.time()}
        try:
            rows = _recall_read(days=90, limit=6000)
            col = b.collection("recall_activity")
            seen, ids, docs, metas, embs = set(), [], [], [], []
            for r in rows:
                key = r["url"]
                if key in seen:
                    continue
                seen.add(key)
                text = (r["query"] + " " + r["title"]).strip() or r["host"]
                if not text:
                    continue
                import hashlib as _h
                rid = _h.sha1(key.encode("utf-8", "ignore")).hexdigest()[:16]
                ids.append(rid); docs.append(text)
                metas.append({"path": r["url"], "name": r["title"][:120] or r["host"],
                              "host": r["host"], "query": r["query"], "mtime": r["ts"],
                              "source": "recall_activity"})
                embs.append(await b._embed(text))
                if len(ids) >= 200:
                    col.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
                    b.status["recall_activity"]["indexed"] += len(ids)
                    ids, docs, metas, embs = [], [], [], []
            if ids:
                col.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
                b.status["recall_activity"]["indexed"] += len(ids)
        except Exception as exc:  # noqa: BLE001
            print(f"Recall: reindex failed: {exc}")
        finally:
            b.status["recall_activity"]["running"] = False
            b.status["recall_activity"]["finished"] = _time.time()
            b.status["recall_activity"]["count"] = b._safe_count("recall_activity")
    asyncio.create_task(_job())
    return SafeJSONResponse({"ok": True, "started": "recall_activity"})


@app.get("/recall/search")
async def recall_search(q: str = "", k: int = 20):
    from tools.brain import get_brain
    if not q.strip():
        return SafeJSONResponse({"query": q, "results": []})
    b = get_brain()
    out = []
    if b._safe_count("recall_activity") > 0:
        hits = await b.search("recall_activity", q, k=k)
        out = [{"title": h.get("name"), "url": h.get("path"),
                "snippet": h.get("snippet"), "score": h.get("score")} for h in hits]
    else:
        # fallback: substring over recent live history
        ql = q.lower()
        for r in _recall_read(days=30):
            hay = (r["title"] + " " + r["url"] + " " + r["query"]).lower()
            if ql in hay:
                out.append({"title": r["title"] or r["host"], "url": r["url"],
                            "snippet": r["url"], "score": None})
            if len(out) >= k:
                break
    return SafeJSONResponse({"query": q, "results": out})


# ── Sentinel (local secret & permission scanner) ─────────────────────────────
# Scans the Jarvis project for exposed secrets, risky file permissions and git
# leaks. Read-only and local — it never sends anything anywhere, and it redacts
# every secret it reports.
import os as _os
import re as _re
import stat as _stat
import subprocess as _subprocess
from pathlib import Path as _Path

PROJECT_DIR = UI_DIR.parent

# The scanner itself lives in tools/sentinel.py so the orchestrator can call it
# as a tool and so it can be tested. Keeping a second copy here would let the
# page and the chat answer drift apart, which is the worst possible failure for
# a security scanner — one surface saying "clean" while the other doesn't.
from tools.sentinel import SentinelTool as _SentinelTool

_sentinel_tool = _SentinelTool(PROJECT_DIR)


@app.get("/sentinel")
async def sentinel_ui():
    return FileResponse(UI_DIR / "sentinel.html")


@app.get("/sentinel/data")
async def sentinel_data(history: bool = False, max_commits: int = 400):
    """Working-tree scan by default; `history=true` walks past commits."""
    if history:
        findings, summary = await asyncio.to_thread(
            _sentinel_tool.scan_history, max(10, min(5000, int(max_commits))))
    else:
        findings, summary = await asyncio.to_thread(_sentinel_tool.scan)
    return SafeJSONResponse({
        "root": str(PROJECT_DIR),
        "findings": [f.to_json() for f in findings],
        "summary": summary,
        "mode": "history" if history else "tree",
        "verdict": _sentinel_tool.summarise(findings, summary),
    })


# ── Forge (developer dashboard) ──────────────────────────────────────────────
# Git status, recent commits and TODO/FIXME scan for your projects. Auto-detects
# git repos on the Desktop plus the Jarvis project itself. Read-only & local.
import os as _os
import re as _re
import subprocess as _subprocess
from pathlib import Path as _Path

FORGE_SCAN_ROOTS = [_os.path.expanduser("~/Desktop"), _os.path.expanduser("~/Documents")]

# Same reasoning as Sentinel: the scanner lives in tools/forge.py so it can be
# tested and called as a tool, and so the page and a chat answer can't drift.
from tools.forge import ForgeTool as _ForgeTool

_forge_tool = _ForgeTool(scan_roots=FORGE_SCAN_ROOTS, always_include=PROJECT_DIR)


@app.get("/forge")
async def forge_ui():
    return FileResponse(UI_DIR / "forge.html")


@app.get("/forge/data")
async def forge_data():
    projects = await asyncio.to_thread(_forge_tool.scan)
    return SafeJSONResponse({
        "projects": [p.to_json() for p in projects],
        "count": len(projects),
        "rollup": _forge_tool.rollup(projects),
        "verdict": _forge_tool.summarise(projects),
    })



# ── WebSocket admission control ────────────────────────────────────────────
# The logic lives in core/ws_guard.py so it can be unit-tested; this module
# imports the whole application and cannot be imported by a test.
#
# Two holes closed here:
#  1. BaseHTTPMiddleware only intercepts scope type "http", so /ws never
#     reached _BasicAuthMiddleware — with a password set, the HTTP surface was
#     locked and the WebSocket beside it was wide open.
#  2. Browsers do not apply the same-origin policy to WebSocket, so any page
#     the user visited could open ws://localhost:8000/ws and drive Jarvis.
from core.ws_guard import (
    CLOSE_POLICY_VIOLATION,
    DEFAULT_MAX_FRAME_BYTES,
    authenticate as _ws_authenticate,
    derive_ws_token as _derive_ws_token,
    origin_allowed as _ws_origin_allowed,
    validate_frame as _ws_validate_frame,
)

WS_MAX_FRAME_BYTES = int(os.getenv("JARVIS_WS_MAX_FRAME", str(DEFAULT_MAX_FRAME_BYTES)))


@app.get("/ws-token")
async def ws_token():
    """Issue the WebSocket token.

    An ordinary HTTP route, so it IS behind _BasicAuthMiddleware when a
    password is set — that's what gates issuing it. Browsers can't set headers
    on a WebSocket handshake, which is why the token travels as a query param.
    """
    return {"token": _derive_ws_token(_AUTH_PASS), "required": bool(_AUTH_PASS)}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time chat.
    Streams responses as they come in.
    """
    # Reject BEFORE accepting: an un-accepted socket never enters the app.
    if not _ws_origin_allowed(websocket.headers.get("origin"), _allowed_origins):
        print(f"🚫 WS rejected — disallowed origin "
              f"{websocket.headers.get('origin')!r}")
        await websocket.close(code=CLOSE_POLICY_VIOLATION, reason="origin not allowed")
        return
    if not _ws_authenticate(
        password=_AUTH_PASS,
        user=_AUTH_USER,
        query_token=websocket.query_params.get("token"),
        authorization_header=websocket.headers.get("authorization"),
    ):
        print("🚫 WS rejected — missing or invalid token")
        await websocket.close(code=CLOSE_POLICY_VIOLATION, reason="authentication required")
        return

    await websocket.accept()

    # Push sidebar data in the background — don't block the receive loop
    async def _push_sidebar():
        try:
            sidebar_data = await sidebar()
            await websocket.send_text(json.dumps({
                "type": "sidebar",
                "data": json.loads(sidebar_data.body),
            }))
        except Exception:
            pass

    asyncio.ensure_future(_push_sidebar())

    # Per-session conversation history (survives across turns in this connection)
    conversation_history: list = []

    try:
        while True:
            data = await websocket.receive_text()

            # Validated in core/ws_guard.py. Previously json.loads sat outside
            # every try block and the outer handler caught only
            # (WebSocketDisconnect, RuntimeError), so a single non-JSON frame
            # — or JSON that wasn't an object — killed the session.
            frame = _ws_validate_frame(data, max_bytes=WS_MAX_FRAME_BYTES)
            if frame.error:
                await websocket.send_text(json.dumps({
                    "type": "error", "message": frame.error,
                }))
                continue
            if frame.ignore:
                continue
            message = frame.message

            # Signal typing immediately
            await websocket.send_text(json.dumps({"type": "typing"}))

            # ── Wait on warmup if it's still in flight ─────────────────────
            # If the user fires a query in the first ~30–60s of server life,
            # we'd otherwise have warmup and the user request fighting for a
            # cold model. Block here briefly so warmup wins.
            warmup_task = getattr(app.state, "warmup_task", None)
            if warmup_task is not None and not warmup_task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(warmup_task), timeout=90.0)
                except asyncio.TimeoutError:
                    print("⚠️  Warmup still running after 90s — proceeding anyway")
                except Exception as exc:
                    print(f"⚠️  Warmup task error: {exc}")

            # ── Stream response via handle_stream() ────────────────────────
            try:
                last_response = None
                async def _stream_with_timeout():
                    async for event in jarvis.handle_stream(
                        message,
                        conversation_history=list(conversation_history),
                    ):
                        yield event

                async def _run():
                    nonlocal last_response
                    async for event in _stream_with_timeout():
                        await websocket.send_text(json.dumps(event))
                        if event.get("type") == "response":
                            last_response = event

                try:
                    # 180s gives cold-loads (~60–90s) + tier-2 streaming
                    # (~30–60s) comfortable headroom. Once warmup completes
                    # and keep_alive holds the model resident, real queries
                    # should land in 1–5s anyway.
                    await asyncio.wait_for(_run(), timeout=180.0)
                except asyncio.TimeoutError:
                    await websocket.send_text(json.dumps({
                        "type": "response",
                        "message": "Request timed out after 3 minutes. The model may be overloaded — please try again.",
                        "success": False,
                        "timeout": True,
                    }))
                    continue

                # Store this exchange in conversation history (keep last 10 turns)
                if last_response and last_response.get("message"):
                    conversation_history.append({"role": "user", "content": message})
                    conversation_history.append({"role": "assistant", "content": last_response["message"]})
                    if len(conversation_history) > 20:  # cap at 10 exchanges
                        conversation_history[:] = conversation_history[-20:]

                # Push updated sidebar data after state-changing actions
                keywords = ["schedule", "email", "spotify", "play", "pause", "volume"]
                if any(kw in message.lower() for kw in keywords):
                    try:
                        updated = await sidebar()
                        body = json.loads(updated.body)
                        for key, val in body.items():
                            spayload = {"type": key}
                            if isinstance(val, dict):
                                spayload.update(val)
                            else:
                                spayload["data"] = val
                            await websocket.send_text(json.dumps(spayload))
                    except Exception:
                        pass

            except Exception as e:
                try:
                    await websocket.send_text(json.dumps({
                        "type": "response",
                        "message": f"Error: {str(e)}",
                        "success": False,
                    }))
                except Exception:
                    break  # Connection gone — exit the loop cleanly

    except (WebSocketDisconnect, RuntimeError):
        pass  # Client disconnected — normal, not an error


# ── Run ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    # Pass the app OBJECT, not the "server:app" import string.
    #
    # `python server.py` executes this file as module `__main__`. Handing
    # uvicorn the string made it import `server` as a SECOND, separate module
    # object, re-running the whole file — so every startup banner appeared
    # twice and the process held two JarvisOrchestrators: two ChromaDB
    # clients on one path, two ReminderStores, two SpotifyTools both
    # refreshing tokens, and 414 memories loaded twice. The routes were bound
    # to the second instance; the first sat orphaned holding open handles.
    #
    # The string form only buys `reload=True`, which is off here anyway.
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        # "info" (not "warning") so uvicorn prints its "Uvicorn running on
        # http://0.0.0.0:8000" banner. With "warning" the terminal looks frozen
        # on startup even though the server is up — a confusing false "hang".
        log_level="info",
    )