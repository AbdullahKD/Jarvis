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
from pathlib import Path

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

from orchestrator import JarvisOrchestrator
from agents.finex_agent import FinExAgent
from tools.reminders import ReminderScheduler

# ── App setup ──────────────────────────────────────────────────────────────
app = FastAPI(title="J.A.R.V.I.S", version="1.0.0")


@app.on_event("startup")
async def startup():
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
    asyncio.ensure_future(_voice_preflight())


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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Optional HTTP Basic Auth (cloud deployment) ────────────────────────────
# If JARVIS_AUTH_PASSWORD is set, every request must include a matching
# Basic-Auth header. The browser handles this transparently with a login
# popup, and curl users can pass `-u admin:<password>`.
#
# Set neither var locally to keep dev unauthenticated.
import base64 as _b64
from fastapi import Request as _Request, Response as _Response
from starlette.middleware.base import BaseHTTPMiddleware as _BaseHTTPMiddleware

_AUTH_USER = os.getenv("JARVIS_AUTH_USER", "admin")
_AUTH_PASS = os.getenv("JARVIS_AUTH_PASSWORD", "")
# Public paths that bypass auth — health checks must be reachable for Fly.
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
                if user == _AUTH_USER and pw == _AUTH_PASS:
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
    """Unauthenticated health endpoint for Fly.io."""
    return {"ok": True}


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
    suspend heavy refresh loops (sidebar, live-tick) during a voice turn."""
    return {"active": _voice_is_active()}

# Shared orchestrator instance
jarvis = JarvisOrchestrator()
finex  = FinExAgent()

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


async def _ensure_voice_session():
    """Lazy-build the VoiceSession with a streaming brain bound to FastAPI's event loop."""
    global _voice_session, _voice_stylizer
    if _voice_session is not None:
        return _voice_session

    main_loop = asyncio.get_running_loop()

    # Persona stylizing is disabled — replies go orchestrator → TTS directly.
    _voice_stylizer = None

    def _voice_brain_streaming(transcript: str, *, tts, on_state, cancel_event) -> str:
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

        sentence_split = _re.compile(r'(?<=[.!?])\s+')
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
                        # Split on sentence boundaries — push every complete
                        # sentence to the queue, keep the trailing partial.
                        parts = sentence_split.split(buf)
                        if len(parts) > 1:
                            for s in parts[:-1]:
                                s = s.strip()
                                if s:
                                    sentence_q.put(s)
                            buf = parts[-1]
                    elif etype == "response":
                        # Final canonical text from the orchestrator. Overrides
                        # the streamed accumulator (the orchestrator may have
                        # post-processed via _enforce_single_paragraph).
                        msg = event.get("message")
                        if msg:
                            full_text = msg
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
     s_pl, s_ucl, s_nba,
     s_cric_test, s_cric_odi, s_cric_t20, s_cric_ipl, s_cric_psl, s_cric_bbl,
     s_rm, prayer, gmail) = await asyncio.gather(
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
        # Cricket: pulled per-format and merged below into one consolidated
        # `sports_cricket` payload so the UI can render a single section.
        safe(jarvis.sports.get_scores("cricket_test", limit=4)),
        safe(jarvis.sports.get_scores("cricket_odi",  limit=4)),
        safe(jarvis.sports.get_scores("cricket_t20",  limit=4)),
        safe(jarvis.sports.get_scores("cricket_ipl",  limit=6)),
        safe(jarvis.sports.get_scores("cricket_psl",  limit=6)),
        safe(jarvis.sports.get_scores("cricket_bbl",  limit=4)),
        safe(jarvis.sports.search_team("Real Madrid", "la_liga")),
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
    # ── Consolidated cricket payload ────────────────────────────────────────
    # All formats and league competitions go into ONE sports_cricket section.
    # Each game gets a `format` tag (Test/ODI/T20I/IPL/PSL/BBL) the UI can
    # render as a small badge. We dedupe by date+teams in case ESPN returns
    # the same fixture under multiple sub-leagues.
    def _ser_cricket_games(raw_list, fmt_label):
        out = []
        for g in raw_list:
            # _ser_games loses the cricket-specific fields — copy them
            # forward by hand. Same shape as football games plus
            # home_innings / away_innings / note / format.
            out.append({
                **{
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
                },
                "home_innings": g.get("home_innings", []),
                "away_innings": g.get("away_innings", []),
                "note":         g.get("note", ""),
                "format":       fmt_label,
            })
        return out

    _cricket_sources = [
        (s_cric_test, "Test"),
        (s_cric_odi,  "ODI"),
        (s_cric_t20,  "T20I"),
        (s_cric_ipl,  "IPL"),
        (s_cric_psl,  "PSL"),
        (s_cric_bbl,  "BBL"),
    ]
    _all_cricket_games = []
    _seen_fixtures = set()  # (date_iso[:10], home, away) → dedupe
    for src, fmt in _cricket_sources:
        if not src or not src.get("success"):
            continue
        for g in _ser_cricket_games(src.get("games", []), fmt):
            key = (g.get("date_iso", "")[:10], g.get("home_team", ""), g.get("away_team", ""))
            if key in _seen_fixtures:
                continue
            _seen_fixtures.add(key)
            _all_cricket_games.append(g)

    if _all_cricket_games:
        result["sports_cricket"] = {
            "success":    True,
            "league":     "Cricket",
            "league_key": "cricket",
            "games":      _all_cricket_games,
        }
    if s_rm.get("success"):
        result["sports_rm"] = {"success":True,"league":s_rm.get("league","La Liga"),"league_key":"la_liga","team":"Real Madrid","games":_ser_games(s_rm.get("games",[]))}
    if prayer.get("success"): result["prayer"] = prayer
    if gmail.get("success"):
        result["emails"] = gmail.get("emails", [])

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

    (markets, s_pl, s_ucl, s_nba,
     s_cric_test, s_cric_odi, s_cric_t20,
     s_cric_ipl, s_cric_psl, s_cric_bbl,
     s_rm) = await asyncio.gather(
        safe(jarvis.markets.get_all()),
        safe(jarvis.sports.get_scores("premier_league", limit=8)),
        safe(jarvis.sports.get_scores("champions_league", limit=6)),
        safe(jarvis.sports.get_scores("nba", limit=6)),
        safe(jarvis.sports.get_scores("cricket_test", limit=4)),
        safe(jarvis.sports.get_scores("cricket_odi",  limit=4)),
        safe(jarvis.sports.get_scores("cricket_t20",  limit=4)),
        safe(jarvis.sports.get_scores("cricket_ipl",  limit=6)),
        safe(jarvis.sports.get_scores("cricket_psl",  limit=6)),
        safe(jarvis.sports.get_scores("cricket_bbl",  limit=4)),
        safe(jarvis.sports.search_team("Real Madrid", "la_liga")),
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

    # Consolidated cricket — same dedupe logic as /sidebar
    def _ser_cricket(raw_list, fmt_label):
        out = []
        for g in raw_list:
            out.append({
                **{k: g.get(k, "") for k in (
                    "home_team", "away_team", "home_score", "away_score",
                    "status", "date_str", "date_iso", "clock",
                    "home_color", "away_color", "home_logo", "away_logo",
                )},
                "home_innings": g.get("home_innings", []),
                "away_innings": g.get("away_innings", []),
                "note":         g.get("note", ""),
                "format":       fmt_label,
            })
        return out

    _cricket_sources = [
        (s_cric_test, "Test"),
        (s_cric_odi,  "ODI"),
        (s_cric_t20,  "T20I"),
        (s_cric_ipl,  "IPL"),
        (s_cric_psl,  "PSL"),
        (s_cric_bbl,  "BBL"),
    ]
    _games = []
    _seen = set()
    for src, fmt in _cricket_sources:
        if not src or not src.get("success"):
            continue
        for g in _ser_cricket(src.get("games", []), fmt):
            key = (g.get("date_iso", "")[:10], g.get("home_team", ""), g.get("away_team", ""))
            if key in _seen:
                continue
            _seen.add(key)
            _games.append(g)
    if _games:
        out["sports_cricket"] = {
            "success": True, "league": "Cricket", "league_key": "cricket", "games": _games,
        }

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
    wifi_iface = iface_raw.strip() or "en0"
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
        jarvis.calendar = CalendarAgent()
        jarvis.gmail = GmailAgent()
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


@app.post("/finex/chat")
async def finex_chat(req: FinExChatRequest):
    """Financial statement Q&A — powered by the FinEx engine (6 reasoning levels)."""
    result = await finex.chat(req.question, req.company, req.history)
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


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time chat.
    Streams responses as they come in.
    """
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
            payload = json.loads(data)
            message = payload.get("message", "")

            if not message:
                continue

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
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="warning",
    )