"""
Single-turn voice session for the web UI.

`VoiceRunner` loops forever and is designed for the terminal. The web UI needs
a different shape: the user clicks a button, we do exactly one turn
(listen → think → speak), and the frontend polls a status endpoint to animate
its orb. That's `VoiceSession`.

Lifecycle:

    session = get_session()
    session.start()              # kicks off a turn in a background thread
    session.status()             # poll: {state, transcript, reply, error}
    session.cancel()             # interrupt mid-turn (esp. mid-TTS)

States: idle → listening → thinking → speaking → idle (or error).
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

from .config import load_config
from .runner import _echo_brain, _try_orchestrator_brain
from .stt import StreamingSTT
from .tts import StreamingTTS

log = logging.getLogger("jarvis.voice.web")

Brain = Callable[[str], str]
State = str  # "idle" | "listening" | "thinking" | "speaking" | "error"


class VoiceSession:
    """Thread-safe single-turn voice session."""

    def __init__(self, brain: Optional[Brain] = None):
        self._cfg = load_config()
        self._stt = StreamingSTT(self._cfg)
        self._tts = StreamingTTS(self._cfg)
        self._brain: Brain = brain or _try_orchestrator_brain() or _echo_brain

        self._lock = threading.Lock()
        self._state: State = "idle"
        self._transcript = ""
        self._reply = ""
        self._error = ""
        self._thread: Optional[threading.Thread] = None
        self._cancel = threading.Event()

        # Warm whisper on construction so the first turn isn't slow
        try:
            self._stt.warmup()
        except Exception as exc:  # noqa: BLE001
            log.warning("STT warmup failed: %s", exc)

    # ── Public API ─────────────────────────────────────────────────────────

    def status(self) -> dict:
        with self._lock:
            return {
                "state": self._state,
                "transcript": self._transcript,
                "reply": self._reply,
                "error": self._error,
                "active": self._thread is not None and self._thread.is_alive(),
            }

    def start(self) -> bool:
        """Begin a single voice turn. Returns False if one is already active."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._cancel.clear()
            self._state = "listening"
            self._transcript = ""
            self._reply = ""
            self._error = ""
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return True

    def cancel(self) -> None:
        """Interrupt the current turn. Safe when idle."""
        self._cancel.set()
        # Calling stop() on TTS lets us cut off speech mid-utterance.
        try:
            self._tts.stop()
        except Exception:  # noqa: BLE001
            pass

    # ── Internals ──────────────────────────────────────────────────────────

    def _set(self, state: State, **kwargs) -> None:
        with self._lock:
            self._state = state
            for k, v in kwargs.items():
                if k == "transcript":
                    self._transcript = v
                elif k == "reply":
                    self._reply = v
                elif k == "error":
                    self._error = v

    def _run(self) -> None:
        try:
            # ── Phase 1: STT ───────────────────────────────────────────────
            transcript = self._stt.listen_once()
            if self._cancel.is_set():
                self._set("idle")
                return
            if not transcript:
                self._set("idle", error="(heard nothing)")
                return
            self._set("thinking", transcript=transcript)

            # ── Phase 2: Brain ─────────────────────────────────────────────
            try:
                reply = self._brain(transcript) or ""
            except Exception as exc:  # noqa: BLE001
                log.exception("brain failed")
                reply = f"Something went wrong, sir. {exc}"
            if self._cancel.is_set():
                self._set("idle")
                return
            if not reply.strip():
                reply = "I don't have a response for that."
            self._set("speaking", reply=reply)

            # ── Phase 3: TTS ───────────────────────────────────────────────
            self._tts.speak(reply)
            self._set("idle")

        except Exception as exc:  # noqa: BLE001
            log.exception("voice turn failed")
            self._set("error", error=str(exc))


# ── Module-level lazy singleton ────────────────────────────────────────────

_session: Optional[VoiceSession] = None
_session_lock = threading.Lock()


def get_session(brain: Optional[Brain] = None) -> VoiceSession:
    """Get (or lazily construct) the shared voice session."""
    global _session
    with _session_lock:
        if _session is None:
            _session = VoiceSession(brain=brain)
        return _session
