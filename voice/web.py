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

Brain modes
-----------
The brain callable can be one of two shapes:

  * Classic ``brain(transcript) -> reply_text`` — VoiceSession blocks on it,
    then runs TTS on the full reply.
  * Streaming ``brain(transcript, *, tts, on_state) -> reply_text`` — the
    brain is responsible for streaming partial replies into ``tts`` itself
    (typically via ``tts.speak_sentence_stream``) and calling
    ``on_state("speaking")`` once the first audio plays. VoiceSession won't
    fire a second TTS pass when this mode is used.

Pass ``brain_streams_tts=True`` to opt into the streaming shape.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

from .config import load_config
from .runner import _echo_brain, _try_orchestrator_brain
from .stt import StreamingSTT, StreamingSTTError
from .tts import StreamingTTS

log = logging.getLogger("jarvis.voice.web")

Brain = Callable[..., str]
State = str  # "idle" | "listening" | "thinking" | "speaking" | "error"


class VoiceSession:
    """Thread-safe single-turn voice session."""

    def __init__(
        self,
        brain: Optional[Brain] = None,
        *,
        brain_streams_tts: bool = False,
    ):
        self._cfg = load_config()
        self._stt = StreamingSTT(self._cfg)
        self._tts = StreamingTTS(self._cfg)
        self._brain: Brain = brain or _try_orchestrator_brain() or _echo_brain
        self._brain_streams_tts = brain_streams_tts

        self._lock = threading.Lock()
        self._state: State = "idle"
        self._transcript = ""
        self._reply = ""
        self._error = ""
        self._thread: Optional[threading.Thread] = None
        self._cancel = threading.Event()
        # Live mic telemetry — updated every frame from the STT loop so the
        # web UI can show "I hear you" feedback on the listening orb.
        self._level: dict = {
            "peak": 0.0,
            "vad_prob": 0.0,
            "max_prob": 0.0,
            "has_speech": False,
            "frames": 0,
            "elapsed_ms": 0,
        }

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
                "level": dict(self._level),
            }

    def start(self) -> bool:
        """Begin a single voice turn. Returns False if one is already active."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._cancel.clear()
            # Reset TTS stop flag so a new turn isn't pre-cancelled by an old
            # session.cancel() that fired on the previous turn.
            self._tts._stop.clear()
            self._state = "listening"
            self._transcript = ""
            self._reply = ""
            self._error = ""
            self._level = {
                "peak": 0.0,
                "vad_prob": 0.0,
                "max_prob": 0.0,
                "has_speech": False,
                "frames": 0,
                "elapsed_ms": 0,
            }
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

    def _on_level(self, snapshot: dict) -> None:
        """STT calls this every frame with live audio level + VAD telemetry."""
        with self._lock:
            self._level = snapshot

    def _on_partial_reply(self, text: str) -> None:
        """
        Streaming brain calls this as the LLM emits tokens, so the UI can show
        a live transcript of what Jarvis is saying while it's still speaking.
        Updates only the reply text; leaves the current state untouched.
        """
        with self._lock:
            self._reply = text

    def _run(self) -> None:
        import time
        t_start = time.perf_counter()
        try:
            # ── Phase 1: STT ───────────────────────────────────────────────
            print("🎙️  [voice] Phase 1/3: listening (Whisper)...")
            t0 = time.perf_counter()
            try:
                transcript = self._stt.listen_once(on_status=self._on_level)
            except StreamingSTTError as exc:
                print(f"🎙️  [voice] STT environmental failure: {exc}")
                self._set("error", error=str(exc))
                return
            stt_ms = (time.perf_counter() - t0) * 1000
            if self._cancel.is_set():
                print("🎙️  [voice] Cancelled during STT")
                self._set("idle")
                return
            if not transcript:
                lvl = self._level
                if lvl.get("max_prob", 0.0) < 0.05 and lvl.get("peak", 0.0) < 0.02:
                    detail = (
                        "(heard nothing — mic appears silent. "
                        "Check System Settings → Privacy & Security → Microphone "
                        "and confirm your terminal app is allowed.)"
                    )
                else:
                    detail = (
                        "(heard some audio but couldn't transcribe it — "
                        "try speaking a little louder or closer to the mic.)"
                    )
                print(f"🎙️  [voice] STT heard nothing  ({stt_ms:.0f} ms)  level={lvl}")
                self._set("idle", error=detail)
                return
            print(f"🎙️  [voice] STT transcript ({stt_ms:.0f} ms): {transcript!r}")
            self._set("thinking", transcript=transcript)

            # ── Phase 2 + 3: Brain (+ TTS if streaming) ────────────────────
            if self._brain_streams_tts:
                # Brain owns both Brain phase and TTS phase. It will call
                # _flip_to_speaking once the first sentence begins playback.
                print("🧠  [voice] Phase 2/3: streaming brain → ElevenLabs...")

                def _flip_to_speaking() -> None:
                    # Keep whatever partial text has already streamed in; don't
                    # clobber it with a placeholder.
                    with self._lock:
                        self._state = "speaking"

                t0 = time.perf_counter()
                try:
                    reply = self._brain(
                        transcript,
                        tts=self._tts,
                        on_state=_flip_to_speaking,
                        cancel_event=self._cancel,
                        on_partial=self._on_partial_reply,
                    ) or ""
                except Exception as exc:  # noqa: BLE001
                    log.exception("streaming brain failed")
                    print(f"🧠  [voice] Streaming brain crashed: {exc!r}")
                    reply = f"Something went wrong, sir. {exc}"
                    # Try to speak the error directly so the user isn't left
                    # staring at the orb.
                    try:
                        self._set("speaking", reply=reply)
                        self._tts.speak(reply)
                    except Exception:
                        pass
                turn_ms = (time.perf_counter() - t0) * 1000
                if not reply.strip():
                    reply = "I don't have a response for that, sir."
                self._set("idle", reply=reply)
                total_ms = (time.perf_counter() - t_start) * 1000
                print(f"✅  [voice] Streaming turn complete in {total_ms:.0f} ms (stt={stt_ms:.0f}, brain+tts={turn_ms:.0f})")
                return

            # Classic mode: brain blocks, then TTS plays full reply.
            print("🧠  [voice] Phase 2/3: thinking (orchestrator + persona)...")
            t0 = time.perf_counter()
            try:
                reply = self._brain(transcript) or ""
            except Exception as exc:  # noqa: BLE001
                log.exception("brain failed")
                print(f"🧠  [voice] Brain crashed: {exc!r}")
                reply = f"Something went wrong, sir. {exc}"
            brain_ms = (time.perf_counter() - t0) * 1000
            if self._cancel.is_set():
                print("🧠  [voice] Cancelled during brain")
                self._set("idle")
                return
            if not reply.strip():
                reply = "I don't have a response for that, sir."
            print(f"🧠  [voice] Brain reply ({brain_ms:.0f} ms): {reply!r}")
            self._set("speaking", reply=reply)

            print("🔊  [voice] Phase 3/3: speaking (ElevenLabs streaming)...")
            t0 = time.perf_counter()
            self._tts.speak(reply)
            tts_ms = (time.perf_counter() - t0) * 1000
            print(f"🔊  [voice] TTS done ({tts_ms:.0f} ms)")
            total_ms = (time.perf_counter() - t_start) * 1000
            print(f"✅  [voice] Turn complete in {total_ms:.0f} ms (stt={stt_ms:.0f}, brain={brain_ms:.0f}, tts={tts_ms:.0f})")
            self._set("idle")

        except Exception as exc:  # noqa: BLE001
            log.exception("voice turn failed")
            print(f"❌  [voice] Turn failed: {type(exc).__name__}: {exc}")
            self._set("error", error=f"{type(exc).__name__}: {exc}")


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
