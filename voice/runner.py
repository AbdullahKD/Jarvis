"""
Voice CLI loop — local stack.

Wires wake-word + STT + brain + TTS into a continuous interaction:

    Hey Jarvis → user speaks → brain(transcript) → Jarvis speaks → loop

Barge-in: while Jarvis is speaking, the wake-word listener stays active on a
shared mic stream, so saying "Hey Jarvis" again cuts him off and we go
straight into STT.

The `brain` is a Callable[[str], str]. If `JarvisOrchestrator` is importable
the runner uses it automatically — otherwise it falls back to an echo stub so
the audio pipeline can be validated independently.
"""

from __future__ import annotations

import logging
import sys
from typing import Callable, Optional

from .audio import MicStream
from .config import VoiceConfig, VoiceConfigError, load_config
from .stt import StreamingSTT
from .tts import StreamingTTS
from .wake_word import WakeWordListener

log = logging.getLogger("jarvis.voice.runner")

Brain = Callable[[str], str]


# ── Brain wiring ───────────────────────────────────────────────────────────


def _echo_brain(transcript: str) -> str:
    """Stub brain. Replaced automatically if the orchestrator is importable."""
    return f"You said: {transcript}. The orchestrator isn't wired up yet, sir."


def _try_orchestrator_brain() -> Optional[Brain]:
    """
    Attempt to instantiate JarvisOrchestrator from the project root and adapt
    its async .handle() into a sync Callable. Returns None on any failure so
    we cleanly fall back to the echo brain.
    """
    try:
        import asyncio

        from orchestrator import JarvisOrchestrator  # type: ignore
    except Exception as exc:  # noqa: BLE001
        log.info("orchestrator unavailable, using echo brain (%s)", exc)
        return None

    try:
        jarvis = JarvisOrchestrator()
    except Exception as exc:  # noqa: BLE001
        log.warning("orchestrator init failed: %s — falling back to echo", exc)
        return None

    def brain(transcript: str) -> str:
        coro = jarvis.handle(transcript)
        response = asyncio.run(coro)
        return getattr(response, "message", str(response))

    log.info("orchestrator brain wired up")
    return brain


# ── Runner ─────────────────────────────────────────────────────────────────


class VoiceRunner:
    """Loop wake → listen → think → speak with optional barge-in."""

    def __init__(
        self,
        brain: Optional[Brain] = None,
        *,
        config: Optional[VoiceConfig] = None,
        use_wake_word: Optional[bool] = None,
    ):
        self._cfg = config or load_config()
        self._brain = brain or _try_orchestrator_brain() or _echo_brain
        self._use_wake = (
            self._cfg.wake_word_enabled if use_wake_word is None else use_wake_word
        )

        # Building these eagerly so config errors (missing piper, etc.) surface
        # immediately rather than on the first turn.
        self._tts = StreamingTTS(self._cfg)
        self._stt = StreamingSTT(self._cfg)
        self._wake = WakeWordListener(self._cfg) if self._use_wake else None

    # ── Public API ─────────────────────────────────────────────────────────

    def run(self) -> None:
        self._banner()
        self._stt.warmup()
        print("[voice] STT model loaded.")
        try:
            while True:
                self._one_turn()
        except KeyboardInterrupt:
            print("\n[voice] Bye.")
        finally:
            self._tts.close()

    # ── Internals ──────────────────────────────────────────────────────────

    def _one_turn(self) -> None:
        if self._use_wake and self._wake is not None:
            print('[voice] Listening for "Hey Jarvis"... (Ctrl-C to quit)')
            if not self._wake.wait_for_wake_word():
                return
            print("[voice] Wake word detected.")
        else:
            try:
                input("[voice] Press Enter to talk (Ctrl-C to quit) ... ")
            except EOFError:
                raise KeyboardInterrupt

        # One shared mic stream for STT (and barge-in during TTS later)
        with MicStream(self._cfg).open() as mic:
            print("[voice] Listening...")
            transcript = self._stt.listen_once(
                mic=mic,
                on_partial=lambda t: print(f"  …{t}", end="\r", flush=True),
            )
            print(" " * 80, end="\r")

            if not transcript:
                print("[voice] (heard nothing)")
                return

            print(f"[you]    {transcript}")
            try:
                reply = self._brain(transcript) or ""
            except Exception as exc:  # noqa: BLE001
                log.exception("brain failed")
                reply = f"Something went wrong, sir. {exc}"

            if not reply.strip():
                reply = "I don't have a response for that."

            print(f"[jarvis] {reply}")
            self._speak_with_barge_in(reply, mic)

    def _speak_with_barge_in(self, reply: str, mic: MicStream) -> None:
        """Speak the reply; if barge-in is enabled, let the wake word interrupt."""
        if not self._cfg.barge_in_enabled or self._wake is None:
            self._tts.speak(reply)
            return

        interrupted = {"hit": False}

        def _on_barge_in():
            interrupted["hit"] = True
            print("[voice] Barge-in — stopping playback.")
            self._tts.stop()

        self._tts.speak_async(reply)
        self._wake.start_listening(mic, on_detected=_on_barge_in)
        self._tts.wait()
        self._wake.stop_listening()

        if interrupted["hit"]:
            # User interrupted: drop straight into STT for their follow-up.
            print("[voice] Listening (post-barge-in)...")
            transcript = self._stt.listen_once(mic=mic)
            print(" " * 80, end="\r")
            if transcript:
                print(f"[you]    {transcript}")
                try:
                    reply = self._brain(transcript) or ""
                except Exception as exc:  # noqa: BLE001
                    log.exception("brain failed (after barge-in)")
                    reply = f"Something went wrong, sir. {exc}"
                if reply.strip():
                    print(f"[jarvis] {reply}")
                    self._tts.speak(reply)

    def _banner(self) -> None:
        cfg = self._cfg
        print("[voice] ── Jarvis voice stack ───────────────────────────")
        print(f"[voice] Wake word    : {cfg.wake_word_model if self._use_wake else 'disabled (push-to-talk)'}")
        print(f"[voice] STT          : faster-whisper {cfg.whisper_model} ({cfg.whisper_compute_type}, {cfg.whisper_device})")
        print(f"[voice] TTS          : ElevenLabs · {cfg.elevenlabs_model} · voice {cfg.elevenlabs_voice_id}")
        print(f"[voice] Barge-in     : {'on' if cfg.barge_in_enabled and self._use_wake else 'off'}")
        print(f"[voice] Brain        : {self._brain.__name__}")


# ── CLI ────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m voice",
        description="Jarvis local voice agent (Whisper + Piper + openWakeWord).",
    )
    parser.add_argument(
        "--no-wake",
        action="store_true",
        help="Skip wake-word — press Enter to talk instead.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show DEBUG logs.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    try:
        runner = VoiceRunner(use_wake_word=False if args.no_wake else None)
    except VoiceConfigError as exc:
        print(f"[voice] Config error:\n{exc}", file=sys.stderr)
        return 2

    runner.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
