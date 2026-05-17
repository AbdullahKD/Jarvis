"""
Streaming, interruptible Text-to-Speech via ElevenLabs.

Uses the official `elevenlabs` Python SDK in streaming mode against the
`eleven_flash_v2_5` model (~75 ms time-to-first-audio). Audio arrives as raw
PCM (22050 Hz mono int16) and is played through sounddevice as it streams, so
the first word is audible long before the last word is synthesised.

Public API matches every prior TTS backend so the runner doesn't change:

    tts = StreamingTTS(config)
    tts.speak("Good morning, sir.")     # blocking
    tts.speak_async("Long answer...")   # non-blocking
    tts.stop()                          # interrupt mid-utterance
"""

from __future__ import annotations

import logging
import threading
from typing import Iterator, Optional

from .audio import play_pcm_stream
from .config import VoiceConfig, VoiceConfigError

log = logging.getLogger("jarvis.voice.tts")

# `pcm_22050` is the lightest streamable format ElevenLabs offers; matches
# sounddevice's default int16 output cleanly.
ELEVENLABS_SAMPLE_RATE = 22050
ELEVENLABS_OUTPUT_FORMAT = "pcm_22050"


class TTSError(RuntimeError):
    """Raised when ElevenLabs synthesis fails for a non-recoverable reason."""


class StreamingTTS:
    """Interruptible ElevenLabs-backed streaming TTS."""

    def __init__(self, config: VoiceConfig):
        if not config.elevenlabs_api_key:
            raise VoiceConfigError(
                "ELEVENLABS_API_KEY is missing.\n"
                "  → Sign up at https://elevenlabs.io (free 10k chars/month)\n"
                "  → Paste your key into `.env` and try again."
            )
        try:
            from elevenlabs.client import ElevenLabs
        except ImportError as exc:
            raise VoiceConfigError(
                "elevenlabs is not installed. "
                "Run: pip install -r voice_requirements.txt"
            ) from exc

        self._cfg = config
        self._client = ElevenLabs(api_key=config.elevenlabs_api_key)

        self._stop = threading.Event()
        self._done = threading.Event()
        self._done.set()
        self._thread: Optional[threading.Thread] = None

    # ── Public API ─────────────────────────────────────────────────────────

    def speak(self, text: str) -> None:
        """Synthesise + play, block until done or interrupted."""
        self.speak_async(text)
        self.wait()

    def speak_async(self, text: str) -> None:
        """Kick off synthesis on a background thread and return."""
        if not text or not text.strip():
            return
        self.stop()
        self._stop.clear()
        self._done.clear()
        self._thread = threading.Thread(
            target=self._run, args=(text,), daemon=True
        )
        self._thread.start()

    def wait(self, timeout: Optional[float] = None) -> bool:
        return self._done.wait(timeout=timeout)

    def stop(self) -> None:
        """Interrupt the current utterance. Safe when idle."""
        if self._done.is_set():
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._done.set()

    def close(self) -> None:
        self.stop()

    @property
    def is_speaking(self) -> bool:
        return not self._done.is_set()

    # ── Internals ──────────────────────────────────────────────────────────

    def _run(self, text: str) -> None:
        try:
            pcm = self._elevenlabs_stream(text)
            play_pcm_stream(
                pcm,
                sample_rate=ELEVENLABS_SAMPLE_RATE,
                stop_event=self._stop,
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("tts run failed")
            raise TTSError(str(exc)) from exc
        finally:
            self._done.set()

    def _elevenlabs_stream(self, text: str) -> Iterator[bytes]:
        """Yield raw PCM int16 bytes from the ElevenLabs streaming endpoint."""
        log.debug(
            "tts.elevenlabs: voice=%s model=%s len=%d",
            self._cfg.elevenlabs_voice_id,
            self._cfg.elevenlabs_model,
            len(text),
        )
        stream = self._client.text_to_speech.stream(
            voice_id=self._cfg.elevenlabs_voice_id,
            model_id=self._cfg.elevenlabs_model,
            text=text,
            output_format=ELEVENLABS_OUTPUT_FORMAT,
        )
        for chunk in stream:
            if self._stop.is_set():
                break
            if chunk:
                yield chunk
