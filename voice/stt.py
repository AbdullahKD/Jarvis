"""
Streaming Speech-to-Text using faster-whisper.

faster-whisper is a CTranslate2 reimplementation of OpenAI Whisper — about 4x
faster than the reference implementation and runs in pure CPU int8 mode on
Apple Silicon, no Metal kernels needed.

The flow for a single utterance:

    1. Open the mic stream.
    2. Feed frames to the Silero VAD endpointer.
    3. When VAD decides the user has stopped, concatenate the buffered audio.
    4. Run faster-whisper transcription on the entire utterance.
    5. Return the final transcript.

We could chunk-and-decode incrementally for lower latency, but the simpler
"record then transcribe" path is plenty fast on M2: <500 ms transcription for
a 5-second utterance with small.en/int8.
"""

from __future__ import annotations

import logging
import queue
import time
from threading import Lock
from typing import Callable, Optional

import numpy as np

from .audio import MicStream
from .config import VoiceConfig
from .vad import SileroEndpointer

log = logging.getLogger("jarvis.voice.stt")


class StreamingSTT:
    """One-utterance-at-a-time recogniser bound to the configured Whisper model."""

    def __init__(self, config: VoiceConfig):
        self._cfg = config
        self._lock = Lock()
        self._model = None  # lazy: only load when first listen_once() is called

    # ── Public API ─────────────────────────────────────────────────────────

    def listen_once(
        self,
        *,
        mic: Optional[MicStream] = None,
        on_partial: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Capture one utterance, transcribe it, return the text.

        If `mic` is None, opens its own MicStream. Pass a shared MicStream when
        composing with the wake-word listener to avoid device contention.
        """
        own_mic = mic is None
        if own_mic:
            mic = MicStream(self._cfg)
            mic.start()

        try:
            audio = self._record_utterance(mic)
        finally:
            if own_mic:
                mic.stop()

        if audio is None or len(audio) == 0:
            log.info("stt: empty utterance")
            return ""

        # Whisper expects float32 in [-1, 1]
        audio_f32 = audio.astype(np.float32) / 32768.0

        model = self._ensure_model()
        t0 = time.perf_counter()
        segments, info = model.transcribe(
            audio_f32,
            beam_size=1,         # greedy decoding for speed
            language="en",
            condition_on_previous_text=False,
            vad_filter=False,    # we already endpointed with Silero
        )
        # Collect lazily-yielded segments
        text_parts: list[str] = []
        for seg in segments:
            t = (seg.text or "").strip()
            if t:
                text_parts.append(t)
                if on_partial:
                    on_partial(" ".join(text_parts))
        elapsed = time.perf_counter() - t0
        audio_seconds = len(audio) / self._cfg.sample_rate
        log.info(
            "stt: %.2fs audio → %.2fs decode (%.1fx realtime)",
            audio_seconds,
            elapsed,
            audio_seconds / elapsed if elapsed > 0 else 0,
        )

        return " ".join(text_parts).strip()

    # ── Internals ──────────────────────────────────────────────────────────

    def _ensure_model(self):
        with self._lock:
            if self._model is None:
                try:
                    from faster_whisper import WhisperModel
                except ImportError as exc:
                    raise RuntimeError(
                        "faster-whisper is not installed. "
                        "Run: pip install -r voice_requirements.txt"
                    ) from exc

                log.info(
                    "stt: loading faster-whisper %s (%s, %s)...",
                    self._cfg.whisper_model,
                    self._cfg.whisper_device,
                    self._cfg.whisper_compute_type,
                )
                self._model = WhisperModel(
                    self._cfg.whisper_model,
                    device=self._cfg.whisper_device,
                    compute_type=self._cfg.whisper_compute_type,
                )
            return self._model

    def _record_utterance(self, mic: MicStream) -> np.ndarray | None:
        """Buffer frames until VAD endpoints or the hard timeout fires."""
        vad = SileroEndpointer(self._cfg)
        vad.reset()

        buffer: list[np.ndarray] = []
        deadline = time.monotonic() + self._cfg.listen_timeout_s

        while time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                frame = mic.read(timeout=remaining)
            except queue.Empty:
                break

            buffer.append(frame)
            decision = vad.feed(frame)
            if decision == "stop":
                break

        if not buffer:
            return None
        return np.concatenate(buffer)

    def warmup(self) -> None:
        """Load the Whisper model now so the first listen_once isn't slow."""
        self._ensure_model()
