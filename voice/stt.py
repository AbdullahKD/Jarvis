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


class StreamingSTTError(RuntimeError):
    """Raised when STT can't proceed for an environmental reason (e.g. mic silent)."""


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
        on_status: Optional[Callable[[dict], None]] = None,
    ) -> str:
        """
        Capture one utterance, transcribe it, return the text.

        If `mic` is None, opens its own MicStream. Pass a shared MicStream when
        composing with the wake-word listener to avoid device contention.

        `on_status` (optional) is invoked ~every mic frame with a dict like:
            {"peak": 0.0..1.0, "vad_prob": 0.0..1.0,
             "has_speech": bool, "frames": int, "elapsed_ms": int}
        It's the hook the web UI uses to drive the live "I hear you" orb.

        Raises StreamingSTTError on hard failure (e.g. mic produced only
        digital silence — usually a permission or device problem).
        """
        own_mic = mic is None
        if own_mic:
            mic = MicStream(self._cfg)
            mic.start()

        try:
            audio = self._record_utterance(mic, on_status=on_status)
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

    def _record_utterance(
        self,
        mic: MicStream,
        *,
        on_status: Optional[Callable[[dict], None]] = None,
    ) -> np.ndarray | None:
        """
        Buffer frames until VAD endpoints or the hard timeout fires.

        Two safety behaviours layered on top of the plain VAD loop:

        1. **Fail-fast on digital silence.** If the first ~1.5 s of frames
           contains peak amplitudes below DIGITAL_SILENCE_PEAK, the mic is
           almost certainly producing zeros (no permission, wrong device, or
           muted). We abort early with a StreamingSTTError so the UI can show
           an actionable message instead of stalling for the full timeout.

        2. **Audible-energy fallback.** If the listen timeout fires *without*
           VAD ever endpointing, but we did hear something (max amplitude
           above MIN_FALLBACK_PEAK, or VAD max-prob above 0.25), still return
           the buffer — Whisper can usually transcribe quiet speech that
           Silero rejected.

        `on_status`, when supplied, gets called every frame so the web layer
        can show a live audio-level indicator.
        """
        # Tunables — units explained inline so .env overrides are obvious.
        DIGITAL_SILENCE_PEAK = 100         # int16 peak under this = effectively zero
        SILENCE_CHECK_FRAMES = 19          # ~1.5 s @ 80-ms frames before fail-fast
        MIN_FALLBACK_PEAK = 800            # absolute peak that justifies fallback transcribe
        MIN_FALLBACK_PROB = 0.25           # OR Silero max-prob ≥ this

        vad = SileroEndpointer(self._cfg)
        vad.reset()

        buffer: list[np.ndarray] = []
        overall_peak = 0
        frames_seen = 0
        start_t = time.monotonic()
        deadline = start_t + self._cfg.listen_timeout_s

        while time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                frame = mic.read(timeout=remaining)
            except queue.Empty:
                break

            buffer.append(frame)
            frames_seen += 1
            frame_peak = int(np.max(np.abs(frame))) if frame.size else 0
            if frame_peak > overall_peak:
                overall_peak = frame_peak

            decision = vad.feed(frame)

            if on_status is not None:
                try:
                    on_status({
                        "peak": frame_peak / 32768.0,
                        "vad_prob": vad.last_prob,
                        "max_prob": vad.max_prob,
                        "has_speech": vad.has_seen_speech,
                        "frames": frames_seen,
                        "elapsed_ms": int((time.monotonic() - start_t) * 1000),
                    })
                except Exception:  # noqa: BLE001
                    pass  # never let a UI hook break the audio loop

            # Fail-fast: after ~1.5 s of frames we should have seen *some*
            # audio. If overall_peak is still essentially zero, the mic is
            # not producing real samples — bail out with a clear error.
            if (
                frames_seen == SILENCE_CHECK_FRAMES
                and overall_peak < DIGITAL_SILENCE_PEAK
            ):
                raise StreamingSTTError(
                    "Microphone produced only digital silence. "
                    "Check System Settings → Privacy & Security → Microphone "
                    "and confirm your terminal app (or VS Code) is allowed. "
                    "Then list devices via GET /voice/devices and set "
                    "AUDIO_INPUT_DEVICE=<index> in .env if the wrong one is "
                    "selected."
                )

            if decision == "stop":
                break

        if not buffer:
            return None

        timeout_fired = time.monotonic() >= deadline and not vad.has_seen_speech
        if timeout_fired:
            # We never endpointed. If there's anything audible, send it
            # through Whisper anyway — better than silently giving up.
            if overall_peak < MIN_FALLBACK_PEAK and vad.max_prob < MIN_FALLBACK_PROB:
                log.info(
                    "stt: listen timeout with no speech (peak=%d, max_prob=%.2f)",
                    overall_peak, vad.max_prob,
                )
                return None
            log.info(
                "stt: VAD never endpointed, but audio was audible "
                "(peak=%d, max_prob=%.2f) — falling through to Whisper",
                overall_peak, vad.max_prob,
            )

        return np.concatenate(buffer)

    def warmup(self) -> None:
        """Load the Whisper model now so the first listen_once isn't slow."""
        self._ensure_model()
