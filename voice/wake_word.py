"""
"Hey Jarvis" wake-word detector using openWakeWord.

openWakeWord ships a pretrained `hey_jarvis_v0.1` ONNX model — no Speech Studio
trip, no API quota, no .table files to manage. Inference is ~5 ms per 80 ms
window on an M2, so we can run it continuously with negligible CPU.

Two listening modes:

- `wait_for_wake_word()` — blocking; opens its own mic stream and returns once
  the score crosses the threshold.
- `listen(mic, callback)` — non-blocking; reuses a shared mic stream and fires
  a callback. The runner uses this to listen for barge-ins *while TTS is
  playing*, so the user can interrupt Jarvis mid-sentence.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Callable, Optional

import numpy as np

from .audio import CHUNK_SAMPLES, MicStream
from .config import VoiceConfig

log = logging.getLogger("jarvis.voice.wake")


class WakeWordListener:
    """Continuous wake-word detection bound to the configured model."""

    def __init__(self, config: VoiceConfig):
        self._cfg = config

        try:
            from openwakeword.model import Model
            from openwakeword.utils import download_models
        except ImportError as exc:
            raise RuntimeError(
                "openwakeword is not installed. "
                "Run: pip install -r voice_requirements.txt"
            ) from exc

        # First-run-only: download the bundled ONNX models (cached afterwards)
        download_models()

        self._model = Model(
            wakeword_models=[config.wake_word_model],
            inference_framework="onnx",
        )

        self._stop_listen = threading.Event()
        self._listen_thread: Optional[threading.Thread] = None

    # ── Blocking single-shot detection ─────────────────────────────────────

    def wait_for_wake_word(self, *, timeout: Optional[float] = None) -> bool:
        """Open a mic, block until the wake word is detected. Returns True on hit."""
        detected = threading.Event()
        with MicStream(self._cfg).open() as mic:
            self._reset_model()
            try:
                while not detected.is_set():
                    try:
                        frame = mic.read(timeout=timeout)
                    except queue.Empty:
                        return False
                    if self._score_frame(frame) >= self._cfg.wake_word_threshold:
                        log.info("wake: detected")
                        return True
            finally:
                pass
        return False

    # ── Non-blocking listener (used during TTS for barge-in) ───────────────

    def start_listening(
        self,
        mic: MicStream,
        on_detected: Callable[[], None],
    ) -> None:
        """
        Spin up a background thread that pulls frames from `mic` and fires
        `on_detected()` the first time the wake word triggers. Caller is
        responsible for managing the mic stream's lifecycle.
        """
        if self._listen_thread is not None and self._listen_thread.is_alive():
            return  # already listening

        self._stop_listen.clear()
        self._reset_model()

        def _run():
            while not self._stop_listen.is_set():
                try:
                    frame = mic.read(timeout=0.2)
                except queue.Empty:
                    continue
                if self._score_frame(frame) >= self._cfg.wake_word_threshold:
                    log.info("wake: detected (background)")
                    try:
                        on_detected()
                    except Exception:  # noqa: BLE001
                        log.exception("wake on_detected callback raised")
                    return

        self._listen_thread = threading.Thread(target=_run, daemon=True)
        self._listen_thread.start()

    def stop_listening(self) -> None:
        self._stop_listen.set()
        if self._listen_thread is not None:
            self._listen_thread.join(timeout=1.0)
            self._listen_thread = None

    # ── Internals ──────────────────────────────────────────────────────────

    def _score_frame(self, frame: np.ndarray) -> float:
        # openWakeWord wants a flat 1D int16 array of CHUNK_SAMPLES (1280)
        if len(frame) < CHUNK_SAMPLES:
            # Pad short trailing frames with zeros so the model doesn't choke
            frame = np.pad(frame, (0, CHUNK_SAMPLES - len(frame)))
        elif len(frame) > CHUNK_SAMPLES:
            frame = frame[-CHUNK_SAMPLES:]
        prediction = self._model.predict(frame)
        return float(prediction.get(self._cfg.wake_word_model, 0.0))

    def _reset_model(self) -> None:
        try:
            self._model.reset()
        except AttributeError:
            pass
