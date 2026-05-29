"""
Voice Activity Detection using the Silero VAD ONNX model.

Silero VAD takes 30 ms windows of 16 kHz int16 audio and returns a
speech-probability in [0, 1]. We wrap it as a stateful endpointer:

    vad = SileroEndpointer(config)
    vad.reset()
    if vad.feed(frame_int16) == "stop":
        # user finished talking — flush the buffered audio to STT

It's dramatically more robust than RMS thresholding because the model
understands what speech *sounds* like rather than just measuring loudness, so
keyboard clatter, fan noise, and AC hum don't falsely extend the utterance.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

from .config import VoiceConfig

log = logging.getLogger("jarvis.voice.vad")

Decision = Literal["continue", "stop"]


class SileroEndpointer:
    """Stateful Silero VAD endpointer for a single utterance."""

    # Silero VAD requires exactly 512 samples (32 ms @ 16k) per inference
    _WINDOW = 512

    def __init__(self, config: VoiceConfig):
        self._cfg = config

        # `silero_vad` exposes a single `load_silero_vad` factory
        try:
            from silero_vad import load_silero_vad
        except ImportError as exc:
            raise RuntimeError(
                "silero-vad is not installed. Run: pip install -r voice_requirements.txt"
            ) from exc

        import torch  # silero-vad ships a torch model under the hood

        self._torch = torch
        self._model = load_silero_vad(onnx=True)  # ONNX = faster, no torch eager mode
        self._spillover = np.zeros(0, dtype=np.int16)
        self._silence_ms = 0
        self._has_seen_speech = False
        # ── Telemetry ──────────────────────────────────────────────────────
        # last_prob: most recent Silero speech probability (0..1) — used by
        # the diagnostic endpoints and the live "I hear you" orb feedback.
        # max_prob: highest speech probability ever observed this utterance —
        # lets us tell "mic was silent" from "mic captured something quiet".
        self._last_prob = 0.0
        self._max_prob = 0.0

    # ── Public API ─────────────────────────────────────────────────────────

    def reset(self) -> None:
        self._spillover = np.zeros(0, dtype=np.int16)
        self._silence_ms = 0
        self._has_seen_speech = False
        self._last_prob = 0.0
        self._max_prob = 0.0
        try:
            self._model.reset_states()
        except AttributeError:
            pass  # ONNX wrapper doesn't always expose reset_states

    def feed(self, frame_int16: np.ndarray) -> Decision:
        """
        Feed one mic frame. Returns "stop" once silence has persisted long
        enough after at least one speech window, otherwise "continue".
        """
        # Concatenate with any leftover samples and chop into 512-sample windows
        buf = np.concatenate([self._spillover, frame_int16])
        n_full = len(buf) // self._WINDOW
        usable = buf[: n_full * self._WINDOW]
        self._spillover = buf[n_full * self._WINDOW :]

        if n_full == 0:
            return "continue"

        windows = usable.reshape(n_full, self._WINDOW)
        # Normalise int16 → float32 in [-1, 1]
        windows_f = windows.astype(np.float32) / 32768.0

        # Silero exposes a __call__ that runs inference window by window
        for w in windows_f:
            tensor = self._torch.from_numpy(w)
            with self._torch.no_grad():
                prob = float(self._model(tensor, self._cfg.sample_rate))
            self._last_prob = prob
            if prob > self._max_prob:
                self._max_prob = prob
            window_ms = int(self._WINDOW * 1000 / self._cfg.sample_rate)

            if prob >= self._cfg.vad_threshold:
                self._silence_ms = 0
                self._has_seen_speech = True
            else:
                self._silence_ms += window_ms

            if (
                self._has_seen_speech
                and self._silence_ms >= self._cfg.end_silence_ms
            ):
                return "stop"

        return "continue"

    @property
    def has_seen_speech(self) -> bool:
        return self._has_seen_speech

    @property
    def last_prob(self) -> float:
        """Most recent Silero speech probability (0..1)."""
        return self._last_prob

    @property
    def max_prob(self) -> float:
        """Highest speech probability seen during this utterance."""
        return self._max_prob
