"""
Low-level audio plumbing — mic capture + speaker playback via sounddevice.

Centralised here so the higher-level voice modules (wake_word, stt, tts) don't
each carry their own audio code. All formats are int16 mono at the configured
sample rate (default 16 kHz), which is what openWakeWord, Silero VAD, and
faster-whisper all expect.
"""

from __future__ import annotations

import logging
import queue
import subprocess
import threading
from contextlib import contextmanager
from typing import Iterator, Optional

import numpy as np
import sounddevice as sd

from .config import VoiceConfig

log = logging.getLogger("jarvis.voice.audio")

# openWakeWord expects 80 ms chunks at 16 kHz = 1280 samples. Most of the rest
# of the pipeline is fine with any chunk size, so we use this as the global.
CHUNK_SAMPLES = 1280


class MicStream:
    """
    Single-producer streaming mic capture.

    Hands out int16 mono frames of CHUNK_SAMPLES at the config sample rate.
    The capture runs on a sounddevice callback thread; consumers pull frames
    from a thread-safe queue via `read()`.
    """

    def __init__(self, config: VoiceConfig, *, queue_size: int = 50):
        self._cfg = config
        self._q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=queue_size)
        self._stream: Optional[sd.InputStream] = None

    def _callback(self, indata, frames, time_info, status):
        if status:
            log.debug("mic status: %s", status)
        # indata arrives as int16 mono (channels=1) — push a copy so the SDK
        # can reuse its buffer.
        try:
            self._q.put_nowait(indata.copy().reshape(-1))
        except queue.Full:
            # Drop the oldest frame and try again — prevents unbounded growth
            # if the consumer falls behind.
            try:
                self._q.get_nowait()
                self._q.put_nowait(indata.copy().reshape(-1))
            except queue.Empty:
                pass

    def start(self) -> None:
        if self._stream is not None:
            return
        self._stream = sd.InputStream(
            samplerate=self._cfg.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=CHUNK_SAMPLES,
            device=self._cfg.input_device,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is None:
            return
        try:
            self._stream.stop()
            self._stream.close()
        finally:
            self._stream = None
        # Drain the queue
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

    def read(self, timeout: float | None = None) -> np.ndarray:
        """Block until a frame is available; raises queue.Empty on timeout."""
        return self._q.get(timeout=timeout)

    def frames(self, timeout: float | None = None) -> Iterator[np.ndarray]:
        """Iterate frames until the stream is stopped or timeout expires."""
        while self._stream is not None:
            try:
                yield self._q.get(timeout=timeout)
            except queue.Empty:
                return

    @contextmanager
    def open(self):
        """Context manager: starts the stream on entry, stops on exit."""
        self.start()
        try:
            yield self
        finally:
            self.stop()


def play_pcm_stream(
    pcm_iter: Iterator[bytes],
    *,
    sample_rate: int,
    stop_event: threading.Event,
) -> None:
    """
    Play a stream of raw PCM int16 bytes through the default output device.

    Yields control back to the caller as soon as `stop_event` is set, killing
    audio mid-utterance — that's how the runner interrupts Jarvis on barge-in.
    """
    with sd.RawOutputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
        blocksize=0,
    ) as out:
        for chunk in pcm_iter:
            if stop_event.is_set():
                log.debug("playback interrupted by stop_event")
                break
            if not chunk:
                continue
            out.write(chunk)


def piper_synth_pcm_stream(
    binary: str,
    model: str,
    config_json: str,
    text: str,
) -> Iterator[bytes]:
    """
    Invoke the Piper binary in streaming mode and yield raw PCM bytes.

    Piper outputs 22050 Hz mono int16 PCM on stdout when given `--output-raw`.
    We read it in 4 KB chunks so the first audio reaches the speaker within
    ~150 ms of pressing Enter.
    """
    proc = subprocess.Popen(
        [binary, "--model", model, "--config", config_json, "--output-raw"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdin and proc.stdout

    # Feed text on a side thread so we can read stdout simultaneously
    def _feed():
        try:
            proc.stdin.write(text.encode("utf-8"))
            proc.stdin.close()
        except BrokenPipeError:
            pass

    threading.Thread(target=_feed, daemon=True).start()

    try:
        while True:
            chunk = proc.stdout.read(4096)
            if not chunk:
                break
            yield chunk
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                proc.kill()
