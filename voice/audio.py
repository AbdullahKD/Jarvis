"""
Low-level audio plumbing — mic capture + speaker playback via sounddevice.

Centralised here so the higher-level voice modules (stt, tts) don't each
carry their own audio code. All formats are int16 mono at the configured
sample rate (default 16 kHz), which is what Silero VAD and faster-whisper
both expect.
"""

from __future__ import annotations

import collections
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

# 80 ms chunks at 16 kHz = 1280 samples. This was historically pinned by
# openWakeWord; we keep the same chunk size now that wake-word is gone so
# Silero VAD's 30-ms windows still tile cleanly with no awkward residuals.
CHUNK_SAMPLES = 1280


class MicStream:
    """
    Single-producer streaming mic capture.

    Hands out int16 mono frames of CHUNK_SAMPLES at the config sample rate.
    The capture runs on a sounddevice callback thread; consumers pull frames
    from a thread-safe queue via `read()`.

    Designed to be opened ONCE and kept alive across many listen sessions.
    `drain()` empties stale audio between sessions; the rolling pre-buffer
    holds the last ~600 ms of audio so words spoken just before /voice/start
    aren't lost.
    """

    # ~600 ms of pre-buffer at 80-ms frames (CHUNK_SAMPLES=1280 @ 16 kHz).
    # If the user clicks the mic button slightly after they start speaking,
    # we replay these frames into the recogniser so the first word survives.
    PREBUFFER_FRAMES = 8

    def __init__(self, config: VoiceConfig, *, queue_size: int = 50):
        self._cfg = config
        self._q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=queue_size)
        self._stream: Optional[sd.InputStream] = None
        # Rolling pre-buffer of recent frames captured while "idle". Updated
        # on every callback regardless of whether anyone is consuming.
        self._prebuffer: collections.deque = collections.deque(
            maxlen=self.PREBUFFER_FRAMES,
        )
        self._prebuffer_lock = threading.Lock()
        # When True, callbacks ALSO push frames into the main queue. We use
        # this to keep the device open in the background (pre-buffer only)
        # without filling the main queue when no listener is active.
        self._enqueue = True

    def _callback(self, indata, frames, time_info, status):
        if status:
            log.debug("mic status: %s", status)
        # indata arrives as int16 mono (channels=1) — copy so the SDK can
        # reuse its buffer, then update both the pre-buffer ring and the
        # main consumer queue.
        frame = indata.copy().reshape(-1)
        with self._prebuffer_lock:
            self._prebuffer.append(frame)
        if not self._enqueue:
            return
        try:
            self._q.put_nowait(frame)
        except queue.Full:
            # Drop the oldest frame and try again — prevents unbounded growth
            # if the consumer falls behind.
            try:
                self._q.get_nowait()
                self._q.put_nowait(frame)
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
    Stream PCM int16 bytes to the speaker in real time.

    Strategy:
      1. Pre-roll a small buffer (~250 ms) so we never underrun mid-sentence
         on slow network bursts — this is what eliminates the click/static
         from the previous collect-then-play approach.
      2. Open one `sd.RawOutputStream` for the whole utterance and `write()`
         chunks as they arrive from ElevenLabs. `write()` blocks when the
         ring buffer is full, so it natural-rate-limits without busy waiting.
      3. After the producer is exhausted, sleep for the device output
         latency so the tail of the audio actually plays out, then stop.

    Total perceived time-to-first-audio: ~75 ms (Flash v2.5 TTFA)
        + ~250 ms (pre-roll)
        ≈ 325 ms instead of the entire synthesis duration.
    """
    import time

    PREROLL_MS = 250            # pre-roll buffer to absorb network jitter
    BYTES_PER_SAMPLE = 2        # int16 mono
    preroll_bytes = int(sample_rate * (PREROLL_MS / 1000) * BYTES_PER_SAMPLE)

    output_device = sd.default.device[1]
    try:
        device_name = sd.query_devices(output_device)["name"]
    except Exception:  # noqa: BLE001
        device_name = f"device #{output_device}"

    pcm = iter(pcm_iter)

    # ── Phase 1: pre-roll ──────────────────────────────────────────────────
    preroll = bytearray()
    t_first_chunk: float | None = None
    for chunk in pcm:
        if stop_event.is_set():
            log.debug("playback pre-roll interrupted by stop_event")
            return
        if not chunk:
            continue
        if t_first_chunk is None:
            t_first_chunk = time.perf_counter()
        preroll.extend(chunk)
        if len(preroll) >= preroll_bytes:
            break

    if not preroll:
        print("⚠️  [audio] No audio bytes received from TTS stream (silent failure).")
        return

    # Make sure we hand the device complete int16 frames (multiple of 2 bytes).
    if len(preroll) % BYTES_PER_SAMPLE:
        preroll.append(0)

    # ── Phase 2: open the stream and start playing ─────────────────────────
    stream: sd.RawOutputStream | None = None
    total_bytes = 0
    try:
        stream = sd.RawOutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            blocksize=0,         # let PortAudio pick the optimal block size
            latency="low",       # request low-latency mode from CoreAudio
            device=output_device,
        )
        stream.start()
        print(
            f"🔊 [audio] Streaming → {device_name!r}  "
            f"(pre-roll {len(preroll):,}B / "
            f"{len(preroll) / (sample_rate * BYTES_PER_SAMPLE) * 1000:.0f}ms)"
        )
        # Flush pre-roll, then continue with the rest of the stream.
        stream.write(bytes(preroll))
        total_bytes += len(preroll)

        for chunk in pcm:
            if stop_event.is_set():
                log.debug("playback interrupted by stop_event")
                break
            if not chunk:
                continue
            # Defensive align — int16 frames must be 2-byte aligned.
            if len(chunk) % BYTES_PER_SAMPLE:
                chunk = chunk + b"\x00"
            stream.write(chunk)
            total_bytes += len(chunk)

        # ── Phase 3: drain ─────────────────────────────────────────────────
        # stream.write blocks while the buffer is full, so by the time we
        # reach here only the device-side latency is unplayed. Sleep for
        # that latency (plus a small margin) before stopping the stream —
        # this is what was being cut off in the old code and producing the
        # tail-of-sentence static.
        latency_s = float(stream.latency) if stream.latency else 0.05
        drain_deadline = time.perf_counter() + latency_s + 0.10
        while time.perf_counter() < drain_deadline:
            if stop_event.is_set():
                break
            time.sleep(0.01)

        duration_s = total_bytes / (sample_rate * BYTES_PER_SAMPLE)
        ttfa_ms = (
            (time.perf_counter() - t_first_chunk) * 1000
            if t_first_chunk
            else 0.0
        )
        print(
            f"🔊 [audio] Playback complete  "
            f"({duration_s:.2f}s audio, latency {latency_s * 1000:.0f}ms, "
            f"total stream {ttfa_ms:.0f}ms)"
        )

    except Exception as exc:  # noqa: BLE001
        print(f"❌ [audio] stream playback failed: {type(exc).__name__}: {exc}")
        # Last-resort fallback: dump everything we have to /tmp and use afplay.
        # We may have already burned the pre-roll into the (broken) stream;
        # afplay re-plays from the buffer we still hold.
        try:
            audio = np.frombuffer(bytes(preroll), dtype=np.int16)
            _afplay_fallback(audio, sample_rate)
        except Exception as exc2:  # noqa: BLE001
            print(f"❌ [audio] fallback also failed: {exc2}")
    finally:
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:  # noqa: BLE001
                pass


def _afplay_fallback(audio: np.ndarray, sample_rate: int) -> None:
    """If sounddevice can't play, dump to WAV and use macOS afplay."""
    try:
        import wave
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        with wave.open(tmp.name, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(audio.tobytes())
        print(f"🔊 [audio] Falling back to afplay → {tmp.name}")
        subprocess.run(["afplay", tmp.name], check=False)
    except Exception as exc:  # noqa: BLE001
        print(f"❌ [audio] afplay fallback also failed: {exc}")


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
