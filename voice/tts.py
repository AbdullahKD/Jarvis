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

        # Build the expressive voice settings once. Wrapped so an older SDK
        # without one of these fields still works (we just drop the extras).
        self._voice_settings = None
        try:
            from elevenlabs import VoiceSettings
            try:
                self._voice_settings = VoiceSettings(
                    stability=config.elevenlabs_stability,
                    similarity_boost=config.elevenlabs_similarity,
                    style=config.elevenlabs_style,
                    use_speaker_boost=config.elevenlabs_speaker_boost,
                    speed=config.elevenlabs_speed,
                )
            except TypeError:
                # Older SDK without `speed`.
                self._voice_settings = VoiceSettings(
                    stability=config.elevenlabs_stability,
                    similarity_boost=config.elevenlabs_similarity,
                    style=config.elevenlabs_style,
                    use_speaker_boost=config.elevenlabs_speaker_boost,
                )
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not build VoiceSettings (%s) — using voice defaults", exc)

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

    def speak_sentence_stream(
        self,
        sentence_queue,
        on_first_audio=None,
    ) -> None:
        """
        Drain a queue of sentences and synthesise+play each one back-to-back
        through ElevenLabs Flash v2.5. Blocks the current thread.

        The queue contract:
          - put(str) → enqueue a sentence (will be spoken in order)
          - put(None) → sentinel: no more sentences, return when drained

        `on_first_audio` (optional, no-args) fires once the first sentence
        begins playing — used by VoiceSession to flip state from
        "thinking" → "speaking" so the UI orb updates the moment the user
        starts hearing audio.

        Pipelined design (eliminates the inter-sentence lag)
        ----------------------------------------------------
        A background *producer* thread pulls sentences off ``sentence_queue``,
        synthesises each via ElevenLabs, and pushes the raw PCM chunks onto an
        internal ``pcm_q``. THIS thread plays ``pcm_q`` as ONE continuous
        output stream.

        The win: synthesis of sentence N+1 runs *while* sentence N is still
        playing (the player consumes at real-time playback rate and
        back-pressures the producer through blocking writes), so there are no
        dead gaps between sentences and no per-sentence stream open/close
        clicks. The first sentence uses the fast/low-latency model for a
        snappy start; later sentences use the natural model, synthesised ahead
        of time so the extra latency is hidden behind earlier playback.
        """
        import queue as _queue
        import threading as _threading

        from .audio import play_pcm_stream as _play

        pcm_q: "_queue.Queue" = _queue.Queue()
        _SENTINEL = object()

        def _producer() -> None:
            first = True
            try:
                while True:
                    sentence = sentence_queue.get()
                    if sentence is None:
                        break
                    if not sentence or not sentence.strip():
                        continue
                    if self._stop.is_set():
                        break
                    # First sentence → fast model (snappy time-to-first-word);
                    # subsequent sentences → natural model, pre-synthesised
                    # while earlier audio is still playing so their higher
                    # latency never reaches the listener as a gap.
                    model = (
                        self._cfg.elevenlabs_fast_model if first
                        else self._cfg.elevenlabs_model
                    )
                    first = False
                    try:
                        for chunk in self._elevenlabs_stream(
                            sentence.strip(), model_override=model
                        ):
                            if self._stop.is_set():
                                break
                            if chunk:
                                pcm_q.put(chunk)
                    except Exception as exc:  # noqa: BLE001
                        # A single 429 / transient error shouldn't kill the
                        # whole turn — skip this sentence and keep going.
                        log.warning(
                            "speak_sentence_stream: sentence skipped: %s", exc
                        )
                        continue
            finally:
                pcm_q.put(_SENTINEL)

        producer = _threading.Thread(
            target=_producer, name="tts-synth-producer", daemon=True
        )
        producer.start()

        def _pcm_iter():
            while True:
                item = pcm_q.get()
                if item is _SENTINEL:
                    return
                yield item

        # One continuous stream for the whole answer: a single pre-roll at the
        # very start, then gap-free playback as sentences stream in.
        _play(
            _pcm_iter(),
            sample_rate=ELEVENLABS_SAMPLE_RATE,
            stop_event=self._stop,
            on_first_audio=on_first_audio,
        )

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

    def _pick_model(self, text: str, model_override: Optional[str]) -> str:
        """
        Choose the synthesis model. Short, quick replies use the fast/low-
        latency model so the answer is spoken almost immediately; longer text
        uses the natural model. Callers can force a model via model_override.
        """
        if model_override:
            return model_override
        if len(text.strip()) <= self._cfg.elevenlabs_fast_max_chars:
            return self._cfg.elevenlabs_fast_model
        return self._cfg.elevenlabs_model

    def _elevenlabs_stream(
        self, text: str, model_override: Optional[str] = None
    ) -> Iterator[bytes]:
        """Yield raw PCM int16 bytes from the ElevenLabs streaming endpoint."""
        model_id = self._pick_model(text, model_override)
        log.debug(
            "tts.elevenlabs: voice=%s model=%s len=%d",
            self._cfg.elevenlabs_voice_id,
            model_id,
            len(text),
        )
        kwargs = dict(
            voice_id=self._cfg.elevenlabs_voice_id,
            model_id=model_id,
            text=text,
            output_format=ELEVENLABS_OUTPUT_FORMAT,
        )
        if self._voice_settings is not None:
            kwargs["voice_settings"] = self._voice_settings
        try:
            stream = self._client.text_to_speech.stream(**kwargs)
        except TypeError:
            # SDK doesn't accept voice_settings on stream() — retry without it.
            kwargs.pop("voice_settings", None)
            stream = self._client.text_to_speech.stream(**kwargs)
        for chunk in stream:
            if self._stop.is_set():
                break
            if chunk:
                yield chunk
