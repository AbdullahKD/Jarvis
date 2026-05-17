"""
Voice subsystem configuration — local stack.

Loads from `.env` at the project root. Everything has a sensible default so the
voice layer works out of the box once dependencies and models are installed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv

    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:  # python-dotenv not installed — assume env is already set
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent


class VoiceConfigError(RuntimeError):
    """Raised when a required voice resource (binary, model) is missing."""


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _resolve_path(value: str) -> Path:
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    return p


@dataclass(frozen=True)
class VoiceConfig:
    """Immutable view of the voice subsystem settings."""

    project_root: Path

    # ── Audio I/O ──────────────────────────────────────────────────────────
    sample_rate: int           # mic capture rate, fixed at 16k for STT + VAD
    input_device: int | None   # None = system default microphone
    output_device: int | None  # None = system default speaker

    # ── Wake word (openWakeWord) ───────────────────────────────────────────
    wake_word_enabled: bool
    wake_word_model: str       # built-in ID, e.g. "hey_jarvis_v0.1"
    wake_word_threshold: float # 0..1; higher = stricter

    # ── VAD (Silero) ───────────────────────────────────────────────────────
    vad_threshold: float       # speech-probability cutoff
    end_silence_ms: int        # silence before we stop listening
    listen_timeout_s: float    # hard cap on a single utterance

    # ── STT (faster-whisper) ───────────────────────────────────────────────
    whisper_model: str         # tiny.en | base.en | small.en | medium.en
    whisper_compute_type: str  # int8 | int8_float16 | float16 | float32
    whisper_device: str        # cpu | cuda | auto

    # ── TTS (ElevenLabs) ───────────────────────────────────────────────────
    elevenlabs_api_key: str
    elevenlabs_voice_id: str
    elevenlabs_model: str

    # ── Behaviour ──────────────────────────────────────────────────────────
    barge_in_enabled: bool     # interrupt TTS if wake word fires mid-speech


def load_config() -> VoiceConfig:
    """Load voice configuration from environment variables."""

    return VoiceConfig(
        project_root=_PROJECT_ROOT,
        # Audio
        sample_rate=int(os.getenv("AUDIO_SAMPLE_RATE", "16000")),
        input_device=_optional_int(os.getenv("AUDIO_INPUT_DEVICE")),
        output_device=_optional_int(os.getenv("AUDIO_OUTPUT_DEVICE")),
        # Wake word
        wake_word_enabled=_get_bool("WAKE_WORD_ENABLED", True),
        wake_word_model=os.getenv("WAKE_WORD_MODEL", "hey_jarvis_v0.1"),
        wake_word_threshold=float(os.getenv("WAKE_WORD_THRESHOLD", "0.5")),
        # VAD
        vad_threshold=float(os.getenv("VAD_THRESHOLD", "0.5")),
        end_silence_ms=int(os.getenv("END_SILENCE_MS", "900")),
        listen_timeout_s=float(os.getenv("LISTEN_TIMEOUT_SECONDS", "12")),
        # STT
        whisper_model=os.getenv("WHISPER_MODEL", "small.en"),
        whisper_compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
        whisper_device=os.getenv("WHISPER_DEVICE", "cpu"),
        # TTS (ElevenLabs)
        elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY", "").strip(),
        elevenlabs_voice_id=os.getenv(
            "ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb"  # George (British male)
        ),
        elevenlabs_model=os.getenv("ELEVENLABS_MODEL", "eleven_flash_v2_5"),
        # Behaviour
        barge_in_enabled=_get_bool("BARGE_IN_ENABLED", True),
    )


def _optional_int(value: str | None) -> int | None:
    if value is None or value.strip() == "":
        return None
    return int(value)
