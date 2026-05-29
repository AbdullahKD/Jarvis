"""
Jarvis voice subsystem — local stack.

Stack:
  - Silero VAD    → end-of-utterance detection
  - faster-whisper → speech-to-text (tiny.en, int8 on CPU)
  - ElevenLabs Flash v2.5 → streaming TTS (~75 ms TTFA)

Wake-word ("Hey Jarvis") has been removed — voice is push-to-talk via the
UI mic button, which makes the demo experience predictable and avoids
acoustic-echo false triggers from Jarvis's own playback.

Public API (lazily loaded so `voice.config` is importable even before the
heavy ML deps are installed):

    from voice import VoiceRunner, StreamingTTS, StreamingSTT
    from voice import load_config, VoiceConfig, VoiceConfigError

Run the CLI loop with `python -m voice`. See VOICE_SETUP.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "VoiceConfig",
    "VoiceConfigError",
    "VoiceRunner",
    "StreamingTTS",
    "TTSError",
    "StreamingSTT",
    "load_config",
]

_LAZY_EXPORTS = {
    "VoiceConfig":      "voice.config",
    "VoiceConfigError": "voice.config",
    "load_config":      "voice.config",
    "VoiceRunner":      "voice.runner",
    "StreamingTTS":     "voice.tts",
    "TTSError":         "voice.tts",
    "StreamingSTT":     "voice.stt",
}


def __getattr__(name: str):
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module 'voice' has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_path)
    value = getattr(module, name)
    globals()[name] = value
    return value


if TYPE_CHECKING:
    from .config import VoiceConfig, VoiceConfigError, load_config  # noqa: F401
    from .runner import VoiceRunner  # noqa: F401
    from .stt import StreamingSTT  # noqa: F401
    from .tts import StreamingTTS, TTSError  # noqa: F401
