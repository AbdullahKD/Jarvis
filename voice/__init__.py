"""
Jarvis voice subsystem — local stack.

Stack:
  - openWakeWord  → wake-word detection ("Hey Jarvis", pretrained ONNX)
  - Silero VAD    → end-of-utterance detection
  - faster-whisper → speech-to-text (small.en, int8 on CPU)
  - Piper          → text-to-speech (en_GB-alan-medium, streaming)

Public API (lazily loaded so `voice.config` is importable even before the
heavy ML deps are installed):

    from voice import VoiceRunner, StreamingTTS, StreamingSTT, WakeWordListener
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
    "WakeWordListener",
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
    "WakeWordListener": "voice.wake_word",
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
    from .wake_word import WakeWordListener  # noqa: F401
