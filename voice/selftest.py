"""
Jarvis Voice — end-to-end self-test / diagnostic.

Run this on your Mac (where the mic, speakers, Ollama and the venv all live)
to confirm every stage of the voice pipeline is healthy before a demo:

    cd ~/Desktop/Jarvis
    source venv/bin/activate
    python -m voice.selftest                 # full check
    python -m voice.selftest --no-mic        # skip the mic capture step
    python -m voice.selftest --no-llm        # skip the Ollama brain step

It checks, in order:
    1. .env config loads + ElevenLabs key present
    2. Python deps import (faster-whisper, silero-vad, elevenlabs, sounddevice)
    3. Ollama reachable + chat model responds (the "brain")
    4. ElevenLabs key valid (HTTP /v1/user) + quota
    5. Audio output device present
    6. Microphone actually captures non-silent audio (3s)
    7. Whisper loads + transcribes the captured audio
    8. ElevenLabs synthesises + plays a short confirmation line

Each step prints PASS / WARN / FAIL with an actionable hint. Exit code is
non-zero if any hard FAIL occurred, so it's CI/script friendly.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
import urllib.error

PASS = "\033[92mPASS\033[0m"
WARN = "\033[93mWARN\033[0m"
FAIL = "\033[91mFAIL\033[0m"


class Report:
    def __init__(self) -> None:
        self.failures = 0
        self.warnings = 0

    def ok(self, step: str, detail: str = "") -> None:
        print(f"  [{PASS}] {step}" + (f" — {detail}" if detail else ""))

    def warn(self, step: str, detail: str = "") -> None:
        self.warnings += 1
        print(f"  [{WARN}] {step}" + (f" — {detail}" if detail else ""))

    def fail(self, step: str, detail: str = "", hint: str = "") -> None:
        self.failures += 1
        print(f"  [{FAIL}] {step}" + (f" — {detail}" if detail else ""))
        if hint:
            print(f"         ↳ {hint}")


def _section(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m")


def check_config(r: Report):
    _section("1. Config")
    try:
        from voice.config import load_config
        cfg = load_config()
    except Exception as exc:  # noqa: BLE001
        r.fail("load .env config", str(exc),
               "Confirm voice/config.py imports and .env exists at project root.")
        return None
    r.ok("config loaded",
         f"whisper={cfg.whisper_model}/{cfg.whisper_compute_type}, "
         f"end_silence={cfg.end_silence_ms}ms, vad={cfg.vad_threshold}, "
         f"voice={cfg.elevenlabs_voice_id}, model={cfg.elevenlabs_model}")
    if not cfg.elevenlabs_api_key:
        r.fail("ELEVENLABS_API_KEY", "empty",
               "Paste a key into .env (free tier at https://elevenlabs.io).")
    elif cfg.elevenlabs_api_key != cfg.elevenlabs_api_key.strip():
        r.warn("ELEVENLABS_API_KEY", "has surrounding whitespace — trim it in .env")
    else:
        r.ok("ELEVENLABS_API_KEY", f"present (…{cfg.elevenlabs_api_key[-4:]})")
    return cfg


def check_imports(r: Report):
    _section("2. Dependencies")
    for mod, pip_name in [
        ("numpy", "numpy"),
        ("sounddevice", "sounddevice"),
        ("faster_whisper", "faster-whisper"),
        ("silero_vad", "silero-vad"),
        ("elevenlabs", "elevenlabs"),
    ]:
        try:
            __import__(mod)
            r.ok(f"import {mod}")
        except Exception as exc:  # noqa: BLE001
            r.fail(f"import {mod}", str(exc),
                   f"pip install {pip_name}  (or: pip install -r voice_requirements.txt)")


def check_ollama(r: Report, cfg):
    _section("3. Ollama brain")
    import os
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:latest")
    # 3a. server reachable
    try:
        with urllib.request.urlopen(f"{base}/api/tags", timeout=5) as resp:
            import json
            tags = json.loads(resp.read())
        names = [m.get("name") for m in tags.get("models", [])]
    except Exception as exc:  # noqa: BLE001
        r.fail("Ollama reachable", f"{base}: {exc}",
               "Start it with `ollama serve`, then `ollama pull llama3.2:latest`.")
        return
    r.ok("Ollama reachable", f"{len(names)} model(s) installed")
    if model not in names:
        r.warn(f"model {model} not pulled",
               f"installed: {', '.join(names) or '(none)'} — run `ollama pull {model}`")
    # 3b. a real generation round-trip (warms the model too)
    try:
        import json
        payload = json.dumps({
            "model": model,
            "prompt": "Reply with exactly: ready",
            "stream": False,
            "options": {"num_predict": 8},
        }).encode()
        t0 = time.perf_counter()
        req = urllib.request.Request(f"{base}/api/generate", data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            out = json.loads(resp.read())
        dt = time.perf_counter() - t0
        text = (out.get("response") or "").strip()
        r.ok("brain responds", f"{dt:.1f}s → {text[:40]!r}")
        if dt > 15:
            r.warn("brain latency high", f"{dt:.1f}s — first call cold-loads; "
                   "keep_alive should keep it warm afterwards.")
    except Exception as exc:  # noqa: BLE001
        r.fail("brain generation", str(exc),
               "Check the model is pulled and Ollama has enough RAM.")


def check_elevenlabs(r: Report, cfg):
    _section("4. ElevenLabs key")
    if not cfg.elevenlabs_api_key:
        r.fail("skip", "no API key in .env")
        return
    try:
        req = urllib.request.Request(
            "https://api.elevenlabs.io/v1/user",
            headers={"xi-api-key": cfg.elevenlabs_api_key},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            import json
            info = json.loads(resp.read())
        sub = info.get("subscription", {})
        used = sub.get("character_count", "?")
        limit = sub.get("character_limit", "?")
        tier = sub.get("tier", "?")
        r.ok("key valid", f"tier={tier}, used {used}/{limit} chars")
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            r.fail("key rejected (401)", "invalid/revoked",
                   "Generate a new key at elevenlabs.io → Account → API Keys.")
        else:
            r.warn("ElevenLabs HTTP", f"{exc.code}")
    except Exception as exc:  # noqa: BLE001
        r.warn("ElevenLabs unreachable", str(exc), )


def check_audio_out(r: Report):
    _section("5. Audio output")
    try:
        import sounddevice as sd
        out_idx = sd.default.device[1]
        name = sd.query_devices(out_idx)["name"]
        r.ok("output device", f"#{out_idx} {name!r}")
    except Exception as exc:  # noqa: BLE001
        r.fail("output device", str(exc),
               "Check System Settings → Sound → Output.")


def check_mic(r: Report, cfg, duration: float = 3.0):
    _section("6. Microphone capture")
    try:
        import numpy as np
        from voice.audio import MicStream
        from voice.vad import SileroEndpointer
    except Exception as exc:  # noqa: BLE001
        r.fail("mic deps", str(exc))
        return None
    print(f"  → Speak now for {duration:.0f}s (say a full sentence)...")
    mic = MicStream(cfg)
    try:
        mic.start()
    except Exception as exc:  # noqa: BLE001
        r.fail("open mic", str(exc),
               "Grant mic permission: System Settings → Privacy & Security → Microphone.")
        return None
    vad = SileroEndpointer(cfg)
    vad.reset()
    frames, peak = [], 0
    deadline = time.monotonic() + duration
    import queue as _q
    while time.monotonic() < deadline:
        try:
            f = mic.read(timeout=max(0.0, deadline - time.monotonic()))
        except _q.Empty:
            break
        frames.append(f)
        peak = max(peak, int(np.max(np.abs(f))) if f.size else 0)
        vad.feed(f)
    mic.stop()
    if not frames:
        r.fail("captured audio", "no frames", "Mic produced nothing — permission/device issue.")
        return None
    audio = np.concatenate(frames)
    if peak < 100:
        r.fail("mic level", f"peak={peak} (digital silence)",
               "macOS is feeding zeros — grant mic permission to your terminal app.")
        return None
    if peak < 800:
        r.warn("mic level low", f"peak={peak} — speak louder/closer for best STT")
    else:
        r.ok("mic level", f"peak={peak}, vad_max={vad.max_prob:.2f}")
    return audio


def check_whisper(r: Report, cfg, audio):
    _section("7. Whisper STT")
    if audio is None:
        r.warn("skip", "no audio captured")
        return
    try:
        import numpy as np
        from voice.stt import StreamingSTT
        stt = StreamingSTT(cfg)
        t0 = time.perf_counter()
        stt.warmup()
        load_s = time.perf_counter() - t0
        model = stt._ensure_model()  # noqa: SLF001
        t0 = time.perf_counter()
        segments, _ = model.transcribe(audio.astype(np.float32) / 32768.0,
                                       beam_size=1, language="en",
                                       condition_on_previous_text=False,
                                       vad_filter=False)
        text = " ".join((s.text or "").strip() for s in segments).strip()
        dec_s = time.perf_counter() - t0
        if text:
            r.ok("transcribe", f"load={load_s:.1f}s decode={dec_s:.2f}s → {text!r}")
        else:
            r.warn("transcribe", "empty result — speak louder next run")
    except Exception as exc:  # noqa: BLE001
        r.fail("whisper", str(exc), "pip install faster-whisper")


def check_tts(r: Report, cfg):
    _section("8. ElevenLabs speak")
    if not cfg.elevenlabs_api_key:
        r.warn("skip", "no API key")
        return
    try:
        from voice.tts import StreamingTTS
        tts = StreamingTTS(cfg)
        phrase = "Voice self test complete. All systems operational, sir."
        print(f"  → Playing: {phrase!r}")
        t0 = time.perf_counter()
        tts.speak(phrase)
        r.ok("spoke phrase", f"{time.perf_counter() - t0:.1f}s (incl. playback)")
    except Exception as exc:  # noqa: BLE001
        r.fail("tts speak", str(exc),
               "401 = bad key; 'text_to_speech' permission error = recreate key with TTS access.")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m voice.selftest",
                                description="Jarvis voice pipeline diagnostic.")
    p.add_argument("--no-mic", action="store_true", help="skip mic + STT steps")
    p.add_argument("--no-llm", action="store_true", help="skip the Ollama brain step")
    p.add_argument("--no-tts", action="store_true", help="skip the ElevenLabs speak step")
    p.add_argument("--mic-seconds", type=float, default=3.0)
    args = p.parse_args(argv)

    print("\033[1m═══ Jarvis Voice — Self Test ═══\033[0m")
    r = Report()

    cfg = check_config(r)
    check_imports(r)
    if cfg is None:
        print("\nAborting — config failed to load.")
        return 1
    if not args.no_llm:
        check_ollama(r, cfg)
    check_elevenlabs(r, cfg)
    check_audio_out(r)
    audio = None
    if not args.no_mic:
        audio = check_mic(r, cfg, duration=args.mic_seconds)
        check_whisper(r, cfg, audio)
    if not args.no_tts:
        check_tts(r, cfg)

    _section("Summary")
    if r.failures == 0 and r.warnings == 0:
        print("  \033[92mAll checks passed. Voice agent is ready.\033[0m")
    else:
        print(f"  {r.failures} failure(s), {r.warnings} warning(s).")
    return 1 if r.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
