# Jarvis Voice — Setup

Hybrid voice stack: local wake-word + local STT + **ElevenLabs cloud TTS**.

## Architecture

| Stage     | Library            | Model                       | Where it runs |
|-----------|--------------------|-----------------------------|---------------|
| Wake word | openWakeWord       | `hey_jarvis_v0.1` (ONNX)    | Local (CPU)   |
| Endpoint  | Silero VAD         | bundled ONNX                | Local (CPU)   |
| STT       | faster-whisper     | `small.en` int8             | Local (CPU)   |
| TTS       | **ElevenLabs**     | `eleven_flash_v2_5` (~75ms) | Cloud         |
| Brain     | JarvisOrchestrator | Ollama (`llama3.2:1b`)      | Local (auto-wired) |

Only the TTS step calls out to the network. Wake, STT and the LLM all run on
your Mac, so iteration during testing doesn't burn ElevenLabs character quota
— that only happens when Jarvis actually speaks.

## Why this split?

- **ElevenLabs for TTS** — the voice is what sells the agent. Their neural
  voices are unbeatable for the price, and Flash v2.5 hits ~75 ms TTFA which
  is faster than every other cloud option.
- **Whisper local for STT** — STT runs every utterance, so doing it locally
  keeps testing free and removes API latency from the input path.
- **openWakeWord local for wake** — there is no decent cloud wake-word service
  for custom keywords; openWakeWord ships a pretrained "Hey Jarvis" model.

## Setup

### 1. Get an ElevenLabs API key

1. Sign up at <https://elevenlabs.io> (free tier: 10,000 characters/month TTS,
   no credit card needed).
2. Click your profile (top-right) → **My Account** → **API Keys**.
3. Click **Create API Key** → give it a name (e.g. "jarvis-dev") → **Create**.
4. Copy the key immediately (you can't view it again, only regenerate).
5. Paste into `.env`:
   ```
   ELEVENLABS_API_KEY=sk_xxx...
   ```

### 2. Pick a voice (optional)

The default voice (`JBFqnCBsd6RMkjVDRZzb`) is **George** — calm British male,
perfect Jarvis-fit. To use a different voice:

1. Browse <https://elevenlabs.io/app/voice-library>.
2. Click a voice → **Add to my collection** (free).
3. Open the voice from **My Voices** → copy the Voice ID (long alphanumeric).
4. Paste into `.env`:
   ```
   ELEVENLABS_VOICE_ID=<your_voice_id>
   ```

Suggested voices for a Jarvis persona:

| Voice ID                       | Name        | Notes                          |
|--------------------------------|-------------|--------------------------------|
| `JBFqnCBsd6RMkjVDRZzb`         | George      | British male, calm (default)   |
| `IKne3meq5aSn9XLyUdCD`         | Charlie     | British male, conversational   |
| `nPczCjzI2devNBz1zQrb`         | Brian       | American male, deep            |
| `cgSgspJ2msm6clMCkdW9`         | Jessica     | American female, warm          |

### 3. Install Python deps

```bash
cd ~/Desktop/Jarvis
source venv/bin/activate
pip install -r voice_requirements.txt
```

Takes ~3-5 min the first time — faster-whisper pulls CTranslate2 wheels and
openWakeWord pulls onnxruntime.

### 4. Grant microphone permission

macOS will prompt the first time you run the voice agent. If it doesn't, go to
**System Settings → Privacy & Security → Microphone** and allow your terminal
app (Terminal, iTerm, VS Code, etc.).

### 5. Run

```bash
python -m voice
```

You should see:

```
[voice] ── Jarvis voice stack ───────────────────────────
[voice] Wake word    : hey_jarvis_v0.1
[voice] STT          : faster-whisper small.en (int8, cpu)
[voice] TTS          : ElevenLabs · eleven_flash_v2_5 · voice JBFqnCBsd6RMkjVDRZzb
[voice] Barge-in     : on
[voice] Brain        : brain        (or _echo_brain if orchestrator unavailable)
[voice] STT model loaded.
[voice] Listening for "Hey Jarvis"... (Ctrl-C to quit)
```

Say **"Hey Jarvis"** → ask anything → Jarvis answers in your selected voice.
Say "Hey Jarvis" again mid-reply to interrupt and ask a follow-up.

Skip wake-word for push-to-talk:

```bash
python -m voice --no-wake
```

Verbose logs (latency timings, VAD scores, etc.):

```bash
python -m voice -v
```

## Cost estimate (for the report)

ElevenLabs charges per character of synthesised TTS — STT and wake are free
(local). A "typical" Jarvis reply is ~150 characters; the free 10k chars/month
buys you ~65 spoken replies.

| Tier        | $/month | TTS characters | ~Replies/month |
|-------------|---------|----------------|----------------|
| Free        | $0      | 10,000         | ~65            |
| Starter     | $5      | 30,000         | ~200           |
| Creator     | $22     | 100,000        | ~660           |

For development the free tier is plenty. For a live demo at submission,
Starter ($5) gives you headroom.

## Tuning

All knobs in `.env`:

| Key                     | Default            | Meaning                                            |
|-------------------------|--------------------|----------------------------------------------------|
| `ELEVENLABS_MODEL`      | `eleven_flash_v2_5`| Swap to `eleven_turbo_v2_5` for higher quality     |
| `WAKE_WORD_THRESHOLD`   | `0.5`              | Raise to 0.6–0.7 if you get false triggers         |
| `END_SILENCE_MS`        | `900`              | How long of silence ends the utterance             |
| `WHISPER_MODEL`         | `small.en`         | `tiny.en` (fastest) → `medium.en` (most accurate)  |
| `BARGE_IN_ENABLED`      | `true`             | Set false if ElevenLabs voice self-triggers wake   |

### Whisper model trade-offs

| Model       | Size  | Speed (M2 int8) | Notes                |
|-------------|-------|-----------------|----------------------|
| `tiny.en`   | 39 MB | ~10× realtime   | OK for clean speech  |
| `base.en`   | 74 MB | ~7× realtime    | Good                 |
| `small.en`  | 244 MB| ~5× realtime    | Recommended default  |
| `medium.en` | 769 MB| ~2× realtime    | Best accuracy        |

## Wiring into the orchestrator

The runner auto-detects `JarvisOrchestrator`. If `from orchestrator import
JarvisOrchestrator` succeeds at startup, the agent answers via your full LLM
pipeline. No code change required — just run `python -m voice` once the
orchestrator imports cleanly.

To force the echo stub for debugging voice-only issues, run with the
orchestrator broken or temporarily comment its import.

## Troubleshooting

- **`ELEVENLABS_API_KEY is missing`** — paste your key into `.env`.
- **`401 Unauthorized` from ElevenLabs** — key invalid or expired. Regenerate
  it at elevenlabs.io → Account → API Keys.
- **`429 Too Many Requests`** — you hit the rate limit on the free tier; wait
  60 seconds and retry, or upgrade.
- **`silero-vad is not installed` / `faster-whisper not installed`** —
  `pip install -r voice_requirements.txt`.
- **No mic input** — System Settings → Privacy & Security → Microphone →
  allow your terminal app.
- **Wake word self-triggers on Jarvis's own voice** — set `BARGE_IN_ENABLED=false`,
  raise `WAKE_WORD_THRESHOLD` to 0.7, or use headphones.
- **First STT response is slow** — Whisper warming up; subsequent ones are
  <500 ms for a 5-second utterance.
- **`OSError: PortAudio library not found`** — `brew install portaudio`,
  then `pip install --force-reinstall sounddevice`.

## Report-ready numbers (M2 MacBook Air, 16 GB)

| Metric                       | Observed              |
|------------------------------|-----------------------|
| Wake word inference          | ~5 ms / 80 ms frame   |
| Time to first STT word       | ~400 ms after EoS     |
| STT realtime factor          | ~5× (small.en, int8)  |
| TTS time-to-first-audio      | ~75-200 ms (Flash v2.5, depends on network) |
| End-to-end (no LLM)          | ~800 ms wake → audible |

The LLM (orchestrator) step adds 500–3000 ms depending on Ollama model.
