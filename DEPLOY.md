# Jarvis — Cloud Deployment Guide (Fly.io + Groq + Neon)

This guide walks you from zero to a public HTTPS URL hosting Jarvis. End-state:

- **App host**: Fly.io (Docker container, 2 GB RAM, London region)
- **LLM**: Groq (Llama 3.3 70B + Llama 3.1 8B for routing) — free tier
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2, bundled in the image
- **Postgres** (FinEx): Neon free tier
- **Vector store**: ChromaDB on a Fly persistent volume
- **Auth**: HTTP Basic Auth (single shared password for reviewers)

Total cost on free tiers: **$0/month** for light demo use.

---

## Architectural notes (read first)

1. **Voice input requires running locally on your Mac.** The voice stack uses
   server-side mic capture (sounddevice + Silero VAD + openWakeWord). A Fly.io
   container has no microphone, so `/voice/start` will degrade gracefully —
   reviewers can use text chat in the cloud and try voice on your local build.
2. **Mac-only tools** (open apps, brightness, volume, screenshots, Spotify
   `open -a`, desktop notifications) detect they're in a Linux container and
   return a "feature not available in cloud" message. The user sees a friendly
   error instead of a crash.
3. **The embedding dimensions changed** (Ollama nomic-embed = 768, MiniLM = 384).
   If you ever pull existing ChromaDB data into the cloud, you'll need to
   re-embed. For a fresh cloud deploy, this is a non-issue — the new volume
   starts empty.

---

## 0. One-time prerequisites on your Mac

Install the Fly CLI:

```bash
brew install flyctl
```

You'll also need Docker Desktop running locally if you want to test the image
before pushing.

---

## 1. Sign up for Groq (free)

1. Go to https://console.groq.com/
2. Sign in with Google or GitHub.
3. Open https://console.groq.com/keys → **Create API Key** → copy it.
4. Stash it somewhere — you'll paste it into a `fly secrets set` command below.
   The key starts with `gsk_…`.

Groq's free tier is generous: ~30 requests/min on Llama 3.3 70B, more on
smaller models. Plenty for an assignment demo.

---

## 2. Sign up for Neon Postgres (free)

FinEx uses Postgres to store extracted financial statements. Neon's free tier
gives you 500 MB of storage and autoscales to zero when idle.

1. Go to https://console.neon.tech/ and sign in.
2. Create a new project (default region; pick the one closest to Fly's `lhr`,
   e.g. London or Frankfurt).
3. Default database name `neondb` is fine.
4. Click **Connection string** → **Pooled connection** → copy the URI. It
   looks like:

   ```
   postgresql://neondb_owner:abcXYZ@ep-foo-123.eu-central-1.aws.neon.tech/neondb?sslmode=require
   ```

5. Stash it. This becomes your `DATABASE_URL`.

**Initialise the schema once** (Jarvis does this automatically on first FinEx
call, but you can prime it manually):

```bash
psql "$YOUR_NEON_URL" -c "SELECT 1;"   # smoke test the connection
```

---

## 3. Sign up for Fly.io (free for first 3 small VMs)

1. Go to https://fly.io/ → **Sign Up**.
2. Add a payment card (required even on free tier, no charge for free usage).
3. From your Mac terminal:

   ```bash
   fly auth login
   ```

   This opens a browser tab — log in.

---

## 4. First-time launch

From the Jarvis repo root on your Mac:

```bash
cd /Users/akd/Desktop/Jarvis

# Pick a unique app name — what's in fly.toml ("jarvis-cloud") is likely taken.
# Edit fly.toml and change `app = "..."` to something like jarvis-<yourname>.

# Create the persistent volume FIRST — fly.toml references it.
fly volumes create jarvis_data --region lhr --size 3 --yes

# Initial launch (this creates the app on Fly's side without deploying yet).
fly launch --no-deploy --copy-config --name jarvis-<yourname> --region lhr
```

If `fly launch` complains the app name is taken, pick a different one and
update `fly.toml` to match.

---

## 5. Set secrets

These are stored encrypted on Fly's side and injected as env vars at runtime.

```bash
# Required: Groq API key
fly secrets set GROQ_API_KEY=gsk_your_key_here

# Required: Neon Postgres connection string (note the quotes — special chars)
fly secrets set DATABASE_URL='postgresql://...sslmode=require'

# Required if you want anyone with the URL gated by a password:
fly secrets set JARVIS_AUTH_PASSWORD='choose-a-strong-passphrase'
# (Default username is "admin"; override with JARVIS_AUTH_USER=... if you like.)

# Optional: ElevenLabs for TTS playback (skip if you only need text)
fly secrets set ELEVENLABS_API_KEY=sk-...
fly secrets set ELEVENLABS_VOICE_ID=JBFqnCBsd6RMkjVDRZzb

# Optional: Google Calendar / Gmail integration. Not recommended for demo
# — the OAuth callback URL would need updating in Google Cloud Console and
# you'd be exposing your own creds.
```

Check what's set:

```bash
fly secrets list
```

---

## 6. Deploy

```bash
fly deploy
```

This will:

1. Build the Dockerfile locally (or remotely on a Fly builder), ~5–10 minutes
   the first time. Subsequent builds are seconds-fast thanks to layer cache.
2. Push the image to Fly's registry.
3. Spin up a machine with the volume mounted, run health checks, then swap
   traffic over.

When it's done you'll see the hostname:

```
Visit your newly deployed app at https://jarvis-yourname.fly.dev/
```

---

## 7. Post-deploy smoke test

```bash
# Health check (unauthenticated)
curl https://jarvis-yourname.fly.dev/healthz

# Chat (authenticated if you set JARVIS_AUTH_PASSWORD)
curl -u admin:your-password -X POST https://jarvis-yourname.fly.dev/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"hello","history":[]}'

# Tail logs
fly logs
```

Then open the URL in a browser — you'll get a Basic Auth popup, enter
`admin` / `your-password`, and the UI loads.

---

## 8. Iterating

```bash
# Re-deploy after code changes
fly deploy

# Update a single secret without redeploying
fly secrets set GROQ_CHAT_MODEL=llama-3.1-70b-versatile

# Shell into the running container for debugging
fly ssh console

# Scale down to save free credits while not demoing
fly scale count 0
# Scale back up for a demo
fly scale count 1
```

---

## Troubleshooting

**"ImportError: cannot import name 'GroqClient'"** — `sentence-transformers`
failed to install. Check the build log for wheel-build errors. The Dockerfile
has `build-essential` and `gcc` to handle source builds.

**"GROQ_API_KEY is not set"** — you forgot `fly secrets set GROQ_API_KEY=...`.
Run `fly secrets list` to confirm.

**Health check fails after deploy** — pull `fly logs` and look at the startup
banner. The first request waits for the LLM warmup; if that exceeds the 60s
`grace_period` in fly.toml, bump it to `120s`.

**FinEx returns DB errors** — `fly secrets set DATABASE_URL=...` is missing
or the connection string is malformed. The leading `postgresql://` and
trailing `?sslmode=require` both matter.

**Voice endpoint returns 503** — expected. The container has no microphone.
Run Jarvis locally on your Mac for voice input.

**Container restarts in a loop** — usually means the Dockerfile pre-download
step failed (network blip on Hugging Face). Bust the cache with
`fly deploy --no-cache`.

---

## Rolling back

```bash
fly releases                       # list previous deploys
fly releases rollback <version>    # revert to a known-good one
```

---

## Going further

- Add Cloudflare in front of the Fly hostname for caching/WAF.
- Move ChromaDB off the volume onto Pinecone / Weaviate if you scale to >1
  machine.
- Replace HTTP Basic Auth with Cloudflare Access (free up to 50 users) so
  reviewers log in with email instead of a shared password.
- Add a browser-mic flow: WebRTC MediaRecorder → POST `/voice/upload` →
  faster-whisper → Groq → ElevenLabs → audio response. This is what unlocks
  true cloud voice.
