"""
Jarvis startup diagnostic.
Run from the project root with the venv activated:

    source venv/bin/activate
    python diagnose.py

Reports the three things that cause `python server.py` to look hung:
  1. Ollama not running / model not pulled  →  warmup task spins forever
  2. Port 8000 already in use               →  uvicorn fails silently
  3. `log_level="warning"` suppresses the "Uvicorn running on..." banner
     →  the server IS up, you just can't tell from the terminal
"""

from __future__ import annotations

import os
import socket
import sys
import urllib.request
import urllib.error
from pathlib import Path

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
DIM = "\033[2m"
RESET = "\033[0m"


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def bad(msg: str) -> None:
    print(f"  {RED}✗{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}!{RESET} {msg}")


def section(title: str) -> None:
    print(f"\n{BLUE}── {title} {'─' * (70 - len(title))}{RESET}")


def load_env() -> dict[str, str]:
    """Parse .env without depending on python-dotenv."""
    env_path = Path(__file__).parent / ".env"
    env: dict[str, str] = {}
    if not env_path.exists():
        return env
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def check_python() -> None:
    section("Python")
    print(f"  Version: {sys.version.split()[0]}")
    print(f"  Executable: {sys.executable}")
    in_venv = sys.prefix != sys.base_prefix
    (ok if in_venv else bad)(
        "Running inside venv" if in_venv else
        "NOT running inside venv — activate with: source venv/bin/activate"
    )


def check_port(port: int) -> None:
    section(f"Port {port}")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1.0)
    try:
        s.bind(("0.0.0.0", port))
        s.close()
        ok(f"Port {port} is free — uvicorn can bind it")
    except OSError:
        bad(f"Port {port} is already in use. Another Jarvis (or other app) is on it.")
        print(f"     {DIM}Find it:  lsof -iTCP:{port} -sTCP:LISTEN{RESET}")
        print(f"     {DIM}Kill it:  kill $(lsof -tiTCP:{port} -sTCP:LISTEN){RESET}")


def check_ollama(base_url: str, chat_model: str) -> None:
    section("Ollama")
    print(f"  Base URL:    {base_url}")
    print(f"  Chat model:  {chat_model}")

    # Reachability
    try:
        req = urllib.request.Request(f"{base_url.rstrip('/')}/api/tags")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = resp.read().decode()
    except (urllib.error.URLError, socket.timeout, ConnectionError) as e:
        bad(f"Cannot reach Ollama at {base_url}  ({e})")
        print(f"     {DIM}Start it:  ollama serve   {RESET}{DIM}(or open the Ollama app){RESET}")
        return
    ok("Ollama is reachable")

    # Model pulled?
    import json
    try:
        tags = json.loads(data)
        models = [m.get("name", "") for m in tags.get("models", [])]
    except Exception:
        models = []
    if not models:
        warn("Ollama is running but has no models pulled")
        print(f"     {DIM}Pull yours:  ollama pull {chat_model}{RESET}")
        return

    print(f"  Pulled models: {', '.join(models)}")
    if chat_model in models or any(m.split(":")[0] == chat_model.split(":")[0] for m in models):
        ok(f"Chat model '{chat_model}' is available")
    else:
        bad(f"Chat model '{chat_model}' is NOT pulled")
        print(f"     {DIM}Pull it:  ollama pull {chat_model}{RESET}")


def check_imports() -> None:
    section("Imports")
    failures = []
    for mod in ("fastapi", "uvicorn", "aiohttp", "chromadb",
                "pydantic", "google.auth", "elevenlabs"):
        try:
            __import__(mod)
            ok(f"import {mod}")
        except Exception as e:
            bad(f"import {mod}  →  {type(e).__name__}: {e}")
            failures.append(mod)
    if failures:
        print(f"     {DIM}Fix:  pip install -r requirements.txt{RESET}")


def check_silent_uvicorn() -> None:
    section("uvicorn log level (very common gotcha)")
    server_py = Path(__file__).parent / "server.py"
    if not server_py.exists():
        return
    text = server_py.read_text()
    if 'log_level="warning"' in text or "log_level='warning'" in text:
        warn("server.py sets uvicorn log_level=\"warning\"")
        print(f"     {DIM}→ uvicorn does not print its \"Uvicorn running on...\" banner.{RESET}")
        print(f"     {DIM}→ The terminal will look frozen even when the server is up.{RESET}")
        print(f"     {DIM}→ Try opening http://localhost:8000 in your browser to confirm.{RESET}")
        print(f"     {DIM}→ Or change log_level to \"info\" to see startup logs.{RESET}")
    else:
        ok("log_level is not set to \"warning\"")


def main() -> None:
    print(f"\n{BLUE}━━━ Jarvis startup diagnostic ━━━{RESET}")
    env = load_env()
    # Inject env so anything we import later sees it
    for k, v in env.items():
        os.environ.setdefault(k, v)

    check_python()
    check_imports()
    check_port(int(env.get("UI_PORT", "8000")))
    check_ollama(
        env.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        env.get("OLLAMA_CHAT_MODEL", "llama3"),
    )
    check_silent_uvicorn()
    print(f"\n{DIM}Done. Fix anything red above, then run: python server.py{RESET}\n")


if __name__ == "__main__":
    main()
