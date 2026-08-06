# Tests

```
python -m pytest              # everything, ~4s, no network
python -m pytest -q --cov=core
```

Nothing in this suite talks to Ollama, Google, macOS or the internet, and
nothing writes to your real data. That's enforced, not just intended.

## What's faked, and what isn't

Only the boundary is faked. The router, planner, critic, registry, executor,
adapters and tools are all the real code.

| Boundary | Fake | Notes |
|---|---|---|
| Ollama | `FakeOllamaClient` | scripted by prompt match; records every call |
| Gmail / Calendar | `FakeGmailService` / `FakeCalendarService` | real API shapes — base64url bodies, multipart payloads, header lists |
| macOS shell | `FakeShell` | intercepts `subprocess.run`; models volume, brightness, dark mode, battery |
| SQLite / ChromaDB | real, in a temp dir | redirected by env vars set in `conftest.py` **at import time** |
| Network | `install_network_block` | autouse; any outbound socket raises |

The Gmail fake stores messages the way Gmail actually returns them rather than
as tidy dicts. The interesting bugs in `gmail_agent.py` are in the parsing —
base64url decoding, walking `payload.parts`, RFC-2822 address splitting — and a
convenient mock skips exactly that code.

The network block is a backstop, not the mechanism. Its job is to turn "this
test quietly hit the real ESPN API and passed" into a named failure.
`test_weather_tool_through_the_registry_is_blocked_from_the_network` asserts it
works: `WeatherTool` has no fake, so it must fail rather than reach Open-Meteo.

## The `jarvis` fixture

A real `JarvisOrchestrator` with all of the above wired in:

```python
async def test_something(jarvis):
    result = await jarvis.tools.execute("gmail", "get_inbox", {"max_results": 5})
    assert result.success
    assert jarvis.fake_gmail.sent == []
```

Attached for assertions: `jarvis.fake_llm`, `.fake_shell`, `.fake_gmail`,
`.fake_calendar`.

It skips rather than fails when ChromaDB or aiohttp aren't installed, so the
fast unit tests still run in a bare environment.

## Writing a test

Script the model by matching on prompt content; register specific rules before
general ones, since the first match wins:

```python
from tests.fakes import plan, route, subtask

fake_llm.when("classify", route("weather_query", "weather"))
fake_llm.when("decompose", plan(subtask("s1", "weather", "get_current")))
```

Then assert on what was *asked*, not just what came back — that's usually the
interesting part and it's invisible from the final answer alone:

```python
call = jarvis.fake_llm.assert_asked_about("High Wycombe")
assert "memory" in call.system
```

## Two things worth knowing

**Env vars are set at import time in `conftest.py`, before any app import.**
`config/settings.py` computes `DATA_DIR`, `SQLITE_PATH` and `CHROMA_DIR` at
module scope. Setting them inside a fixture would be too late and a test run
would write into the live reminder and evaluation databases.

**`pretend_macos` patches `is_mac()` in two namespaces.** `MacControlTool`
imports the name directly *and* checks it in `__init__`, where it replaces
every coroutine method with a "macOS only" stub. Without the patch the mac tool
is inert anywhere but a Mac, so its output parsing — the regexes, and therefore
the bugs — would only ever be exercised there.
