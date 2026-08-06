"""Test doubles for everything Jarvis talks to.

The point of this package is that a test can exercise the *real* orchestrator,
the *real* agents and the *real* tools while nothing leaves the process. What
gets faked is only the boundary: the model, Google's APIs, the macOS shell, and
the two on-disk stores.

Import from here rather than from the submodules — the layout may move, the
names shouldn't::

    from tests.fakes import FakeOllamaClient, FakeGmailService, FakeShell, plan, subtask
"""

from tests.fakes.google import (
    FakeCalendarService,
    FakeEvent,
    FakeGmailService,
    FakeMessage,
)
from tests.fakes.llm import (
    EMBED_DIM,
    FakeOllamaClient,
    LLMCall,
    plan,
    route,
    subtask,
)
from tests.fakes.system import (
    FakeShell,
    NetworkAccessError,
    ShellResult,
    install_network_block,
)

__all__ = [
    "FakeOllamaClient", "LLMCall", "route", "plan", "subtask", "EMBED_DIM",
    "FakeGmailService", "FakeCalendarService", "FakeMessage", "FakeEvent",
    "FakeShell", "ShellResult", "install_network_block", "NetworkAccessError",
]
