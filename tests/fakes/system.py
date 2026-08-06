"""Fakes for the machine itself: macOS shell-outs, the network, and the stores.

Three separate concerns, one module because tests almost always want them
together.

**macOS.** ``MacControlTool`` shells out to ``osascript``, ``pmset``,
``networksetup`` and friends. On CI — and in any container — those either don't
exist or return something meaningless, and on a developer's own Mac they'd
change the real volume. ``FakeShell`` intercepts ``subprocess.run`` and answers
from a table keyed on the command, so the tool's own parsing of that output is
what gets tested.

**Network.** ``block_network`` patches ``socket.socket.connect`` to raise. It's
a backstop, not the primary mechanism: the point isn't to stub HTTP, it's to
make an *accidentally un-faked* dependency fail loudly and immediately instead
of silently hitting the real Open-Meteo, ESPN or Gmail. A test that hangs for
30 seconds and then passes on a cached value is worse than one that fails.

**Stores.** SQLite and ChromaDB both want real paths. ``temp_stores`` points
them at a tmpdir so tests never touch ~/.jarvis or the live jarvis.db, which
holds real reminders and evaluation history.
"""

from __future__ import annotations

import socket
import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Union


# ── macOS shell ─────────────────────────────────────────────────────────────


@dataclass
class ShellResult:
    """Stands in for subprocess.CompletedProcess."""

    stdout: str = ""
    stderr: str = ""
    returncode: int = 0


Matcher = Union[str, Callable[[Sequence[str]], bool]]


@dataclass
class MacState:
    """The bit of the machine the tool can observe.

    A purely static fake isn't good enough here: several MacControlTool methods
    write a setting and then read it back to confirm it took, and a fake that
    always returns 45% makes every set_volume look like the Bluetooth-device
    failure case. Modelling the state is what lets the tool's real
    verify-after-write logic run.
    """

    volume: int = 45
    brightness: float = 0.65
    dark_mode: bool = True
    clipboard: str = "clipboard contents"
    battery_percent: int = 82
    charging: bool = False


class FakeShell:
    """Intercepts subprocess.run and answers from a rule table.

    Rules are matched against the joined command line, most-recently-registered
    first, so a test can override a default without clearing it. Anything not
    matched falls through to a small stateful macOS model (see MacState).
    """

    #: Defaults chosen to mirror what these commands actually print on macOS,
    #: because the tool parses them with regexes and string splits.
    DEFAULTS: Dict[str, ShellResult] = {
        "output volume": ShellResult(stdout="45\n"),
        "set volume": ShellResult(),
        "brightness -l": ShellResult(stdout="display 0: brightness 0.650000\n"),
        "pmset -g batt": ShellResult(
            stdout="Now drawing from 'Battery Power'\n"
                   " -InternalBattery-0 (id=12345)\t82%; discharging; 4:31 remaining present: true\n"),
        "top -l 1": ShellResult(stdout="12.5%\n"),
        "vm_stat": ShellResult(stdout="1234567\n"),
        "df -h": ShellResult(
            stdout="Filesystem  Size  Used Avail Capacity  Mounted on\n"
                   "/dev/disk3s5  926Gi  412Gi  501Gi    46%    /\n"),
        "networksetup -listallhardwareports": ShellResult(
            stdout="Hardware Port: Wi-Fi\nDevice: en0\n"
                   "Ethernet Address: aa:bb:cc:dd:ee:ff\n\n"),
        "pbpaste": ShellResult(stdout="clipboard contents\n"),
        "pbcopy": ShellResult(),
        "AppleInterfaceStyle": ShellResult(stdout="Dark\n"),
        "screencapture": ShellResult(),
        "osascript": ShellResult(stdout="ok\n"),
    }

    def __init__(self, *, defaults: bool = True, state: Optional[MacState] = None):
        self.rules: List[tuple[Matcher, ShellResult]] = []
        self.calls: List[List[str]] = []
        self._use_defaults = defaults
        self.state = state or MacState()

    def when(self, matcher: Matcher, result: ShellResult) -> "FakeShell":
        self.rules.insert(0, (matcher, result))
        return self

    def fails(self, matcher: Matcher, *, stderr: str = "error",
              returncode: int = 1) -> "FakeShell":
        return self.when(matcher, ShellResult(stderr=stderr, returncode=returncode))

    def run(self, args, **kwargs) -> ShellResult:
        cmd = args if isinstance(args, (list, tuple)) else [str(args)]
        cmd = [str(a) for a in cmd]
        self.calls.append(cmd)
        joined = " ".join(cmd)

        for matcher, result in self.rules:
            if callable(matcher):
                if matcher(cmd):
                    return result
            elif matcher in joined:
                return result

        if self._use_defaults:
            stateful = self._stateful(joined)
            if stateful is not None:
                return stateful
            for needle, result in self.DEFAULTS.items():
                if needle in joined:
                    return result
        return ShellResult(stdout="", stderr="", returncode=0)

    def _stateful(self, joined: str) -> Optional[ShellResult]:
        """Commands that read or write the modelled state.

        Ordered so writes are matched before reads — "set volume output volume
        30" contains "output volume", and matching the read first would make
        every write a silent no-op.
        """
        st = self.state

        # ── writes ──────────────────────────────────────────────────────────
        if "set volume output volume" in joined:
            digits = "".join(c for c in joined.split("set volume output volume")[1]
                             if c.isdigit())
            if digits:
                st.volume = max(0, min(100, int(digits)))
            return ShellResult()
        if joined.startswith("brightness ") and "-l" not in joined:
            try:
                st.brightness = float(joined.split()[-1])
            except ValueError:
                pass
            return ShellResult()
        if "to not dark mode" in joined or "dark mode is not dark mode" in joined:
            st.dark_mode = not st.dark_mode
            return ShellResult(stdout="true" if st.dark_mode else "false")
        if "pbcopy" in joined:
            return ShellResult()

        # ── reads ───────────────────────────────────────────────────────────
        if "output volume of" in joined or "output volume" in joined:
            return ShellResult(stdout=f"{st.volume}\n")
        if "brightness -l" in joined:
            return ShellResult(stdout=f"display 0: brightness {st.brightness:.6f}\n")
        if "AppleInterfaceStyle" in joined:
            return (ShellResult(stdout="Dark\n") if st.dark_mode
                    else ShellResult(stderr="does not exist", returncode=1))
        if "pbpaste" in joined:
            return ShellResult(stdout=st.clipboard)
        if "pmset -g batt" in joined:
            mode = "AC Power" if st.charging else "Battery Power"
            action = "charging" if st.charging else "discharging"
            return ShellResult(
                stdout=f"Now drawing from '{mode}'\n"
                       f" -InternalBattery-0 (id=12345)\t{st.battery_percent}%; "
                       f"{action}; 4:31 remaining present: true\n")
        return None

    # ── Assertions ──────────────────────────────────────────────────────────

    def ran(self, needle: str) -> bool:
        return any(needle in " ".join(c) for c in self.calls)

    def assert_ran(self, needle: str) -> List[str]:
        for c in self.calls:
            if needle in " ".join(c):
                return c
        raise AssertionError(
            f"no command matched {needle!r}; ran: {[' '.join(c)[:70] for c in self.calls]}")

    def assert_never_ran(self, needle: str) -> None:
        assert not self.ran(needle), (
            f"command matching {needle!r} was executed: "
            f"{[' '.join(c) for c in self.calls if needle in ' '.join(c)]}")

    @property
    def call_count(self) -> int:
        return len(self.calls)


# ── Network guard ───────────────────────────────────────────────────────────


class NetworkAccessError(RuntimeError):
    """Raised when a test tries to open a real socket."""


_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex


def install_network_block(allow_local: bool = True) -> Callable[[], None]:
    """Make outbound sockets raise. Returns an undo callable.

    Local connections stay allowed by default: ChromaDB's embedded server and
    SQLite's WAL machinery both use loopback in some configurations, and
    blocking those would fail tests for a reason unrelated to what they check.
    """

    def guard(self, address, *args, **kwargs):
        host = address[0] if isinstance(address, tuple) else str(address)
        if allow_local and str(host) in ("127.0.0.1", "::1", "localhost", ""):
            return _real_connect(self, address, *args, **kwargs)
        raise NetworkAccessError(
            f"test tried to reach {host!r} — a real service is not faked. "
            f"Add a fake rather than letting the suite depend on the network."
        )

    def guard_ex(self, address, *args, **kwargs):
        try:
            guard(self, address, *args, **kwargs)
            return 0
        except NetworkAccessError:
            raise

    socket.socket.connect = guard          # type: ignore[method-assign]
    socket.socket.connect_ex = guard_ex    # type: ignore[method-assign]

    def undo() -> None:
        socket.socket.connect = _real_connect          # type: ignore[method-assign]
        socket.socket.connect_ex = _real_connect_ex    # type: ignore[method-assign]

    return undo


__all__ = [
    "FakeShell", "ShellResult", "install_network_block", "NetworkAccessError",
]
