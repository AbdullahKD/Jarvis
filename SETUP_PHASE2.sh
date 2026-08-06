#!/usr/bin/env bash
# Jarvis v2 — baseline + Phase 2 foundation commit.
# Run this from /Users/akd/Desktop/Jarvis
#
# Step 0 clears a stale lock file my session left behind (my session can create
# files in your repo but not delete them, so git's own cleanup failed).

set -euo pipefail
cd "$(dirname "$0")"

echo "── 0. clear the stale git lock ─────────────────────────────────────────"
rm -f .git/index.lock

echo "── 1. baseline branch ──────────────────────────────────────────────────"
git checkout -b jarvis-v2

echo "── 2. commit your existing work as the baseline ────────────────────────"
# Tracked modifications
git add -u
# Untracked source that was never committed (real code, not artefacts):
git add config/logging_config.py config/profile.py tools/brain.py \
        tools/query_parser.py voice/selftest.py \
        tests/sports_probe.py tests/test_jarvis_health.py tests/test_router_offline.py \
        ui/atlas.html ui/brief.html ui/forge.html ui/health.html ui/jams.html \
        ui/recall.html ui/sentinel.html ui/vault.html
git commit -m "baseline: work in progress prior to v2 refactor

Snapshot of the working tree as of the Phase 1 audit. Includes previously
untracked source (logging config, profile loader, brain/atlas indexer, voice
selftest, the health/router test scripts and eight UI pages) so the v2 work
starts from a complete, committed baseline."

echo "── 3. keep build artefacts out of the tree from now on ─────────────────"
cat >> .gitignore <<'GITIGNORE'

# Editor / tooling backups — use git, not .bak files
*.bak
*.bak[0-9]
_to_delete/
GITIGNORE
git add .gitignore
git commit -m "chore: gitignore .bak backups and the _to_delete staging folder"

echo "── 4. Phase 2 foundation ───────────────────────────────────────────────"
git add core/ tests/conftest.py tests/test_tool_interface.py tests/test_registry.py \
        pytest.ini requirements-dev.txt MISSING_DEPS.md
git commit -m "feat(core): common Tool interface and registry

Adds the contract every Jarvis capability will implement:

  core/tool.py      Tool / BaseTool with name, description, actions,
                    execute() and health_check(). Schema validation,
                    timeout enforcement, latency measurement and
                    exception-to-typed-error mapping live in the base
                    class so the 14 adapters stay thin.

  core/registry.py  ToolRegistry replacing the 240-line if/elif chain in
                    orchestrator._dispatch, with tool and action aliases,
                    bounded-concurrency health checks, and catalogue
                    generation for the Planner prompt.

Action input schemas are JSON Schema, matching what MCP tools/list expects,
so the Phase 3 conversion needs no translation layer.

48 tests, all passing offline (no Ollama, Google, network or macOS).
"

echo
echo "── done ────────────────────────────────────────────────────────────────"
git log --oneline -4
echo
echo "Now run the tests:"
echo "  pip install -r requirements-dev.txt"
echo "  python -m pytest tests/test_tool_interface.py tests/test_registry.py -v"
