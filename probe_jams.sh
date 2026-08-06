#!/usr/bin/env bash
# Ask Jarvis what it can actually reach in n8n.
echo "── Is n8n up? ─────────────────────────────────────────"
curl -s -o /dev/null -w "  http://localhost:5678  ->  HTTP %{http_code}\n" http://localhost:5678/ || echo "  unreachable"
echo
echo "── What Jarvis sees ───────────────────────────────────"
curl -s http://localhost:8000/jams/workflows | python3 -m json.tool 2>/dev/null || echo "  (Jarvis not running?)"
echo
echo "── Raw probe of each candidate webhook ────────────────"
for p in discovery responses actions hud-data apply-job; do
  code=$(curl -s -o /tmp/_b -w "%{http_code}" "http://localhost:5678/webhook/$p")
  printf "  %-12s GET -> %s  %s\n" "$p" "$code" "$(head -c 90 /tmp/_b | tr -d '\n')"
done
