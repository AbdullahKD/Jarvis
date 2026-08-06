# Dependencies imported but absent from requirements.txt

Found by reconciling every third-party import against the requirements files.
Add these to `requirements.txt` — I've left the file untouched because you have
uncommitted changes in it.

```
# ── JSON Schema validation (Phase 2 tool interface) ────────────────────────
jsonschema>=4.21.0

# ── PDF text extraction ────────────────────────────────────────────────────
# tools/document.py imports PyPDF2, which is deprecated upstream. `pypdf` is
# the maintained successor with a near-identical API — recommend switching the
# import rather than pinning the dead package.
pypdf>=4.0.0

# ── Local embedding fallback (config/groq_client.py) ───────────────────────
# Only needed if you keep the Groq backend path. ~2GB with torch — consider
# making it a documented optional extra instead.
# sentence-transformers>=2.5.0
```

`pydantic` and `starlette` are also imported directly by `server.py` but arrive
transitively via FastAPI. Worth declaring explicitly so a FastAPI major bump
can't silently change them under you.
