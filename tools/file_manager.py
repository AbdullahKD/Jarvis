"""
File Manager Tool
Gives Jarvis full access to Desktop, Documents, and Downloads.

Operations:
  - list_directory   — browse a folder
  - search           — find files by name or content
  - read_file        — read text files (txt, md, py, js, etc.)
  - create_file      — create a new file with content
  - create_folder    — create a new folder
  - move             — move/rename a file or folder  [requires approval]
  - rename           — rename a file or folder       [requires approval]
  - delete           — delete a file or folder       [requires approval]
  - get_info         — size, dates, type for a path

Destructive operations (move, rename, delete) return a PendingFileOp
that must be confirmed by the user before execution.
"""

from __future__ import annotations

import mimetypes
import os
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Allowed roots ────────────────────────────────────────────────────────────
HOME = Path.home()
ALLOWED_ROOTS = {
    "desktop":   HOME / "Desktop",
    "documents": HOME / "Documents",
    "downloads": HOME / "Downloads",
}

# File types we can read as text
TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".py", ".js", ".ts", ".jsx", ".tsx",
    ".html", ".css", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".sh", ".bash", ".zsh", ".env", ".csv", ".xml", ".rst", ".log",
    ".swift", ".kt", ".java", ".c", ".cpp", ".h", ".go", ".rb", ".php",
    ".r", ".sql", ".graphql", ".tf", ".dockerfile",
}

MAX_READ_BYTES = 50_000   # 50 KB max for file reads
MAX_SEARCH_RESULTS = 20

# Plural / colloquial → canonical extension set. Used by `search()` so
# "find all PDFs" / "list spreadsheets" actually filter by extension
# instead of searching for a file literally named "PDFs".
EXTENSION_ALIASES = {
    "pdf": [".pdf"], "pdfs": [".pdf"],
    "doc": [".doc", ".docx"], "docs": [".doc", ".docx"],
    "word": [".doc", ".docx"], "word docs": [".doc", ".docx"],
    "spreadsheet": [".xls", ".xlsx", ".csv", ".numbers"],
    "spreadsheets": [".xls", ".xlsx", ".csv", ".numbers"],
    "excel": [".xls", ".xlsx"], "csv": [".csv"], "csvs": [".csv"],
    "deck": [".pptx", ".key"], "decks": [".pptx", ".key"],
    "slides": [".pptx", ".key"], "presentation": [".pptx", ".key"],
    "presentations": [".pptx", ".key"],
    "keynote": [".key"], "powerpoint": [".pptx"],
    "image": [".png", ".jpg", ".jpeg", ".gif", ".heic", ".webp", ".tiff", ".bmp"],
    "images": [".png", ".jpg", ".jpeg", ".gif", ".heic", ".webp", ".tiff", ".bmp"],
    "photo": [".png", ".jpg", ".jpeg", ".heic"],
    "photos": [".png", ".jpg", ".jpeg", ".heic"],
    "picture": [".png", ".jpg", ".jpeg", ".heic", ".webp"],
    "pictures": [".png", ".jpg", ".jpeg", ".heic", ".webp"],
    "screenshot": [".png", ".jpg"], "screenshots": [".png", ".jpg"],
    "video": [".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"],
    "videos": [".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"],
    "movie": [".mp4", ".mov", ".mkv", ".avi"],
    "movies": [".mp4", ".mov", ".mkv", ".avi"],
    "audio": [".mp3", ".wav", ".m4a", ".aac", ".flac"],
    "music": [".mp3", ".wav", ".m4a", ".aac", ".flac"],
    "songs": [".mp3", ".wav", ".m4a", ".aac", ".flac"],
    "podcast": [".mp3", ".m4a"], "podcasts": [".mp3", ".m4a"],
    "note": [".txt", ".md", ".markdown"], "notes": [".txt", ".md", ".markdown"],
    "markdown": [".md", ".markdown"], "text": [".txt"],
    "python": [".py"], "scripts": [".py", ".sh", ".js", ".ts"],
    "javascript": [".js", ".jsx", ".mjs"], "typescript": [".ts", ".tsx"],
    "html": [".html", ".htm"], "css": [".css"],
    "zip": [".zip"], "archive": [".zip", ".tar", ".gz", ".7z"],
    "archives": [".zip", ".tar", ".gz", ".7z"],
}

# Max directory recursion depth — prevents `rglob` from disappearing into
# deep node_modules / venv trees during a search.
MAX_SEARCH_DEPTH = 5


# ── Pending operation (approval flow) ────────────────────────────────────────

@dataclass
class PendingFileOp:
    operation: str          # "delete", "move", "rename"
    source: Path
    destination: Optional[Path] = None
    confirmed: bool = False

    def summary(self) -> str:
        """Human-readable description for the confirmation card."""
        if self.operation == "delete":
            kind = "folder" if self.source.is_dir() else "file"
            return (
                f"⚠️  Delete {kind}: {self._short(self.source)}\n"
                f"   Full path: {self.source}\n"
                f"   This cannot be undone. Type **confirm** to proceed or **cancel**."
            )
        elif self.operation == "move":
            return (
                f"Move: {self._short(self.source)}\n"
                f"  → {self._short(self.destination)}\n"
                f"Type **confirm** to proceed or **cancel**."
            )
        elif self.operation == "rename":
            return (
                f"Rename: {self.source.name}\n"
                f"     → {self.destination.name}\n"
                f"  In: {self.source.parent}\n"
                f"Type **confirm** to proceed or **cancel**."
            )
        return f"Pending: {self.operation}"

    def _short(self, p: Path) -> str:
        try:
            return str(p.relative_to(HOME))
        except ValueError:
            return str(p)


# ── Main tool ─────────────────────────────────────────────────────────────────

class FileManagerTool:
    """
    Full file system access for Desktop, Documents, Downloads.
    Destructive operations return a PendingFileOp for user confirmation.
    """

    def __init__(self):
        # Ensure all root dirs exist
        for root in ALLOWED_ROOTS.values():
            root.mkdir(parents=True, exist_ok=True)
        print("📁 FileManagerTool ready — roots: Desktop, Documents, Downloads")

    # ── Path resolution ───────────────────────────────────────────────────────

    def resolve(self, path_str: str) -> Optional[Path]:
        """
        Resolve a user-supplied path string to an absolute Path.
        Handles: absolute paths, ~/... paths, root aliases like "desktop"/"documents".
        Returns None if the path is outside allowed roots.
        """
        # First: check if it's a root alias (desktop, documents, downloads)
        low = path_str.lower().strip("/ ")
        if low in ALLOWED_ROOTS:
            return ALLOWED_ROOTS[low]

        p = Path(path_str).expanduser()
        if p.is_absolute():
            return p if self._is_allowed(p) else None

        # Relative path — try resolving under each allowed root
        for root in ALLOWED_ROOTS.values():
            candidate = root / p
            if candidate.exists() and self._is_allowed(candidate):
                return candidate

        # Not found under any root but path is relative — return under
        # Desktop as default (caller will check .exists() themselves).
        # The duplicate `return` below the previous line was dead code.
        candidate = ALLOWED_ROOTS["desktop"] / p
        return candidate if self._is_allowed(candidate) else None

    def _is_allowed(self, p: Path) -> bool:
        """Check path is within one of the allowed roots."""
        try:
            p_resolved = p.resolve()
            return any(
                p_resolved == r.resolve() or p_resolved.is_relative_to(r.resolve())
                for r in ALLOWED_ROOTS.values()
            )
        except Exception:
            return False

    def _detect_root(self, query: str) -> Optional[Path]:
        """Detect which root folder a natural language query refers to."""
        q = query.lower()
        if any(w in q for w in ["desktop"]):
            return ALLOWED_ROOTS["desktop"]
        if any(w in q for w in ["document", "docs"]):
            return ALLOWED_ROOTS["documents"]
        if any(w in q for w in ["download"]):
            return ALLOWED_ROOTS["downloads"]
        return None

    # ── List directory ────────────────────────────────────────────────────────

    def list_directory(
        self,
        path_str: str = "desktop",
        *,
        include_hidden: bool = False,
        sort_by: str = "name",
        reverse: bool = False,
    ) -> Dict[str, Any]:
        """
        List contents of a folder.

        Args:
            path_str: Folder path or alias (desktop, documents, downloads).
            include_hidden: If True, include dotfiles (.env, .gitignore, etc.).
            sort_by: "name" (default), "size", "modified", "type".
            reverse: Flip the sort order. Combined with sort_by="modified"
                     this gives "newest first".
        """
        p = self.resolve(path_str)
        if p is None:
            return {"success": False, "error": f"Path not found or not allowed: {path_str}"}
        if not p.exists():
            return {"success": False, "error": f"Path does not exist: {path_str}"}
        if not p.is_dir():
            return {"success": False, "error": f"Not a directory: {path_str}"}

        items: List[Dict[str, Any]] = []
        try:
            for entry in p.iterdir():
                name = entry.name
                # Skip iCloud placeholder files always — they're noise.
                if name.endswith(".icloud"):
                    continue
                # Skip hidden files by default
                if name.startswith(".") and not include_hidden:
                    continue
                try:
                    stat = entry.stat()
                except (FileNotFoundError, OSError):
                    continue
                items.append({
                    "name": name,
                    "type": "folder" if entry.is_dir() else "file",
                    "size": stat.st_size if entry.is_file() else None,
                    "size_human": self._fmt_size(stat.st_size) if entry.is_file() else None,
                    "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%d %b %Y %H:%M"),
                    "modified_ts": stat.st_mtime,
                    "extension": entry.suffix.lower() if entry.is_file() else None,
                    "hidden": name.startswith("."),
                })
        except PermissionError:
            return {"success": False, "error": f"Permission denied: {path_str}"}

        # Sort
        key_funcs = {
            "name":     lambda i: (i["type"] == "file", i["name"].lower()),
            "size":     lambda i: (i.get("size") or 0),
            "modified": lambda i: i.get("modified_ts", 0.0),
            "type":     lambda i: (i["type"], i["name"].lower()),
        }
        sort_key = key_funcs.get(sort_by, key_funcs["name"])
        items.sort(key=sort_key, reverse=reverse)

        return {
            "success": True,
            "path": str(p),
            "display_path": self._display_path(p),
            "items": items,
            "count": len(items),
            "folders": sum(1 for i in items if i["type"] == "folder"),
            "files": sum(1 for i in items if i["type"] == "file"),
            "hidden_count": sum(1 for i in items if i.get("hidden")),
            "sort_by": sort_by,
            "reverse": reverse,
            "include_hidden": include_hidden,
        }

    def format_listing(self, data: Dict[str, Any]) -> str:
        """Text-mode listing — chat-bubble friendly with emoji icons."""
        if not data.get("success"):
            return data.get("error", "Could not list directory.")
        items = data.get("items", [])
        dp = data.get("display_path", "")
        folders = data["folders"]
        files = data["files"]

        header_bits = []
        if folders:
            header_bits.append(f"{folders} folder{'s' if folders != 1 else ''}")
        if files:
            header_bits.append(f"{files} file{'s' if files != 1 else ''}")
        if data.get("include_hidden") and data.get("hidden_count"):
            header_bits.append(f"{data['hidden_count']} hidden")
        header = ", ".join(header_bits) or "empty"

        lines = [f"**{dp}** — {header}", ""]
        if not items:
            lines.append("  (nothing here yet)")
            return "\n".join(lines)

        # Folders first, then files (when sort is by name)
        for item in items:
            icon = "📁" if item["type"] == "folder" else self._file_icon(item.get("extension", ""))
            size = f"  {item['size_human']}" if item.get("size_human") else ""
            hidden_tag = " (hidden)" if item.get("hidden") else ""
            lines.append(f"{icon} {item['name']}{size}  — {item['modified']}{hidden_tag}")
        return "\n".join(lines)

    def format_listing_voice(self, data: Dict[str, Any]) -> str:
        """
        Voice-mode listing — one short sentence summarising the folder.
        Used when the orchestrator is in voice_mode and TTS will read this.
        """
        if not data.get("success"):
            return data.get("error", "Could not list that folder.")
        folders = data.get("folders", 0)
        files = data.get("files", 0)
        dp = data.get("display_path", "your folder")
        if folders == 0 and files == 0:
            return f"{dp} is empty."
        bits = []
        if folders:
            bits.append(f"{folders} folder{'s' if folders != 1 else ''}")
        if files:
            bits.append(f"{files} file{'s' if files != 1 else ''}")
        summary = " and ".join(bits)
        # Mention the top 3 names if there are few enough
        items = data.get("items", [])
        if len(items) <= 4:
            names = ", ".join(i["name"] for i in items)
            return f"{dp} has {summary}: {names}."
        first_three = ", ".join(i["name"] for i in items[:3])
        return f"{dp} has {summary}, including {first_three}."

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        location: str = "all",
        content_search: bool = False,
    ) -> Dict[str, Any]:
        """
        Search for files/folders.

        - If `query` matches an EXTENSION_ALIASES entry (e.g. "pdfs",
          "images", "spreadsheets", "videos"), filter by extension —
          NOT by literal filename.
        - Otherwise, match by substring in filename.
        - With `content_search=True`, also grep text files.
        - Always skips iCloud placeholder files (*.icloud), the macOS
          `.DS_Store` junk, and recursion past MAX_SEARCH_DEPTH so a
          search through a heavy `~/Documents` doesn't take 30s.
        """
        q = query.lower().strip()
        roots = (
            [ALLOWED_ROOTS[location]] if location in ALLOWED_ROOTS
            else list(ALLOWED_ROOTS.values())
        )

        # Detect extension intent. "all my pdfs" → extensions = [".pdf"]
        ext_filter: Optional[List[str]] = None
        for alias, exts in EXTENSION_ALIASES.items():
            # Match alias as a whole word, not a substring (so "scripts"
            # doesn't match "scriptsystem"). Use word boundaries.
            import re as _re
            if _re.search(rf"\b{_re.escape(alias)}\b", q):
                ext_filter = exts
                break

        results: List[Dict[str, Any]] = []

        def _walk(base: Path, depth: int):
            if len(results) >= MAX_SEARCH_RESULTS or depth > MAX_SEARCH_DEPTH:
                return
            try:
                for entry in base.iterdir():
                    if len(results) >= MAX_SEARCH_RESULTS:
                        return
                    name = entry.name
                    # Skip noise
                    if name.startswith(".") or name.endswith(".icloud"):
                        continue
                    # Skip heavy dev / cache dirs that are never the answer
                    if entry.is_dir() and name in (
                        "node_modules", ".git", "venv", ".venv", "__pycache__",
                        ".cache", "Pods", "DerivedData", ".next", "dist", "build",
                    ):
                        continue

                    matched = False
                    if ext_filter is not None:
                        # Extension intent — only return matching files
                        if entry.is_file() and entry.suffix.lower() in ext_filter:
                            results.append(self._file_info(entry))
                            matched = True
                    else:
                        # Free-text intent — filename substring match
                        if q in name.lower():
                            results.append(self._file_info(entry))
                            matched = True
                        elif content_search and entry.is_file() and entry.suffix.lower() in TEXT_EXTENSIONS:
                            try:
                                text = entry.read_text(errors="ignore")[:10_000]
                                if q in text.lower():
                                    results.append({**self._file_info(entry), "content_match": True})
                                    matched = True
                            except Exception:
                                pass

                    if entry.is_dir():
                        _walk(entry, depth + 1)
            except (PermissionError, OSError):
                return

        for root in roots:
            if not root.exists():
                continue
            _walk(root, 0)

        # Sort: folders first (alphabetically), then files (by modified desc
        # when extension filter is in play — newest pictures first, etc.)
        if ext_filter is not None:
            results.sort(
                key=lambda r: -(Path(r["path"]).stat().st_mtime if Path(r["path"]).exists() else 0)
            )
        else:
            results.sort(key=lambda r: (r["type"] == "file", r["name"].lower()))

        return {
            "success": True,
            "query": query,
            "extension_filter": ext_filter,
            "results": results[:MAX_SEARCH_RESULTS],
            "count": min(len(results), MAX_SEARCH_RESULTS),
        }

    def format_search(self, data: Dict[str, Any]) -> str:
        if not data.get("success"):
            return data.get("error", "Search failed.")
        results = data.get("results", [])
        query = data.get("query", "")
        ext_filter = data.get("extension_filter")

        if not results:
            if ext_filter:
                return f"No {query} found in your Desktop / Documents / Downloads."
            return f'No files found matching "{query}".'

        # Header line tells the user what we actually searched for
        if ext_filter:
            header = (
                f'Found {len(results)} {query} '
                f'(filter: {", ".join(ext_filter)}):'
            )
        else:
            header = f'Found {len(results)} result{"s" if len(results) != 1 else ""} for "{query}":'

        lines = [header, ""]
        for r in results:
            icon = "📁" if r["type"] == "folder" else self._file_icon(r.get("extension", ""))
            cm = " (content match)" if r.get("content_match") else ""
            size = ""
            if r.get("size") is not None and r["type"] == "file":
                size = f" — {self._fmt_size(r['size'])}"
            lines.append(f"{icon} {r['display_path']}{size}{cm}")
        return "\n".join(lines)

    # ── Read file ─────────────────────────────────────────────────────────────

    def read_file(self, path_str: str) -> Dict[str, Any]:
        """Read a text file and return its content."""
        p = self.resolve(path_str)
        if p is None:
            return {"success": False, "error": f"Path not allowed: {path_str}"}
        if not p.exists():
            return {"success": False, "error": f"File not found: {path_str}"}
        if p.is_dir():
            return {"success": False, "error": f"That's a folder — use list_directory instead."}
        if p.suffix.lower() not in TEXT_EXTENSIONS:
            return {"success": False, "error": f"Cannot read binary file: {p.name}. Supported: txt, md, py, js, json, etc."}
        try:
            content = p.read_text(errors="ignore")
            truncated = len(content) > MAX_READ_BYTES
            return {
                "success": True,
                "path": str(p),
                "display_path": self._display_path(p),
                "name": p.name,
                "content": content[:MAX_READ_BYTES],
                "truncated": truncated,
                "size": p.stat().st_size,
                "lines": content.count("\n"),
            }
        except PermissionError:
            return {"success": False, "error": f"Permission denied: {p.name}"}

    # ── Create file ───────────────────────────────────────────────────────────

    def create_file(self, path_str: str, content: str = "") -> Dict[str, Any]:
        """Create a new file with optional content."""
        p = self.resolve(path_str)
        if p is None:
            # If relative path given, default to Desktop
            p = ALLOWED_ROOTS["desktop"] / path_str
            if not self._is_allowed(p):
                return {"success": False, "error": f"Path not allowed: {path_str}"}
        if p.exists():
            return {"success": False, "error": f"File already exists: {p.name}. Use a different name or rename the existing file."}
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
            return {
                "success": True,
                "path": str(p),
                "display_path": self._display_path(p),
                "name": p.name,
                "message": f"Created: {self._display_path(p)}",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── Create folder ─────────────────────────────────────────────────────────

    def create_folder(self, path_str: str) -> Dict[str, Any]:
        """Create a new folder (and any missing parents)."""
        p = self.resolve(path_str)
        if p is None:
            # Default to Desktop if just a name given
            p = ALLOWED_ROOTS["desktop"] / path_str
            if not self._is_allowed(p):
                return {"success": False, "error": f"Path not allowed: {path_str}"}
        if p.exists():
            return {"success": False, "error": f"Folder already exists: {p.name}"}
        try:
            p.mkdir(parents=True, exist_ok=False)
            return {
                "success": True,
                "path": str(p),
                "display_path": self._display_path(p),
                "name": p.name,
                "message": f"Folder created: {self._display_path(p)}",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── Move (requires approval) ───────────────────────────────────────────────

    def prepare_move(self, source_str: str, dest_str: str) -> PendingFileOp | Dict:
        """Prepare a move operation — returns PendingFileOp for confirmation."""
        src = self.resolve(source_str)
        if src is None or not src.exists():
            return {"success": False, "error": f"Source not found: {source_str}"}
        dst = self.resolve(dest_str)
        if dst is None:
            dst = ALLOWED_ROOTS["desktop"] / dest_str
        if not self._is_allowed(dst):
            return {"success": False, "error": f"Destination not allowed: {dest_str}"}
        return PendingFileOp(operation="move", source=src, destination=dst)

    def execute_move(self, op: PendingFileOp) -> Dict[str, Any]:
        try:
            shutil.move(str(op.source), str(op.destination))
            return {
                "success": True,
                "message": f"Moved: {op.source.name} → {self._display_path(op.destination)}",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── Rename (requires approval) ────────────────────────────────────────────

    def prepare_rename(self, path_str: str, new_name: str) -> PendingFileOp | Dict:
        """Prepare a rename operation — returns PendingFileOp for confirmation."""
        p = self.resolve(path_str)
        if p is None or not p.exists():
            return {"success": False, "error": f"Not found: {path_str}"}
        new_path = p.parent / new_name
        if new_path.exists():
            return {"success": False, "error": f"A file named '{new_name}' already exists here."}
        return PendingFileOp(operation="rename", source=p, destination=new_path)

    def execute_rename(self, op: PendingFileOp) -> Dict[str, Any]:
        try:
            op.source.rename(op.destination)
            return {
                "success": True,
                "message": f"Renamed: {op.source.name} → {op.destination.name}",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── Delete (requires approval) ────────────────────────────────────────────

    def prepare_delete(self, path_str: str) -> PendingFileOp | Dict:
        """Prepare a delete operation — returns PendingFileOp for confirmation."""
        p = self.resolve(path_str)
        if p is None or not p.exists():
            return {"success": False, "error": f"Not found: {path_str}"}
        return PendingFileOp(operation="delete", source=p)

    def execute_delete(self, op: PendingFileOp) -> Dict[str, Any]:
        try:
            if op.source.is_dir():
                shutil.rmtree(op.source)
            else:
                op.source.unlink()
            return {
                "success": True,
                "message": f"Deleted: {op.source.name}",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── Get info ──────────────────────────────────────────────────────────────

    def get_info(self, path_str: str) -> Dict[str, Any]:
        p = self.resolve(path_str)
        if p is None or not p.exists():
            return {"success": False, "error": f"Not found: {path_str}"}
        stat = p.stat()
        info = self._file_info(p)
        info.update({
            "success": True,
            "created": datetime.fromtimestamp(stat.st_ctime).strftime("%d %b %Y %H:%M"),
            "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%d %b %Y %H:%M"),
            "size_bytes": stat.st_size,
            "size_human": self._fmt_size(stat.st_size),
        })
        if p.is_dir():
            try:
                children = list(p.iterdir())
                info["children"] = len(children)
            except Exception:
                pass
        return info

    def format_info(self, data: Dict[str, Any]) -> str:
        if not data.get("success"):
            return data.get("error", "Could not get info.")
        lines = [
            f"{'📁' if data['type']=='folder' else self._file_icon(data.get('extension',''))} {data['name']}",
            f"  Location: {data.get('display_path','')}",
            f"  Type: {data['type']}",
            f"  Size: {data.get('size_human', '—')}",
            f"  Modified: {data.get('modified', '—')}",
            f"  Created: {data.get('created', '—')}",
        ]
        if "children" in data:
            lines.append(f"  Contains: {data['children']} items")
        return "\n".join(lines)

    # ── Natural language query parsing ────────────────────────────────────────

    def parse_request(self, user_request: str) -> Dict[str, Any]:
        """
        Parse a natural language file request into structured intent.
        Returns dict with: action, path, destination, name, content, location
        """
        req = user_request.lower().strip()
        result: Dict[str, Any] = {"action": None, "path": None, "destination": None,
                                   "name": None, "content": "", "location": "all"}

        # Detect location
        if "desktop" in req:
            result["location"] = "desktop"
        elif "document" in req or " docs " in req:
            result["location"] = "documents"
        elif "download" in req:
            result["location"] = "downloads"

        # Create folder
        folder_match = re.search(
            r'(?:create|make|new|add)\s+(?:a\s+)?(?:new\s+)?folder\s+(?:called\s+|named\s+)?["\']?'
            r'([^"\']+?)["\']?'
            r'\s*(?:(?:on|in|at|to|onto)\s+(?:the\s+)?(?:desktop|documents|downloads|my\s+desktop|my\s+documents|my\s+downloads))?'
            r'\s*$',
            user_request, re.IGNORECASE
        )
        if folder_match:
            result["action"] = "create_folder"
            raw_name = folder_match.group(1).strip()
            # Strip trailing location words that leaked in
            raw_name = re.sub(
                r'\s+(?:on|in|at|to|onto)?\s*(?:my\s+)?(?:desktop|documents|downloads)\s*$',
                '', raw_name, flags=re.IGNORECASE
            ).strip()
            result["name"] = raw_name.title() if raw_name.islower() else raw_name
            # Detect location from full request
            for loc_word, loc_key in [("desktop", "desktop"), ("document", "documents"), ("download", "downloads")]:
                if loc_word in req:
                    result["location"] = loc_key
                    break
            return result

        # Create file. Capture filename AND optional content body.
        # Patterns supported:
        #   "create file notes.txt"
        #   "create a file called notes.txt"
        #   "create file notes.txt with content hello world"
        #   "create a file named foo.md saying  '# Hi\n\nNotes here'"
        #   "make a new note with text 'remember the milk'"
        #   "write a file called todo.md with the following:\nbullet 1\nbullet 2"
        #
        # Body delimiters recognised: with content / with text /
        # with body / saying / containing / that says / : / -
        file_create_match = re.search(
            r'(?:create|make|new|write|save)\s+(?:a\s+)?(?:new\s+)?'
            r'(?:file|note|document|doc|text\s+file|markdown\s+file)\s+'
            r'(?:called\s+|named\s+|titled\s+)?'
            r'["\']?([\w][\w\-\.\s]*?)["\']?'
            r'(?:\s+(?:with\s+(?:content|text|body|the\s+following|the\s+text)|'
            r'saying|containing|that\s+says|that\s+contains)[\s:\-]*'
            r'(.+))?$',
            user_request,
            re.IGNORECASE | re.DOTALL,
        )
        if file_create_match:
            result["action"] = "create_file"
            raw_name = file_create_match.group(1).strip().rstrip(",")
            # Strip leaked location words
            raw_name = re.sub(
                r'\s+(?:on|in|at|to|onto)?\s*(?:my\s+)?(?:desktop|documents|downloads)\s*$',
                '', raw_name, flags=re.IGNORECASE,
            ).strip()
            result["name"] = raw_name
            body = (file_create_match.group(2) or "").strip()
            # Strip wrapping quotes if the user wrapped the body
            if (body.startswith('"') and body.endswith('"')) or \
               (body.startswith("'") and body.endswith("'")):
                body = body[1:-1]
            result["content"] = body
            return result

        # Save the current conversation / chat to a file.
        # "save this conversation as foo.md", "save the chat to notes.txt"
        save_chat_match = re.search(
            r'(?:save|export|dump)\s+(?:this|the|our)?\s*'
            r'(?:conversation|chat|discussion|transcript|history)\s+'
            r'(?:as|to|into)?\s+["\']?([\w][\w\-\.\s]*?)["\']?\s*$',
            user_request, re.IGNORECASE,
        )
        if save_chat_match:
            result["action"] = "save_conversation"
            result["name"] = save_chat_match.group(1).strip()
            return result

        # Delete
        if any(w in req for w in ["delete", "remove", "trash"]):
            result["action"] = "delete"
            dm = re.search(
                r'(?:delete|remove|trash)\s+(?:the\s+)?(?:file\s+|folder\s+)?'
                r'(?:called\s+|named\s+)?["\']?(.+?)["\']?'
                r'(?:\s+(?:from|on|in|at)\s+(?:my\s+)?(?:desktop|documents|downloads))?\s*$',
                user_request, re.IGNORECASE
            )
            if dm:
                raw = dm.group(1).strip()
                # Strip any leaked location suffix
                raw = re.sub(
                    r'\s+(?:from|on|in|at)?\s*(?:my\s+)?(?:desktop|documents|downloads)\s*$',
                    '', raw, flags=re.IGNORECASE
                ).strip()
                result["path"] = raw
            return result

        # Rename
        rename_match = re.search(
            r'rename\s+["\']?(.+?)["\']?\s+to\s+["\']?(.+?)["\']?\s*$',
            req, re.IGNORECASE
        )
        if rename_match:
            result["action"] = "rename"
            result["path"] = rename_match.group(1).strip()
            result["name"] = rename_match.group(2).strip()
            return result

        # Move
        move_match = re.search(
            r'move\s+["\']?(.+?)["\']?\s+to\s+["\']?(.+?)["\']?\s*$',
            req, re.IGNORECASE
        )
        if move_match:
            result["action"] = "move"
            result["path"] = move_match.group(1).strip()
            result["destination"] = move_match.group(2).strip()
            return result

        # Search — extract full query, strip trailing location phrases
        if any(w in req for w in ["find", "search", "look for", "where is", "locate"]):
            result["action"] = "search"
            sm = re.search(
                r'(?:find|search\s+for|look\s+for|locate|where\s+is)\s+["\']?(.+?)["\']?'
                r'(?:\s+(?:in|on|at|from|inside)\s+(?:my\s+)?(?:desktop|documents|downloads|files|laptop|mac))?'
                r'\s*$',
                user_request, re.IGNORECASE
            )
            if sm:
                raw_q = sm.group(1).strip()
                # Strip trailing location words
                raw_q = re.sub(
                    r'\s+(?:in|on|at|inside)?\s*(?:my\s+)?(?:desktop|documents|downloads|files|laptop|mac)\s*$',
                    '', raw_q, flags=re.IGNORECASE
                ).strip()
                result["path"] = raw_q
            return result

        # List / browse — must come BEFORE read since "show me" is ambiguous
        _list_phrases = ["list", "show", "browse", "what's on", "whats on",
                         "what files", "what's in", "whats in", "show me my files",
                         "show me what", "what do i have", "what's on my", "whats on my"]
        _is_list = any(phrase in req for phrase in _list_phrases)
        # "show me X" with a location word = list; without = could be read
        _has_location = any(w in req for w in ["desktop", "documents", "downloads", "my files"])
        if _is_list and (_has_location or not any(w in req for w in ["read", "open", "content"])):
            result["action"] = "list"
            result["path"] = result["location"]
            return result

        # Read / open — only when a specific filename is clearly intended
        if any(w in req for w in ["read", "open", "display", "what's in", "whats in", "content of"]):
            result["action"] = "read"
            rm = re.search(r'(?:read|open|display|content of)\s+["\']?(.+?)["\']?\s*$', req, re.IGNORECASE)
            if rm:
                result["path"] = rm.group(1).strip()
            return result

        return result

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _file_info(self, p: Path) -> Dict[str, Any]:
        try:
            stat = p.stat()
            size = stat.st_size if p.is_file() else None
        except Exception:
            size = None
        return {
            "name": p.name,
            "path": str(p),
            "display_path": self._display_path(p),
            "type": "folder" if p.is_dir() else "file",
            "extension": p.suffix.lower() if p.is_file() else None,
            "size": size,
        }

    def _display_path(self, p: Path) -> str:
        try:
            rel = p.relative_to(HOME)
            return f"~/{rel}"
        except ValueError:
            return str(p)

    def _fmt_size(self, size: Optional[int]) -> str:
        if size is None:
            return "—"
        if size < 1024:
            return f"{size} B"
        elif size < 1024 ** 2:
            return f"{size/1024:.1f} KB"
        elif size < 1024 ** 3:
            return f"{size/1024**2:.1f} MB"
        return f"{size/1024**3:.2f} GB"

    def _file_icon(self, ext: str) -> str:
        icons = {
            ".pdf": "📄", ".doc": "📝", ".docx": "📝", ".txt": "📄",
            ".md": "📝", ".py": "🐍", ".js": "📜", ".ts": "📜",
            ".json": "📋", ".csv": "📊", ".xlsx": "📊", ".xls": "📊",
            ".png": "🖼", ".jpg": "🖼", ".jpeg": "🖼", ".gif": "🖼",
            ".mp4": "🎬", ".mov": "🎬", ".mp3": "🎵", ".wav": "🎵",
            ".zip": "🗜", ".tar": "🗜", ".gz": "🗜",
            ".html": "🌐", ".css": "🎨", ".sh": "⚙️", ".yaml": "⚙️",
            ".pptx": "📊", ".ppt": "📊",
        }
        return icons.get(ext.lower(), "📄")