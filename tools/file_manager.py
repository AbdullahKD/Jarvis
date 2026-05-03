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

        # Not found under any root but path is relative — return under desktop as default
        # (caller will check .exists() themselves)
        candidate = ALLOWED_ROOTS["desktop"] / p
        return candidate if self._is_allowed(candidate) else None
        return p if self._is_allowed(p) else None

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

    def list_directory(self, path_str: str = "desktop") -> Dict[str, Any]:
        """List contents of a folder."""
        p = self.resolve(path_str)
        if p is None:
            return {"success": False, "error": f"Path not found or not allowed: {path_str}"}
        if not p.exists():
            return {"success": False, "error": f"Path does not exist: {path_str}"}
        if not p.is_dir():
            return {"success": False, "error": f"Not a directory: {path_str}"}

        items = []
        try:
            for entry in sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name.lower())):
                if entry.name.startswith("."):
                    continue  # Skip hidden files
                stat = entry.stat()
                items.append({
                    "name": entry.name,
                    "type": "folder" if entry.is_dir() else "file",
                    "size": stat.st_size if entry.is_file() else None,
                    "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%d %b %Y %H:%M"),
                    "extension": entry.suffix.lower() if entry.is_file() else None,
                })
        except PermissionError:
            return {"success": False, "error": f"Permission denied: {path_str}"}

        return {
            "success": True,
            "path": str(p),
            "display_path": self._display_path(p),
            "items": items,
            "count": len(items),
            "folders": sum(1 for i in items if i["type"] == "folder"),
            "files": sum(1 for i in items if i["type"] == "file"),
        }

    def format_listing(self, data: Dict[str, Any]) -> str:
        if not data.get("success"):
            return data.get("error", "Could not list directory.")
        items = data.get("items", [])
        dp = data.get("display_path", "")
        lines = [f"{dp}  ({data['folders']} folders, {data['files']} files)\n"]
        for item in items:
            icon = "📁" if item["type"] == "folder" else self._file_icon(item.get("extension",""))
            size = f"  {self._fmt_size(item['size'])}" if item["size"] is not None else ""
            lines.append(f"{icon} {item['name']}{size}  — {item['modified']}")
        return "\n".join(lines)

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query: str, location: str = "all", content_search: bool = False) -> Dict[str, Any]:
        """
        Search for files/folders by name, optionally by content.
        location: 'all', 'desktop', 'documents', 'downloads'
        """
        query_lower = query.lower()
        roots = (
            [ALLOWED_ROOTS[location]] if location in ALLOWED_ROOTS
            else list(ALLOWED_ROOTS.values())
        )

        results = []
        for root in roots:
            if not root.exists():
                continue
            for p in root.rglob("*"):
                if p.name.startswith("."):
                    continue
                if len(results) >= MAX_SEARCH_RESULTS:
                    break
                # Name match
                if query_lower in p.name.lower():
                    results.append(self._file_info(p))
                    continue
                # Content match (text files only)
                if content_search and p.is_file() and p.suffix.lower() in TEXT_EXTENSIONS:
                    try:
                        text = p.read_text(errors="ignore")[:10_000]
                        if query_lower in text.lower():
                            results.append({**self._file_info(p), "content_match": True})
                    except Exception:
                        pass

        return {
            "success": True,
            "query": query,
            "results": results,
            "count": len(results),
        }

    def format_search(self, data: Dict[str, Any]) -> str:
        if not data.get("success"):
            return data.get("error", "Search failed.")
        results = data.get("results", [])
        if not results:
            return f"No files found matching \"{data.get('query', '')}\"."
        lines = [f"Found {len(results)} result(s) for \"{data['query']}\":\n"]
        for r in results:
            icon = "📁" if r["type"] == "folder" else self._file_icon(r.get("extension",""))
            cm = " (content match)" if r.get("content_match") else ""
            lines.append(f"{icon} {r['display_path']}{cm}")
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

        # Create file
        file_create_match = re.search(
            r'(?:create|make|new|write)\s+(?:a\s+)?(?:new\s+)?file\s+(?:called\s+|named\s+)?["\']?([^\s"\']+)["\']?',
            req, re.IGNORECASE
        )
        if file_create_match:
            result["action"] = "create_file"
            result["name"] = file_create_match.group(1).strip()
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