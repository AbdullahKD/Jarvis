"""
Document Tool
Extracts and processes text from PDFs, DOCX, and plain text files.
Feeds into the Summariser and Memory agents.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List


class DocumentTool:
    """
    Extracts text from documents for processing by other agents.
    Supports: PDF, DOCX, TXT, MD, CSV.
    """

    async def extract(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text from a document file.

        Args:
            file_path: Absolute or relative path to the file

        Returns:
            Dict with text content and metadata
        """
        path = Path(file_path).expanduser().resolve()

        if not path.exists():
            return {"success": False, "error": f"File not found: {file_path}"}

        suffix = path.suffix.lower()

        if suffix == ".pdf":
            return await self._extract_pdf(path)
        elif suffix in (".docx", ".doc"):
            return await self._extract_docx(path)
        elif suffix in (".txt", ".md", ".rst"):
            return self._extract_text(path)
        elif suffix == ".csv":
            return self._extract_csv(path)
        else:
            return {"success": False, "error": f"Unsupported file type: {suffix}"}

    # ── PDF ────────────────────────────────────────────────────────────────

    async def _extract_pdf(self, path: Path) -> Dict[str, Any]:
        try:
            import pdfplumber
            pages_text = []
            with pdfplumber.open(path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    pages_text.append(text)

            full_text = "\n\n".join(pages_text)
            return {
                "success": True,
                "file": str(path),
                "type": "pdf",
                "pages": len(pages_text),
                "text": full_text,
                "char_count": len(full_text),
            }
        except ImportError:
            # Fallback to PyPDF2
            try:
                import PyPDF2
                pages_text = []
                with open(path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        pages_text.append(page.extract_text() or "")
                full_text = "\n\n".join(pages_text)
                return {
                    "success": True,
                    "file": str(path),
                    "type": "pdf",
                    "pages": len(pages_text),
                    "text": full_text,
                    "char_count": len(full_text),
                }
            except Exception as exc:
                return {"success": False, "error": f"PDF extraction failed: {exc}"}
        except Exception as exc:
            return {"success": False, "error": f"PDF extraction failed: {exc}"}

    # ── DOCX ───────────────────────────────────────────────────────────────

    async def _extract_docx(self, path: Path) -> Dict[str, Any]:
        try:
            from docx import Document
            doc = Document(path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            full_text = "\n\n".join(paragraphs)
            return {
                "success": True,
                "file": str(path),
                "type": "docx",
                "paragraphs": len(paragraphs),
                "text": full_text,
                "char_count": len(full_text),
            }
        except ImportError:
            return {"success": False, "error": "python-docx not installed. Run: pip install python-docx"}
        except Exception as exc:
            return {"success": False, "error": f"DOCX extraction failed: {exc}"}

    # ── Plain text ─────────────────────────────────────────────────────────

    def _extract_text(self, path: Path) -> Dict[str, Any]:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
            return {
                "success": True,
                "file": str(path),
                "type": path.suffix.lstrip("."),
                "text": text,
                "char_count": len(text),
                "line_count": text.count("\n"),
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    # ── CSV ────────────────────────────────────────────────────────────────

    def _extract_csv(self, path: Path) -> Dict[str, Any]:
        try:
            import csv
            rows = []
            with open(path, newline="", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames or []
                for row in reader:
                    rows.append(dict(row))
            preview = "\n".join(str(r) for r in rows[:10])
            return {
                "success": True,
                "file": str(path),
                "type": "csv",
                "headers": list(headers),
                "row_count": len(rows),
                "preview": preview,
                "text": preview,
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def format_result(self, data: Dict[str, Any], max_chars: int = 2000) -> str:
        if not data.get("success"):
            return f"Document error: {data.get('error')}"
        meta = f"[{data['type'].upper()}] {Path(data['file']).name}"
        text = data.get("text", "")[:max_chars]
        return f"{meta}\n\n{text}"
