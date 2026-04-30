"""
Contact Book
Local JSON-based contact store.
Allows Jarvis to resolve names to email addresses.
Contacts are stored in ~/.jarvis/contacts.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

CONTACTS_PATH = Path.home() / ".jarvis" / "contacts.json"


class ContactBook:
    """
    Local contact store. Maps names to email addresses.
    Persists to ~/.jarvis/contacts.json between sessions.
    """

    def __init__(self):
        CONTACTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._contacts: Dict[str, Dict] = {}
        self._load()

    def _load(self) -> None:
        if CONTACTS_PATH.exists():
            try:
                self._contacts = json.loads(CONTACTS_PATH.read_text())
            except Exception:
                self._contacts = {}

    def _save(self) -> None:
        CONTACTS_PATH.write_text(json.dumps(self._contacts, indent=2))

    def add(self, name: str, email: str, notes: str = "") -> None:
        """Add or update a contact."""
        key = name.lower().strip()
        self._contacts[key] = {
            "name": name.strip(),
            "email": email.strip().lower(),
            "notes": notes,
        }
        self._save()
        print(f"📒 Contact saved: {name} → {email}")

    def find(self, name: str) -> Optional[Dict]:
        """
        Find a contact by name. Supports partial matching.
        e.g. "John" matches "John Smith"
        """
        key = name.lower().strip()

        # Exact match first
        if key in self._contacts:
            return self._contacts[key]

        # Partial match — first name or last name
        for stored_key, contact in self._contacts.items():
            if key in stored_key or stored_key in key:
                return contact

        return None

    def find_by_email(self, email: str) -> Optional[Dict]:
        """Find a contact by email address."""
        email = email.lower().strip()
        for contact in self._contacts.values():
            if contact["email"] == email:
                return contact
        return None

    def list_all(self) -> List[Dict]:
        """Return all contacts."""
        return list(self._contacts.values())

    def delete(self, name: str) -> bool:
        """Delete a contact by name."""
        key = name.lower().strip()
        if key in self._contacts:
            del self._contacts[key]
            self._save()
            return True
        return False

    def format_list(self) -> str:
        """Format contact list for display."""
        if not self._contacts:
            return "No contacts saved yet."
        lines = [f"📒 {len(self._contacts)} contact(s):"]
        for c in self._contacts.values():
            lines.append(f"  • {c['name']} — {c['email']}")
            if c.get("notes"):
                lines.append(f"    {c['notes']}")
        return "\n".join(lines)