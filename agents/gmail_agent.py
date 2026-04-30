"""
Gmail Agent
Google Gmail integration with OAuth2.
Supports: read inbox, search emails, draft, send.
Falls back to mock mode if credentials not configured.
"""

from __future__ import annotations

import base64
import email as email_lib
from datetime import datetime
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

from config.settings import GOOGLE_CREDENTIALS_PATH, GOOGLE_TOKEN_PATH, GOOGLE_SCOPES

try:
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False


class GmailAgent:
    """
    Gmail agent with OAuth2 authentication.
    Shares credentials with CalendarAgent via the same token file.
    """

    def __init__(self):
        self.service = None
        self.mock_inbox: List[Dict] = self._seed_mock_inbox()
        self._init_service()

    def _init_service(self) -> None:
        if not GOOGLE_AVAILABLE or not GOOGLE_CREDENTIALS_PATH.exists():
            print("📧 GmailAgent: mock mode (no credentials)")
            return
        try:
            creds = None
            if GOOGLE_TOKEN_PATH.exists():
                creds = Credentials.from_authorized_user_file(
                    str(GOOGLE_TOKEN_PATH), GOOGLE_SCOPES
                )
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        str(GOOGLE_CREDENTIALS_PATH), GOOGLE_SCOPES
                    )
                    creds = flow.run_local_server(port=0)
                    GOOGLE_TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
                    GOOGLE_TOKEN_PATH.write_text(creds.to_json())

            self.service = build("gmail", "v1", credentials=creds)
            print("📧 GmailAgent: Gmail connected ✅")
        except Exception as e:
            print(f"📧 GmailAgent: Auth failed ({e}) — mock mode")

    @property
    def is_mock(self) -> bool:
        return self.service is None

    # ── Read emails ────────────────────────────────────────────────────────

    async def get_inbox(
        self, max_results: int = 5, query: str = "is:unread"
    ) -> Dict[str, Any]:
        """Fetch emails from inbox."""
        if self.is_mock:
            return {
                "success": True,
                "emails": self.mock_inbox[:max_results],
                "count": min(max_results, len(self.mock_inbox)),
                "message": self._format_emails(self.mock_inbox[:max_results]),
                "mock": True,
            }

        try:
            result = self.service.users().messages().list(
                userId="me", q=query, maxResults=max_results
            ).execute()

            messages = result.get("messages", [])
            emails = []
            for msg in messages:
                detail = self.service.users().messages().get(
                    userId="me", id=msg["id"], format="metadata",
                    metadataHeaders=["From", "Subject", "Date"]
                ).execute()
                headers = {
                    h["name"]: h["value"]
                    for h in detail.get("payload", {}).get("headers", [])
                }
                emails.append({
                    "id": msg["id"],
                    "from": headers.get("From", "Unknown"),
                    "subject": headers.get("Subject", "(no subject)"),
                    "date": headers.get("Date", ""),
                    "snippet": detail.get("snippet", ""),
                })

            return {
                "success": True,
                "emails": emails,
                "count": len(emails),
                "message": self._format_emails(emails),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def search_emails(
        self, query: str, max_results: int = 5
    ) -> Dict[str, Any]:
        """Search emails by query string."""
        return await self.get_inbox(max_results=max_results, query=query)

    # ── Send email ─────────────────────────────────────────────────────────

    async def send_email(
        self,
        to: str,
        subject: str,
        body: str,
        cc: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send an email."""
        if self.is_mock:
            sent = {
                "to": to, "subject": subject,
                "body": body[:100], "cc": cc, "mock": True,
            }
            print(f"📧 [Mock] Email sent to {to}: {subject}")
            return {
                "success": True,
                "message": f"[Mock] Sent email to {to} — Subject: {subject}",
                **sent,
            }

        try:
            mime = MIMEText(body)
            mime["to"] = to
            mime["subject"] = subject
            if cc:
                mime["cc"] = cc

            raw = base64.urlsafe_b64encode(mime.as_bytes()).decode()
            result = self.service.users().messages().send(
                userId="me", body={"raw": raw}
            ).execute()

            return {
                "success": True,
                "id": result["id"],
                "message": f"Email sent to {to} — Subject: {subject}",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── Draft email ────────────────────────────────────────────────────────

    async def draft_email(
        self,
        to: str,
        subject: str,
        body: str,
    ) -> Dict[str, Any]:
        """Save a draft without sending."""
        if self.is_mock:
            return {
                "success": True,
                "message": f"[Mock] Draft saved — To: {to}, Subject: {subject}",
                "mock": True,
            }
        try:
            mime = MIMEText(body)
            mime["to"] = to
            mime["subject"] = subject
            raw = base64.urlsafe_b64encode(mime.as_bytes()).decode()
            result = self.service.users().drafts().create(
                userId="me", body={"message": {"raw": raw}}
            ).execute()
            return {
                "success": True,
                "id": result["id"],
                "message": f"Draft saved — To: {to}, Subject: {subject}",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── Helpers ────────────────────────────────────────────────────────────

    def _format_emails(self, emails: List[Dict]) -> str:
        if not emails:
            return "No emails found."
        lines = [f"📧 {len(emails)} email(s):"]
        for e in emails:
            lines.append(f"\n  From: {e.get('from', 'Unknown')}")
            lines.append(f"  Subject: {e.get('subject', '(no subject)')}")
            if e.get("snippet"):
                lines.append(f"  Preview: {e['snippet'][:100]}...")
        return "\n".join(lines)

    def _seed_mock_inbox(self) -> List[Dict]:
        return [
            {
                "id": "mock_001",
                "from": "alice@company.com",
                "subject": "Q3 Project Update",
                "date": "Wed, 30 Apr 2026 09:00:00",
                "snippet": "Hi, just wanted to share the latest update on the Q3 project...",
            },
            {
                "id": "mock_002",
                "from": "bob@company.com",
                "subject": "Meeting tomorrow at 2pm",
                "date": "Wed, 30 Apr 2026 08:30:00",
                "snippet": "Can we confirm the meeting is still on for tomorrow at 2pm?",
            },
            {
                "id": "mock_003",
                "from": "notifications@github.com",
                "subject": "Pull request review requested",
                "date": "Tue, 29 Apr 2026 17:00:00",
                "snippet": "Review requested on: feat/multi-agent-improvements...",
            },
        ]