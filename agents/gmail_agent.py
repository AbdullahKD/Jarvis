"""
Gmail Agent
Google Gmail integration with OAuth2.
Supports: read inbox, get full body, reply in thread, mark as read, archive, send, draft.
Falls back to mock mode if credentials not configured.
"""

from __future__ import annotations

import base64
import email as email_lib
import re
from datetime import datetime
from email.mime.multipart import MIMEMultipart
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


# ── Skip lists for contact extraction ─────────────────────────────────────────
_SKIP_RAW = {
    "gmail", "google", "notifications", "noreply", "no-reply", "no_reply",
    "mailer", "bounce", "support", "help", "info", "hello", "contact",
    "team", "news", "newsletter", "updates", "alert", "alerts", "reply",
    "donotreply", "do-not-reply", "mail", "automated", "system", "service",
    "services", "admin", "administrator", "billing", "accounts", "account",
    "security", "privacy", "legal", "abuse", "postmaster", "webmaster",
    "mail delivery subsystem", "mailer daemon", "mail daemon",
}
_SKIP_DISPLAY = {
    "gmail", "google", "mail delivery subsystem", "mailer-daemon",
    "mailer daemon", "mail daemon", "postmaster",
}

def _norm(s: str) -> str:
    return re.sub(r'[\-_.\s]', '', s).lower()

_skip_normalised = {_norm(s) for s in _SKIP_RAW}


class GmailAgent:
    """
    Gmail agent with OAuth2 authentication.
    Shares credentials with CalendarAgent via the same token file.
    """

    def __init__(self):
        self.service = None
        self.mock_inbox: List[Dict] = self._seed_mock_inbox()
        # Diagnostic: captured for the UI / /google/status endpoint so a
        # user can see why Gmail is in mock mode rather than guessing.
        self.auth_error: Optional[str] = None
        self._init_service()

    def _init_service(self) -> None:
        if not GOOGLE_AVAILABLE:
            self.auth_error = "google-api-python-client not installed"
            print(f"📧 GmailAgent: {self.auth_error} — mock mode")
            return
        if not GOOGLE_CREDENTIALS_PATH.exists():
            self.auth_error = f"credentials.json not found at {GOOGLE_CREDENTIALS_PATH}"
            print(f"📧 GmailAgent: {self.auth_error} — mock mode")
            return
        try:
            creds = None
            if GOOGLE_TOKEN_PATH.exists():
                creds = Credentials.from_authorized_user_file(
                    str(GOOGLE_TOKEN_PATH), GOOGLE_SCOPES
                )
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    # Refresh + persist. Persistence was missing before,
                    # so each restart re-refreshed needlessly and any
                    # refresh-token issue stayed hidden.
                    try:
                        creds.refresh(Request())
                        print("📧 GmailAgent: refreshed expired token")
                        GOOGLE_TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
                        GOOGLE_TOKEN_PATH.write_text(creds.to_json())
                    except Exception as refresh_exc:
                        self.auth_error = (
                            f"token refresh failed ({type(refresh_exc).__name__}: "
                            f"{refresh_exc}). Delete token.json and restart to "
                            f"trigger fresh OAuth — or move the app to 'Production' "
                            f"in Google Cloud Console to stop refresh tokens "
                            f"expiring every 7 days."
                        )
                        print(f"📧 GmailAgent: {self.auth_error}")
                        return
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        str(GOOGLE_CREDENTIALS_PATH), GOOGLE_SCOPES
                    )
                    creds = flow.run_local_server(port=0)
                    GOOGLE_TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
                    GOOGLE_TOKEN_PATH.write_text(creds.to_json())
                    print("📧 GmailAgent: completed fresh OAuth flow")

            self.service = build("gmail", "v1", credentials=creds)
            self.auth_error = None
            print("📧 GmailAgent: Gmail connected ✅")
        except Exception as e:
            self.auth_error = f"{type(e).__name__}: {e}"
            print(f"📧 GmailAgent: Auth failed — {self.auth_error}")
            print("   Tip: delete token.json and restart to re-auth.")

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
                    metadataHeaders=["From", "Subject", "Date", "Message-ID", "Thread-Index"]
                ).execute()
                headers = {
                    h["name"]: h["value"]
                    for h in detail.get("payload", {}).get("headers", [])
                }
                label_ids = detail.get("labelIds", [])
                emails.append({
                    "id": msg["id"],
                    "thread_id": detail.get("threadId", msg["id"]),
                    "from": headers.get("From", "Unknown"),
                    "subject": headers.get("Subject", "(no subject)"),
                    "date": headers.get("Date", ""),
                    "message_id": headers.get("Message-ID", ""),
                    "snippet": detail.get("snippet", ""),
                    "unread": "UNREAD" in label_ids,
                })

            return {
                "success": True,
                "emails": emails,
                "count": len(emails),
                "message": self._format_emails(emails),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_email_body(self, email_id: str) -> Dict[str, Any]:
        """Fetch the full body of a specific email."""
        if self.is_mock:
            email = next((e for e in self.mock_inbox if e["id"] == email_id), None)
            if not email:
                return {"success": False, "error": "Email not found"}
            body = email.get("body", email.get("snippet", "(no body)"))
            return {
                "success": True,
                "id": email_id,
                "from": email.get("from", ""),
                "subject": email.get("subject", ""),
                "date": email.get("date", ""),
                "body": body,
                "thread_id": email.get("thread_id", email_id),
                "message_id": email.get("message_id", ""),
            }

        try:
            detail = self.service.users().messages().get(
                userId="me", id=email_id, format="full"
            ).execute()
            headers = {
                h["name"]: h["value"]
                for h in detail.get("payload", {}).get("headers", [])
            }
            body = self._extract_body(detail.get("payload", {}))
            return {
                "success": True,
                "id": email_id,
                "thread_id": detail.get("threadId", email_id),
                "from": headers.get("From", "Unknown"),
                "subject": headers.get("Subject", "(no subject)"),
                "date": headers.get("Date", ""),
                "message_id": headers.get("Message-ID", ""),
                "body": body,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_thread(self, thread_id: str) -> Dict[str, Any]:
        """Fetch all messages in a thread."""
        if self.is_mock:
            return {"success": True, "messages": [], "thread_id": thread_id}
        try:
            result = self.service.users().threads().get(
                userId="me", id=thread_id, format="full"
            ).execute()
            messages = []
            for msg in result.get("messages", []):
                headers = {
                    h["name"]: h["value"]
                    for h in msg.get("payload", {}).get("headers", [])
                }
                messages.append({
                    "id": msg["id"],
                    "from": headers.get("From", ""),
                    "subject": headers.get("Subject", ""),
                    "date": headers.get("Date", ""),
                    "message_id": headers.get("Message-ID", ""),
                    "body": self._extract_body(msg.get("payload", {})),
                })
            return {"success": True, "thread_id": thread_id, "messages": messages}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def search_emails(
        self, query: str, max_results: int = 5
    ) -> Dict[str, Any]:
        """Search emails by query string."""
        return await self.get_inbox(max_results=max_results, query=query)

    # ── Send / Reply ───────────────────────────────────────────────────────

    @staticmethod
    def _body_to_html(body: str) -> str:
        """
        Convert plain-text email body into clean, readable HTML.

        Why: Gmail/Outlook etc. collapse single newlines in plain-text emails,
        so a body like "Hi Alice,\\n\\nThanks!\\n\\nBest,\\nAbdullah" arrives
        as one wall of run-on text. Wrapping in <p> + <br> preserves the
        paragraph structure the user actually sees in the draft preview.
        """
        if not body:
            return "<html><body></body></html>"

        # Escape HTML so any literal angle brackets in the body don't break
        # rendering (e.g. someone pastes "<placeholder>" into the draft).
        import html as _html
        safe = _html.escape(body.strip())

        # Split on blank lines → paragraphs; within a paragraph, single
        # newlines become <br/> (preserves sign-off line breaks).
        paragraphs = re.split(r'\n\s*\n', safe)
        html_paragraphs = [
            "<p>" + p.replace("\n", "<br/>") + "</p>"
            for p in paragraphs
            if p.strip()
        ]

        return (
            "<html><body style=\"font-family:-apple-system,BlinkMacSystemFont,"
            "'Segoe UI',Roboto,Helvetica,Arial,sans-serif;font-size:14px;"
            "line-height:1.5;color:#222;\">"
            + "".join(html_paragraphs)
            + "</body></html>"
        )

    def _build_mime(
        self,
        to: str,
        subject: str,
        body: str,
        cc: Optional[str] = None,
        in_reply_to: Optional[str] = None,
        references: Optional[str] = None,
    ) -> str:
        """
        Build a multipart/alternative MIME message (plain + HTML) and return
        a urlsafe-base64 encoded string ready for the Gmail API.

        Both parts share the same content so old clients still render the
        plain version while modern clients pick up the HTML formatting.
        """
        mime = MIMEMultipart("alternative")
        mime["to"] = to
        mime["subject"] = subject
        if cc:
            mime["cc"] = cc
        if in_reply_to:
            mime["In-Reply-To"] = in_reply_to
        if references:
            mime["References"] = references

        # Order matters: per RFC 2046 the last alternative is the preferred
        # one when the client supports it. HTML must come second.
        mime.attach(MIMEText(body, "plain", "utf-8"))
        mime.attach(MIMEText(self._body_to_html(body), "html", "utf-8"))

        return base64.urlsafe_b64encode(mime.as_bytes()).decode()

    async def send_email(
        self,
        to: str,
        subject: str,
        body: str,
        cc: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send an email (multipart/alternative: plain + HTML)."""
        if self.is_mock:
            print(f"📧 [Mock] Email sent to {to}: {subject}")
            return {
                "success": True,
                "message": f"[Mock] Sent email to {to} — Subject: {subject}",
                "to": to, "subject": subject, "mock": True,
            }

        try:
            raw = self._build_mime(to=to, subject=subject, body=body, cc=cc)
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

    async def reply_to_email(
        self,
        original_email: Dict[str, Any],
        body: str,
    ) -> Dict[str, Any]:
        """Reply in-thread to an existing email (multipart/alternative)."""
        to = original_email.get("from", "")
        subject = original_email.get("subject", "")
        if not subject.lower().startswith("re:"):
            subject = f"Re: {subject}"
        thread_id = original_email.get("thread_id", "")
        message_id = original_email.get("message_id", "")

        if self.is_mock:
            print(f"📧 [Mock] Reply sent to {to}: {subject}")
            return {
                "success": True,
                "message": f"[Mock] Reply sent to {to} — {subject}",
                "mock": True,
            }

        try:
            raw = self._build_mime(
                to=to,
                subject=subject,
                body=body,
                in_reply_to=message_id or None,
                references=message_id or None,
            )
            msg_body: Dict[str, Any] = {"raw": raw}
            if thread_id:
                msg_body["threadId"] = thread_id

            result = self.service.users().messages().send(
                userId="me", body=msg_body
            ).execute()

            return {
                "success": True,
                "id": result["id"],
                "message": f"Reply sent to {self._parse_address(to)['name'] or to} — {subject}",
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
        """Save a draft (multipart/alternative) without sending."""
        if self.is_mock:
            return {
                "success": True,
                "message": f"[Mock] Draft saved — To: {to}, Subject: {subject}",
                "mock": True,
            }
        try:
            raw = self._build_mime(to=to, subject=subject, body=body)
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

    # ── Label actions ──────────────────────────────────────────────────────

    async def mark_as_read(self, email_id: str) -> Dict[str, Any]:
        """Remove UNREAD label from an email."""
        if self.is_mock:
            for e in self.mock_inbox:
                if e["id"] == email_id:
                    e["unread"] = False
            return {"success": True, "message": "Marked as read."}
        try:
            self.service.users().messages().modify(
                userId="me", id=email_id,
                body={"removeLabelIds": ["UNREAD"]}
            ).execute()
            return {"success": True, "message": "Marked as read."}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def mark_all_as_read(self) -> Dict[str, Any]:
        """Mark all unread emails as read."""
        if self.is_mock:
            for e in self.mock_inbox:
                e["unread"] = False
            return {"success": True, "message": "All emails marked as read."}
        try:
            result = self.service.users().messages().list(
                userId="me", q="is:unread", maxResults=50
            ).execute()
            messages = result.get("messages", [])
            for msg in messages:
                self.service.users().messages().modify(
                    userId="me", id=msg["id"],
                    body={"removeLabelIds": ["UNREAD"]}
                ).execute()
            return {"success": True, "message": f"Marked {len(messages)} emails as read."}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def archive_email(self, email_id: str) -> Dict[str, Any]:
        """Archive an email (remove INBOX label)."""
        if self.is_mock:
            self.mock_inbox = [e for e in self.mock_inbox if e["id"] != email_id]
            return {"success": True, "message": "Email archived."}
        try:
            self.service.users().messages().modify(
                userId="me", id=email_id,
                body={"removeLabelIds": ["INBOX"]}
            ).execute()
            return {"success": True, "message": "Email archived."}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── Contact extraction ─────────────────────────────────────────────────

    def extract_contacts_from_emails(self, emails: List[Dict]) -> List[Dict]:
        """
        Extract unique non-system contacts from a list of emails.
        Returns list of {name, email} dicts.
        """
        seen_emails: set = set()
        contacts = []

        for email in emails:
            from_field = email.get("from", "")
            parsed = self._parse_address(from_field)
            addr = parsed["email"].lower()
            name = parsed["name"]

            if not addr or "@" not in addr:
                continue

            local = addr.split("@")[0]
            domain = addr.split("@")[1] if "@" in addr else ""

            # Skip system/automated senders
            if _norm(local) in _skip_normalised:
                continue
            if any(_norm(part) in _skip_normalised for part in local.split(".")):
                continue
            if any(skip in domain for skip in ["noreply", "no-reply", "bounce", "mailer"]):
                continue
            if name and any(_norm(name).startswith(_norm(s)) for s in _SKIP_DISPLAY):
                continue

            if addr in seen_emails:
                continue

            seen_emails.add(addr)
            contacts.append({
                "name": name or local.replace(".", " ").title(),
                "email": addr,
            })

        return contacts

    # ── Helpers ────────────────────────────────────────────────────────────

    def _extract_body(self, payload: Dict) -> str:
        """Recursively extract plain text body from a MIME payload."""
        mime_type = payload.get("mimeType", "")
        body_data = payload.get("body", {}).get("data", "")

        if mime_type == "text/plain" and body_data:
            try:
                return base64.urlsafe_b64decode(body_data + "==").decode("utf-8", errors="replace")
            except Exception:
                pass

        if mime_type == "text/html" and body_data:
            try:
                html = base64.urlsafe_b64decode(body_data + "==").decode("utf-8", errors="replace")
                return self._strip_html(html)
            except Exception:
                pass

        # Recurse into parts, preferring text/plain
        parts = payload.get("parts", [])
        plain = next((p for p in parts if p.get("mimeType") == "text/plain"), None)
        if plain:
            return self._extract_body(plain)

        for part in parts:
            result = self._extract_body(part)
            if result.strip():
                return result

        return ""

    def _strip_html(self, html: str) -> str:
        """Fast HTML to plain text without external dependencies."""
        # Remove script/style blocks
        html = re.sub(r'<(script|style)[^>]*>.*?</(script|style)>', '', html, flags=re.DOTALL | re.IGNORECASE)
        # Replace block elements with newlines
        html = re.sub(r'<br\s*/?>', '\n', html, flags=re.IGNORECASE)
        html = re.sub(r'</(p|div|li|tr|h[1-6])>', '\n', html, flags=re.IGNORECASE)
        # Remove remaining tags
        html = re.sub(r'<[^>]+>', '', html)
        # Decode common HTML entities
        for entity, char in [('&amp;', '&'), ('&lt;', '<'), ('&gt;', '>'),
                               ('&nbsp;', ' '), ('&quot;', '"'), ('&#39;', "'")]:
            html = html.replace(entity, char)
        # Collapse whitespace
        lines = [line.strip() for line in html.splitlines()]
        lines = [l for l in lines if l]
        return '\n'.join(lines)

    def _parse_address(self, address: str) -> Dict[str, str]:
        """Parse 'Name <email@domain.com>' into name and email parts."""
        m = re.match(r'^"?([^"<]*?)"?\s*<([^>]+)>', address.strip())
        if m:
            return {"name": m.group(1).strip(), "email": m.group(2).strip().lower()}
        # Plain email address
        if "@" in address:
            return {"name": "", "email": address.strip().lower()}
        return {"name": address.strip(), "email": ""}

    def _format_emails(self, emails: List[Dict]) -> str:
        if not emails:
            return "No emails found."
        lines = [f"📧 {len(emails)} email(s):\n"]
        for i, e in enumerate(emails, 1):
            parsed = self._parse_address(e.get("from", "Unknown"))
            sender = parsed["name"] or parsed["email"] or "Unknown"
            unread_dot = "🔵 " if e.get("unread", False) else "   "
            lines.append(f"{unread_dot}{i}. From: {sender}")
            lines.append(f"      Subject: {e.get('subject', '(no subject)')}")
            if e.get("snippet"):
                snippet = e["snippet"][:90].replace("\n", " ")
                lines.append(f"      Preview: {snippet}...")
            lines.append("")
        lines.append("To read an email say: read email 1")
        lines.append("To reply say: reply to email 1")
        lines.append("To archive say: archive email 1")
        return "\n".join(lines)

    def _seed_mock_inbox(self) -> List[Dict]:
        return [
            {
                "id": "mock_001",
                "thread_id": "thread_001",
                "from": "Alice Johnson <alice@company.com>",
                "subject": "Q3 Project Update",
                "date": "Wed, 30 Apr 2026 09:00:00",
                "snippet": "Hi, just wanted to share the latest update on the Q3 project...",
                "body": "Hi Abdullah,\n\nJust wanted to share the latest update on the Q3 project. We are on track and should hit all milestones by end of May.\n\nLet me know if you need anything.\n\nBest,\nAlice",
                "message_id": "<mock001@company.com>",
                "unread": True,
            },
            {
                "id": "mock_002",
                "thread_id": "thread_002",
                "from": "Bob Smith <bob@company.com>",
                "subject": "Meeting tomorrow at 2pm",
                "date": "Wed, 30 Apr 2026 08:30:00",
                "snippet": "Can we confirm the meeting is still on for tomorrow at 2pm?",
                "body": "Hi Abdullah,\n\nCan we confirm the meeting is still on for tomorrow at 2pm? Please let me know.\n\nThanks,\nBob",
                "message_id": "<mock002@company.com>",
                "unread": True,
            },
            {
                "id": "mock_003",
                "thread_id": "thread_003",
                "from": "notifications@github.com",
                "subject": "Pull request review requested",
                "date": "Tue, 29 Apr 2026 17:00:00",
                "snippet": "Review requested on: feat/multi-agent-improvements...",
                "body": "A pull request review has been requested on: feat/multi-agent-improvements",
                "message_id": "<mock003@github.com>",
                "unread": False,
            },
        ]
