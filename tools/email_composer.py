"""
Email Composer
LLM-powered email drafting with contact resolution and confirmation.
Implements Level 2-4 of the smart email pipeline:
  Level 2: LLM extracts intent and writes the email body
  Level 3: Contact book resolves names to email addresses
  Level 4: Confirmation step before sending
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from config.llm_client import OllamaClient


EXTRACT_PROMPT = """You are J.A.R.V.I.S, a personal AI assistant helping your user send emails.

Extract email details and write a compelling, well-crafted email. Respond with valid JSON only:
{
  "recipient_name": "John",
  "recipient_email": "",
  "subject": "engaging specific subject line",
  "body": "full email body",
  "tone": "formal or casual",
  "intent": "one sentence description"
}

Rules for writing the email body:
- Write with genuine warmth and personality — not robotic or generic
- Be specific to what the user asked — add relevant detail and context
- Keep it concise but impactful — 2-4 short paragraphs
- Open with a warm, human line suited to the relationship
- Formal for professional emails, warm and personal for casual ones
- Sign off naturally: Best, / Warm regards, / Cheers, based on tone
- Sign with: Abdullah
- Subject must be specific and engaging, never generic
- ONLY put an email in recipient_email if explicitly provided by user
- NEVER invent or guess email addresses — leave as empty string if unknown
"""


@dataclass
class EmailDraft:
    recipient_name: str
    recipient_email: str
    subject: str
    body: str
    tone: str
    intent: str
    contact_found: bool = False
    needs_email: bool = False


class EmailComposer:
    """
    Handles the full email composition pipeline:
    1. Extract intent from natural language
    2. Resolve recipient via contact book
    3. Generate email with LLM
    4. Return draft for user confirmation
    """

    def __init__(self, llm_client: OllamaClient):
        self.llm = llm_client

    async def compose(
        self,
        user_request: str,
        contact_book=None,
    ) -> EmailDraft:
        """
        Compose an email from a natural language request.

        Args:
            user_request: What the user said
            contact_book: ContactBook instance for name resolution

        Returns:
            EmailDraft with all fields populated
        """
        messages = [
            {"role": "system", "content": EXTRACT_PROMPT},
            {"role": "user", "content": f'User request: "{user_request}"'},
        ]

        try:
            data = await self.llm.chat_json(messages)
        except Exception as e:
            # Fallback: try basic regex extraction
            return self._fallback_compose(user_request)

        draft = EmailDraft(
            recipient_name=data.get("recipient_name", ""),
            recipient_email=data.get("recipient_email", ""),
            subject=data.get("subject", "Message from Jarvis"),
            body=data.get("body", ""),
            tone=data.get("tone", "formal"),
            intent=data.get("intent", ""),
        )

        # Level 3: Contact book resolution
        if contact_book and draft.recipient_name and not draft.recipient_email:
            contact = contact_book.find(draft.recipient_name)
            if contact:
                draft.recipient_email = contact["email"]
                draft.contact_found = True
                print(f"📒 Resolved '{draft.recipient_name}' → {draft.recipient_email}")
            else:
                draft.needs_email = True
                print(f"📒 Contact '{draft.recipient_name}' not found")

        elif not draft.recipient_email:
            # Try to extract email directly from request
            email_match = re.search(
                r'[\w._%+\-]+@[\w.\-]+\.[a-zA-Z]{2,}',
                user_request
            )
            if email_match:
                draft.recipient_email = email_match.group(0)

        return draft

    def format_draft_for_confirmation(self, draft: EmailDraft) -> str:
        """Format the draft for user review."""
        return (
            f"\n📧 Here's the email I've drafted:\n"
            f"{'─'*40}\n"
            f"To:      {draft.recipient_name} <{draft.recipient_email}>\n"
            f"Subject: {draft.subject}\n"
            f"{'─'*40}\n"
            f"{draft.body}\n"
            f"{'─'*40}\n"
            f"\nShall I send this? (yes / no / edit)"
        )

    def _fallback_compose(self, user_request: str) -> EmailDraft:
        """Basic regex fallback if LLM fails."""
        email_match = re.search(r'[\w._%+\-]+@[\w.\-]+\.[a-zA-Z]{2,}', user_request)
        email = email_match.group(0) if email_match else ""
        return EmailDraft(
            recipient_name=email,
            recipient_email=email,
            subject="Message from Jarvis",
            body=user_request,
            tone="formal",
            intent="send email",
        )