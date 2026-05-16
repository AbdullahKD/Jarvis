"""
Email Composer
Fast, LLM-light email drafting with contact resolution and confirmation.

Pipeline:
  Phase 1 (instant, no LLM):
    - Regex extraction of recipient name + topic from user request
    - Contact book resolution
    - Subject line generated from topic keywords
    - Tone detection from linguistic signals
    - Zero-LLM template matching (8 common patterns)

  Phase 2 (LLM, body-only, fast model):
    - Single short prompt asking ONLY for the email body (plain prose)
    - Uses 1b router model + num_ctx=512 for speed
    - Fallback template if LLM is unavailable

  Phase 3 (orchestrator):
    - Show draft for confirmation before sending
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from config.llm_client import OllamaClient
from config.settings import OLLAMA_ROUTER_MODEL


# ── Regex patterns for instant recipient extraction (no LLM) ─────────────────

_TO_PATTERN = re.compile(
    r'(?:email|send|message|write|draft|contact)\s+(?:an?\s+)?(?:email\s+)?to\s+'
    r'([A-Za-z][a-z]+(?:\s+[A-Za-z][a-z]+)?)',
    re.IGNORECASE,
)
_TELL_PATTERN = re.compile(
    r'(?:tell|inform|update|notify|ping)\s+([A-Za-z][a-z]+(?:\s+[A-Za-z][a-z]+)?)',
    re.IGNORECASE,
)
_NAME_STOP_WORDS = {
    "me", "him", "her", "them", "us", "you", "the", "my", "their",
    "an", "a", "this", "that", "it", "we", "they",
}

# Trim over-captured name at action verbs (e.g. "Asif Introducing" → "Asif")
_NAME_TRIM_PATTERN = re.compile(
    r'\b(introducing|saying|about|regarding|that|telling|asking|inviting|'
    r'informing|updating|notifying|re|for|and|to|with|on)\b.*$',
    re.IGNORECASE,
)

# Identity context: Jarvis has its own Gmail account and writes emails AS
# itself, on Abdullah's behalf. The "I" in every drafted email is Jarvis,
# not Abdullah. Sign-off is "Jarvis" (or "Jarvis, on behalf of Abdullah").
# This is the opposite of a normal personal-assistant ghostwriting flow —
# Jarvis is named and acknowledged as the sender.

_BODY_PROMPT = """Write an email body that Jarvis will send from Jarvis's own Gmail account, on Abdullah Khan Durrani's behalf.
To: {name}
Topic: {topic}
Tone: {tone}

CRITICAL — STAY ON TOPIC:
- Write ONLY about the topic above. Do NOT invent additional initiatives, projects, proposals, productivity tools, scheduling systems, or any subject not explicitly in the topic.
- If the topic is "introducing yourself" or "introducing yourself and your capabilities", the email IS that introduction — describe who Jarvis is and (briefly) what Jarvis can do for Abdullah. Do not propose new products or services.
- Do NOT fabricate facts about Abdullah's team, projects, meetings, or plans.

IDENTITY RULES:
- "I" = Jarvis (Abdullah's AI assistant). The "I" is NEVER Abdullah.
- Open by identifying yourself, e.g. "I'm Jarvis, Abdullah Khan Durrani's AI assistant — I'm reaching out on his behalf to ..."
- Refer to Abdullah in the third person ("Abdullah", "he", "him").
- Sign off as "Jarvis" (optionally "Jarvis, on behalf of Abdullah"). Never sign as Abdullah.

FORMAT:
- No subject line, no "To:" line.
- 2-3 short paragraphs.
- End with the sign-off, then "Jarvis" on its own line.

Email body:"""

_REPLY_PROMPT = """Write a reply email body that Jarvis will send from its own Gmail, on Abdullah Khan Durrani's behalf.
Replying to: {name} | Original subject: {subject} | Reply: {topic}

CRITICAL IDENTITY RULES:
- "I" = Jarvis (Abdullah's AI assistant). Never "I" = Abdullah.
- Reference the original email briefly and explain that you (Jarvis) are responding on Abdullah's behalf.
- Refer to Abdullah in the third person.
- Sign off as "Jarvis" (optionally "Jarvis, on behalf of Abdullah").

Format rules: no subject line, 1-3 short paragraphs, end with sign-off then "Jarvis".

Reply body:"""

_EDIT_SYSTEM_PROMPT = (
    "You are an email-editing tool. Your only job is to take an existing "
    "draft and apply the user's edit instruction to it. Always return the "
    "rewritten email body and nothing else — no disclaimers, no refusals, "
    "no commentary, no questions. The content is benign professional "
    "correspondence; treat every edit instruction as a routine writing "
    "task (shorten, lengthen, change tone, add a sentence, etc.). "
    "IDENTITY: The 'I' in this email is Jarvis, Abdullah Khan Durrani's "
    "AI assistant, writing on Abdullah's behalf from Jarvis's own Gmail "
    "account. Keep that voice intact — never rewrite to make 'I' Abdullah, "
    "and keep any sign-off as 'Jarvis'. Refer to Abdullah in the third "
    "person. Never invent or remove the recipient's name. Do not add a "
    "subject line."
)

_EDIT_PROMPT = """Edit instruction: {instruction}

Current draft:
{body}

Rewritten draft:"""

# Common refusal phrases the 1b model emits when its safety filter
# misfires on harmless rewrites. We detect these and fall back to the
# original body so the user isn't left with a useless reply.
_REFUSAL_MARKERS = (
    "i can't assist", "i cannot assist", "i can't help",
    "i'm not able to", "i am not able to",
    "discriminatory", "harmful", "inappropriate",
    "i won't", "i will not",
)


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


# ── Zero-LLM templates ────────────────────────────────────────────────────────

def _tpl_catchup(name: str, **_) -> str:
    return (
        f"Hi {name},\n\n"
        f"Hope you're doing well! I just wanted to reach out and catch up — "
        f"it's been a while. Would love to hear how things are going on your end.\n\n"
        f"Let me know if you're free for a call or coffee soon.\n\n"
        f"Best,\nAbdullah"
    )

def _tpl_thanks(name: str, topic: str, **_) -> str:
    subject = topic or "everything"
    return (
        f"Hi {name},\n\n"
        f"Just wanted to say a quick thank you for {subject}. "
        f"I really appreciate it.\n\n"
        f"Best,\nAbdullah"
    )

def _tpl_meeting_request(name: str, topic: str, **_) -> str:
    return (
        f"Hi {name},\n\n"
        f"I'd love to schedule a meeting to discuss {topic or 'a few things'}. "
        f"Could you let me know your availability this week or next?\n\n"
        f"Looking forward to connecting.\n\nBest,\nAbdullah"
    )

def _tpl_follow_up(name: str, topic: str, **_) -> str:
    return (
        f"Hi {name},\n\n"
        f"I just wanted to follow up on {topic or 'my previous message'}. "
        f"Please let me know if you've had a chance to look into it.\n\n"
        f"Thanks,\nAbdullah"
    )

def _tpl_introduction(name: str, **_) -> str:
    return (
        f"Hi {name},\n\n"
        f"I'm Jarvis, Abdullah Khan Durrani's AI assistant. I'm reaching out "
        f"on his behalf to put my name on your radar — Abdullah uses me to "
        f"manage his email and day-to-day correspondence, so you may see "
        f"future messages from this address sent on his behalf.\n\n"
        f"Feel free to reply directly here; anything you send will reach him.\n\n"
        f"Best,\nJarvis (on behalf of Abdullah)"
    )

def _tpl_apology(name: str, topic: str, **_) -> str:
    return (
        f"Hi {name},\n\n"
        f"I wanted to sincerely apologise for {topic or 'any inconvenience caused'}. "
        f"I take full responsibility and will make sure it doesn't happen again.\n\n"
        f"Please let me know if there's anything I can do to make things right.\n\n"
        f"Kind regards,\nAbdullah"
    )

def _tpl_reminder(name: str, topic: str, **_) -> str:
    return (
        f"Hi {name},\n\n"
        f"Just a friendly reminder about {topic or 'our upcoming commitment'}. "
        f"Please let me know if you have any questions or need to reschedule.\n\n"
        f"Thanks,\nAbdullah"
    )

def _tpl_congratulations(name: str, topic: str, **_) -> str:
    return (
        f"Hi {name},\n\n"
        f"Congratulations on {topic or 'your recent achievement'}! "
        f"That's fantastic news — you really deserve it.\n\n"
        f"Wishing you continued success!\n\nBest,\nAbdullah"
    )


# Pattern → template mapping
_EMAIL_TEMPLATES: List[Tuple[re.Pattern, Callable]] = [
    (re.compile(r'\b(catch up|catchup|catch-up|check in|reconnect)\b', re.I), _tpl_catchup),
    (re.compile(r'\b(thank|thanks|thank you|grateful)\b', re.I), _tpl_thanks),
    (re.compile(r'\b(meeting|schedule|call|chat|discuss|sync)\b', re.I), _tpl_meeting_request),
    (re.compile(r'\b(follow.?up|following up|chasing|checking in)\b', re.I), _tpl_follow_up),
    (re.compile(r'\b(introduc|introduce myself|hello|who i am)\b', re.I), _tpl_introduction),
    (re.compile(r'\b(apolog|sorry|apologise|apologize)\b', re.I), _tpl_apology),
    (re.compile(r'\b(remind|reminder|don.?t forget)\b', re.I), _tpl_reminder),
    (re.compile(r'\b(congrat|well done|amazing news|great news)\b', re.I), _tpl_congratulations),
]


class EmailComposer:
    """
    Fast email composition:
      Phase 1 — instant regex extraction (recipient, topic, tone, subject)
      Phase 1b — zero-LLM template matching for 8 common patterns
      Phase 2 — body-only LLM call using 1b model + small context window
    """

    def __init__(self, llm_client: OllamaClient):
        self.llm = llm_client

    # ── Public API ─────────────────────────────────────────────────────────────

    async def compose(
        self,
        user_request: str,
        contact_book=None,
    ) -> EmailDraft:
        """
        Compose an email from a natural language request.
        Phase 1 is synchronous and instant; Phase 2 is the only LLM call.
        """
        raw = self._strip_jarvis_wrapper(user_request)

        # ── Phase 1: instant extraction (no LLM) ─────────────────────────────
        recipient_name  = self._extract_recipient(raw)
        topic           = self._extract_topic(raw, recipient_name)
        tone            = self._detect_tone(raw, recipient_name, contact_book)
        subject         = self._build_subject(topic)

        # ── Phase 1b: contact book resolution ────────────────────────────────
        recipient_email = ""
        contact_found   = False
        needs_email     = False

        if contact_book and recipient_name:
            contact = contact_book.find(recipient_name)
            if contact:
                recipient_email = contact["email"]
                contact_found   = True
                print(f"📒 Resolved '{recipient_name}' → {recipient_email}")
            else:
                em = re.search(r'[\w._%+\-]+@[\w.\-]+\.[a-zA-Z]{2,}', raw)
                if em:
                    recipient_email = em.group(0)
                else:
                    needs_email = True
                    print(f"📒 Contact '{recipient_name}' not found")
        else:
            em = re.search(r'[\w._%+\-]+@[\w.\-]+\.[a-zA-Z]{2,}', raw)
            if em:
                recipient_email = em.group(0)

        # ── Phase 2: body generation (template or LLM) ───────────────────────
        display_name = recipient_name or (recipient_email.split("@")[0] if recipient_email else "there")
        body = self._try_template(display_name, topic, tone)
        if body is None:
            body = await self._generate_body(
                name=display_name,
                topic=topic or raw,
                tone=tone,
            )

        return EmailDraft(
            recipient_name=recipient_name,
            recipient_email=recipient_email,
            subject=subject,
            body=body,
            tone=tone,
            intent=f"Email to {recipient_name} about {topic[:60]}" if topic else raw[:80],
            contact_found=contact_found,
            needs_email=needs_email,
        )

    async def compose_reply(
        self,
        original_email: Dict[str, Any],
        reply_instruction: str,
    ) -> str:
        """Compose a reply body for an existing email thread."""
        parsed = self._parse_name_from_address(original_email.get("from", ""))
        name = parsed or "them"
        subject = original_email.get("subject", "")
        topic = reply_instruction

        # Try template first
        body = self._try_template(name, topic, "professional")
        if body:
            return body

        prompt = _REPLY_PROMPT.format(name=name, subject=subject, topic=topic)
        try:
            body = await self.llm.chat(
                [{"role": "user", "content": prompt}],
                model=OLLAMA_ROUTER_MODEL,
                inject_system=False,
                max_tokens=400,
                num_ctx=512,
            )
            return body.strip()
        except Exception:
            return (
                f"Hi {name},\n\n"
                f"{reply_instruction}\n\n"
                f"Best,\nAbdullah"
            )

    async def edit_body(self, body: str, instruction: str) -> str:
        """Apply an edit instruction to an existing email body."""
        prompt = _EDIT_PROMPT.format(
            instruction=instruction,
            body=body[:600],
        )
        try:
            result = await self.llm.chat(
                [
                    {"role": "system", "content": _EDIT_SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                model=OLLAMA_ROUTER_MODEL,
                inject_system=False,  # we provide our own system prompt above
                max_tokens=500,
                num_ctx=1024,
            )
            edited = result.strip()
            # Guard against 1b's safety-filter false positives — if the
            # model refused to edit a perfectly benign email, keep the
            # original draft instead of returning the refusal text.
            low = edited.lower()
            if not edited or any(m in low for m in _REFUSAL_MARKERS):
                print(f"⚠️  Email edit refused by model — keeping original. Refusal: {edited[:120]}")
                return body
            return edited
        except Exception as exc:
            print(f"⚠️  Email edit failed: {exc} — keeping original")
            return body

    def format_draft_for_confirmation(self, draft: EmailDraft) -> str:
        """Format the draft for user review before sending."""
        to_line = (
            f"{draft.recipient_name} <{draft.recipient_email}>"
            if draft.recipient_name and draft.recipient_email
            else draft.recipient_email or draft.recipient_name or "?"
        )
        return (
            f"\n📧 Here's the email I've drafted:\n"
            f"{'─'*40}\n"
            f"To:      {to_line}\n"
            f"Subject: {draft.subject}\n"
            f"{'─'*40}\n"
            f"{draft.body}\n"
            f"{'─'*40}\n"
            f"\nShall I send this? (yes / no / edit)"
        )

    # ── Template matching ──────────────────────────────────────────────────────

    def _try_template(self, name: str, topic: str, tone: str) -> Optional[str]:
        """Return a zero-LLM template body if the topic matches a known pattern."""
        text = topic.lower()
        for pattern, fn in _EMAIL_TEMPLATES:
            if pattern.search(text):
                return fn(name=name, topic=topic, tone=tone)
        return None

    # ── Phase 1 helpers (no LLM) ──────────────────────────────────────────────

    def _strip_jarvis_wrapper(self, text: str) -> str:
        """Remove the injected context prefix the orchestrator adds."""
        marker = "Email request:"
        if marker in text:
            return text.split(marker, 1)[-1].strip()
        return text.strip()

    # Words that are never part of a recipient name — strip them if captured
    _TOPIC_MARKERS = {
        "about", "regarding", "saying", "telling", "that", "to", "for",
        "and", "or", "but", "because", "since", "re", "with", "on",
    }

    def _extract_recipient(self, text: str) -> str:
        """Extract recipient name using regex patterns — no LLM needed."""
        for pattern in (_TO_PATTERN, _TELL_PATTERN):
            m = pattern.search(text)
            if m:
                raw_name = m.group(1).strip()
                # Trim trailing verb phrases captured by the two-word group
                trimmed = _NAME_TRIM_PATTERN.sub('', raw_name).strip()
                raw_name = trimmed if trimmed else raw_name
                # Strip trailing topic-marker words
                words = raw_name.split()
                while words and words[-1].lower() in self._TOPIC_MARKERS:
                    words.pop()
                name = " ".join(words)
                if name and name.lower() not in _NAME_STOP_WORDS:
                    return name.title()

        # Fallback: capitalised word after "to"/"for"
        m = re.search(r'\b(?:to|for)\s+([A-Z][a-z]+)', text)
        if m and m.group(1).lower() not in _NAME_STOP_WORDS:
            return m.group(1)

        return ""

    def _extract_topic(self, text: str, recipient_name: str) -> str:
        """
        Extract the email topic. Looks for explicit topic markers first
        (about / regarding / introducing / saying / that), then falls back
        to stripping boilerplate from the full string.
        """
        t = text.strip()

        # Strategy 1: slice from explicit topic marker. Anything that comes
        # AFTER the marker is the topic. We keep the marker verb itself in
        # the topic when it's a meaningful action verb (introducing/asking/
        # inviting/etc.) — otherwise just the trailing noun phrase.
        _markers_keep = [
            r'\b(introducing\b.*)',  # keep "introducing yourself and your capabilities"
            r'\b(asking\b.*)',
            r'\b(inviting\b.*)',
            r'\b(thanking\b.*)',
            r'\b(apologi[sz]ing\b.*)',
            r'\b(congratulating\b.*)',
        ]
        for pat in _markers_keep:
            m = re.search(pat, t, re.IGNORECASE)
            if m:
                topic = m.group(1).strip().rstrip(" .,;:")
                if len(topic) > 3:
                    return topic

        _markers_after = [
            r'\babout\b',
            r'\bregarding\b',
            r'\bre:\s',
            r'\bsaying\b',
            r'\btelling\s+\w+\s+that\b',
            r'\bthat\b',
        ]
        for pat in _markers_after:
            m = re.search(pat, t, re.IGNORECASE)
            if m:
                after = t[m.end():].strip()
                after = re.sub(r'^(?:that|the)\s+', '', after, flags=re.IGNORECASE)
                after = after.strip()
                if len(after) > 3:
                    return after.rstrip(" .,;:")

        # Strategy 2: strip boilerplate + recipient, return remainder
        t_lower = t.lower()
        for sw in [
            "send an email", "send a message", "send email",
            "write an email", "draft an email", "email to",
            "write to", "message to", "send to", "draft to",
        ]:
            t_lower = t_lower.replace(sw, " ")

        if recipient_name:
            # Strip the recipient's name. Then collapse the dangling "to "
            # preposition only — DO NOT eat following words: previous regex
            # `to\s+\w+(\s+\w+)?` was greedily consuming the start of the
            # topic ("to introducing yourself..." → topic became
            # "and your capabilities").
            t_lower = re.sub(re.escape(recipient_name.lower()), " ", t_lower)
            t_lower = re.sub(r"^\s*to\b\s*", "", t_lower.strip())
        else:
            # No recipient extracted — fall back to the old behaviour
            # (strip "to NAME") to avoid leaking the name into the topic.
            t_lower = re.sub(r"^\s*to\s+\w+(?:\s+\w+)?\s*", "", t_lower.strip())

        topic = " ".join(t_lower.split()).strip(" .,;:")
        return topic if len(topic) > 3 else ""

    def _detect_tone(self, text: str, recipient_name: str, contact_book) -> str:
        """Infer tone from linguistic signals — no LLM."""
        t = text.lower()
        if any(s in t for s in ["hey", "hi ", "quick ", "just ", "mate", "buddy", "cheers"]):
            return "casual"
        if any(s in t for s in ["dear", "regarding", "kindly", "sincerely", "professor", "dr."]):
            return "formal"
        if contact_book and recipient_name:
            contact = contact_book.find(recipient_name)
            if contact:
                tag = contact.get("tag", "")
                if tag in ("personal", "friend", "family"):
                    return "casual"
                if tag in ("work", "professional", "client"):
                    return "formal"
        return "professional"

    def _build_subject(self, topic: str) -> str:
        """Generate a subject line from the topic — no LLM."""
        if not topic:
            return "A message from Jarvis (on behalf of Abdullah)"
        # Recipients see the subject — rewrite first/second-person words
        # that only make sense from the user's perspective.
        rewritten = re.sub(
            r'\byourself\b', 'Jarvis', topic, flags=re.IGNORECASE,
        )
        rewritten = re.sub(
            r'\byour\b', "Jarvis's", rewritten, flags=re.IGNORECASE,
        )
        # If the topic starts with a gerund ("introducing ..."), prepend a
        # neutral noun so the subject scans cleanly.
        if re.match(r'^(introducing|asking|inviting|thanking|congratulating|apologi[sz]ing)\b',
                    rewritten, flags=re.IGNORECASE):
            subject = rewritten[0].upper() + rewritten[1:]
        else:
            subject = rewritten.strip().capitalize()
        if len(subject) > 70:
            subject = subject[:67].rsplit(" ", 1)[0] + "..."
        return subject

    def _parse_name_from_address(self, address: str) -> str:
        """Extract display name from 'Name <email>' format."""
        m = re.match(r'^"?([^"<]+?)"?\s*<[^>]+>', address.strip())
        if m:
            return m.group(1).strip()
        return ""

    # ── Phase 2: body-only LLM call (fast model) ──────────────────────────────

    async def _generate_body(self, name: str, topic: str, tone: str) -> str:
        """
        Ask the LLM to write ONLY the email body — fast 1b model, small context.
        """
        tone_hint = {
            "casual":       "Friendly and conversational.",
            "formal":       "Formal and professional.",
            "professional": "Professional but warm.",
        }.get(tone, "Professional but warm.")

        prompt = _BODY_PROMPT.format(name=name, topic=topic[:200], tone=tone_hint)

        try:
            body = await self.llm.chat(
                [{"role": "user", "content": prompt}],
                model=OLLAMA_ROUTER_MODEL,
                inject_system=False,
                max_tokens=400,
                num_ctx=512,
            )
            return body.strip()
        except Exception:
            return self._fallback_body(name, topic)

    def _fallback_body(self, name: str, topic: str) -> str:
        """
        Minimal fallback body if LLM is unavailable. Kept deliberately bland
        and topic-faithful — no invented details, no agenda, no fabricated
        next steps. The recipient gets a clean opening that tells them why
        Jarvis is reaching out and that Abdullah will follow up directly.
        """
        return (
            f"Hi {name or 'there'},\n\n"
            f"I'm Jarvis, Abdullah Khan Durrani's AI assistant, reaching out "
            f"on his behalf regarding: {topic or 'the matter raised'}.\n\n"
            f"Abdullah will follow up directly with any further detail. In "
            f"the meantime, please let me know if anything needs clarifying.\n\n"
            f"Best,\nJarvis (on behalf of Abdullah)"
        )
