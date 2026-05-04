"""
LLM_SQL.py — Hybrid financial Q&A engine
All 8 improvements implemented:
1. Derived field fallbacks for L1
2. Hallucination guard — available fields passed to every prompt
3. L2 formatting enforcement
4. Mistral extraction retry logic (in extract_pdf.py)
5. Performance: financial context cached per company
6. Multi-document isolation fix
7. L3 comprehensive formula library
8. Unit normalization awareness
"""

import re
import requests
from finex.db_query import run_query, get_financial_context, get_all_years, format_result
from finex.chroma_store import pdf_store

# ── Global stores ──────────────────────────────────────────────────────────────
_context_cache  = {}  # Fix 5: cache financial context per company

def store_pdf_text(company: str, text: str, filename: str = ""):
    """Persist PDF text to disk (survives server restarts)."""
    pdf_store.save(company, filename, text)
    # Invalidate context cache when new data uploaded
    if company in _context_cache:
        del _context_cache[company]

def get_pdf_text(company: str) -> str:
    """Retrieve PDF text from persistent store."""
    return pdf_store.load(company)

def search_pdf_text(company: str, query: str, n: int = 5):
    """Semantic search within the company PDF (uses ChromaDB if available)."""
    return pdf_store.search(company, query, n)

def get_cached_context(company: str) -> str:
    # Fix 5: return cached context, rebuild if stale
    if company not in _context_cache:
        _context_cache[company] = get_financial_context(company)
    return _context_cache[company]

def invalidate_cache(company: str):
    if company in _context_cache:
        del _context_cache[company]


# ── Ollama wrapper (HTTP, not subprocess — saves 3-5s per call) ───────────────

_OLLAMA_URL = "http://localhost:11434/api/chat"
_FINEX_MODEL = "llama3.2:latest"


def ask_llm(prompt: str, system: str = "") -> str:
    """
    Send a prompt to the local Ollama server via HTTP POST.
    Uses the REST API directly — no subprocess overhead.
    Falls back gracefully if Ollama is unreachable.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        resp = requests.post(
            _OLLAMA_URL,
            json={
                "model": _FINEX_MODEL,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 512},
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        return "[Ollama is not running. Please start Ollama and try again.]"
    except Exception as e:
        return f"[LLM error: {e}]"


# ── Fix 2: Hallucination guard ─────────────────────────────────────────────────

def get_available_fields(company: str, year: int = None) -> dict:
    """
    Returns dict of {field_name: value} for fields that actually have data.
    Used to prevent LLM from referencing fields that are NULL.
    """
    years = get_all_years(company)
    if not years:
        return {}
    target_year = year or years[0]

    result = run_query(f"SELECT * FROM financials WHERE company = '{company}' AND year = {target_year} LIMIT 1")
    if "error" in result or not result["rows"]:
        return {}

    cols = result["columns"]
    row = result["rows"][0]
    record = dict(zip(cols, row))

    # Only return non-null numeric fields
    return {k: v for k, v in record.items()
            if v is not None and isinstance(v, float) and k not in ("id",)}


def fields_summary(available: dict) -> str:
    """Format available fields as a clear list for the LLM."""
    if not available:
        return "No financial data available."
    lines = ["Available data fields (PKR thousands):"]
    for k, v in available.items():
        lines.append(f"  {k.replace('_', ' ').title()}: PKR {v:,.0f}")
    return "\n".join(lines)


# ── Fix 1: Derived field calculator ───────────────────────────────────────────

DERIVED_FIELDS = {
    "total_assets": lambda d: (
        (d.get("current_assets", 0) or 0) + (d.get("non_current_assets", 0) or 0)
        if d.get("current_assets") and d.get("non_current_assets") else None
    ),
    "total_liabilities": lambda d: (
        (d.get("current_liabilities", 0) or 0) + (d.get("non_current_liabilities", 0) or 0)
        if d.get("current_liabilities") and d.get("non_current_liabilities") else None
    ),
    "gross_profit": lambda d: (
        (d.get("revenue", 0) or 0) - (d.get("cost_of_goods_sold", 0) or 0)
        if d.get("revenue") and d.get("cost_of_goods_sold") else None
    ),
    "operating_profit": lambda d: (
        (d.get("gross_profit", 0) or 0) - (d.get("operating_expenses", 0) or 0)
        if d.get("gross_profit") and d.get("operating_expenses") else None
    ),
    "net_profit": lambda d: (
        (d.get("profit_before_tax", 0) or 0) - (d.get("tax_expense", 0) or 0)
        if d.get("profit_before_tax") and d.get("tax_expense") else None
    ),
}


def enrich_with_derived(data: dict) -> dict:
    """Add derived fields where primary fields are missing but can be calculated."""
    enriched = dict(data)
    for field, formula in DERIVED_FIELDS.items():
        if enriched.get(field) is None:
            derived = formula(enriched)
            if derived is not None:
                enriched[field] = derived
                enriched[f"_{field}_derived"] = True  # flag as derived
    return enriched


# ── Deterministic Python router ────────────────────────────────────────────────

DETAIL_PHRASES = [
    "more detail", "elaborate", "explain more", "expand on", "go deeper",
    "tell me more", "provide more", "break it down", "drill down", "dig deeper"
]

OFF_TOPIC_WORDS = [
    "weather", "recipe", "cook", "sport", "football", "cricket", "movie",
    "song", "music", "celebrity", "politics", "war", "travel", "holiday",
    "joke", "story", "poem", "code", "programming", "python", "javascript"
]

TEXT_PHRASES = [
    "who is", "who are", "who signed", "who audited", "board of directors",
    "company secretary", "chief executive", "managing director", "cfo",
    "registered office", "plant location", "product portfolio", "csr",
    "corporate social responsibility", "environment", "water conservation",
    "alternative energy", "future outlook", "dividend policy", "auditor",
    "statutory auditor", "banker", "legal advisor", "incorporation",
    "subsidiaries", "related party", "shariah", "notes to", "accounting policy",
    "basis of preparation", "ifrs", "ias 34", "contingencies", "commitments"
]

L6_PHRASES = [
    "long-term", "long term", "growth strategy", "strategic risk",
    "forecast", "predict next", "next period", "next year revenue",
    "executive briefing", "executive summary", "capital allocation",
    "acquisitions", "expansion", "bearish", "bullish", "industry position",
    "compare to industry", "compare to typical", "prioritise", "prioritize",
    "12 months", "going forward", "summarise overall", "summarize overall",
    "what should management", "how does this company compare",
    "is the company's capital", "what would a"
]

L5_PHRASES = [
    "attractive to investor", "worth investing", "would you invest",
    "financially healthy", "overall health", "overall assessment",
    "strengths and weakness", "strengths based", "weaknesses visible",
    "stakeholders be concerned", "dividend reflect", "overvalued",
    "undervalued", "growth-oriented", "deploying capital efficiently",
    "key metrics would", "should i invest", "investment decision"
]
L5_WORDS = ["attractive", "overvalued", "undervalued", "stakeholder"]

L4_PHRASES = [
    "why did", "why has", "why might", "why is", "why are",
    "what caused", "what drove", "despite", "even though",
    "major contributor", "becoming more leveraged", "liquidity improving",
    "liquidity worsening", "efficiently managing", "gross margin trend",
    "how sustainable", "dependent on debt", "financial risk",
    "identify risk", "what are the risk", "cost pressure",
    "operating expense", "what does the trend"
]
L4_WORDS = ["why", "despite", "leveraged", "liquidity", "sustainable", "risks", "efficiency"]

L3_PHRASES = [
    "gross profit margin", "net profit margin", "profit margin",
    "return on asset", "return on equity", "roa", "roe",
    "debt-to-equity", "debt to equity", "current ratio", "quick ratio",
    "asset turnover", "operating margin", "eps growth", "percentage of revenue",
    "calculate", "what is the ratio", "what is the margin"
]
L3_WORDS = ["margin", "ratio", "roa", "roe", "turnover ratio"]

L2_PHRASES = [
    "compare", "compared to", "last year", "prior year", "previous year",
    "year-over-year", "year on year", "2024 and 2025", "2025 and 2024",
    "changed from", "how much did", "increase from", "decrease from",
    "higher than last", "lower than last", "versus last", "vs last",
    "half year ended", "same period last year", "which increased more",
    "which decreased more"
]
L2_WORDS = ["compare", "comparison", "versus", "vs", "change", "grew", "fell", "declined", "improved"]

L1_PHRASES = [
    "what is the", "what are the", "what was the", "how much is",
    "how much was", "how much cash", "what is revenue", "what is profit",
    "what is eps", "total revenue", "total assets", "total liabilities",
    "net profit", "gross profit", "share capital", "cash balance",
    "finance cost", "earnings per share", "profit after tax",
    "profit before tax", "operating profit", "net turnover", "gross turnover"
]
L1_WORDS = [
    "revenue", "profit", "assets", "liabilities", "cash", "eps",
    "turnover", "dividend", "depreciation", "tax", "ebit", "ebitda"
]


def route_question(question: str, history: str = "") -> str:
    q = question.lower().strip()

    if any(p in q for p in DETAIL_PHRASES):
        return "DETAIL"

    if any(w in q for w in OFF_TOPIC_WORDS):
        financial_words = ["revenue", "profit", "asset", "liability", "cash", "cement",
                           "company", "financial", "report", "turnover", "margin"]
        if not any(fw in q for fw in financial_words):
            return "OFF_TOPIC"

    if any(p in q for p in L6_PHRASES):
        return "L6"

    if any(p in q for p in L5_PHRASES):
        return "L5"
    if any(w in q for w in L5_WORDS) and any(w in q for w in ["investor", "invest", "attractive", "health"]):
        return "L5"

    if any(p in q for p in L4_PHRASES):
        return "L4"
    if any(w in q for w in L4_WORDS):
        return "L4"

    if any(p in q for p in L3_PHRASES):
        return "L3"
    if any(w in q for w in L3_WORDS):
        return "L3"

    if any(p in q for p in L2_PHRASES):
        return "L2"
    year_matches = re.findall(r"\b(20\d{2})\b", q)
    if len(year_matches) >= 2:
        return "L2"

    if any(p in q for p in TEXT_PHRASES):
        return "TEXT"

    if any(p in q for p in L1_PHRASES):
        return "L1"
    if any(w in q for w in L1_WORDS):
        return "L1"

    if re.search(r"\b20\d{2}\b", q):
        return "L1"

    return "L1"


def is_detail_request(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in DETAIL_PHRASES)


# ── SQL generation ─────────────────────────────────────────────────────────────

FULL_SCHEMA = """
Table: financials
Columns: id, company, year, period,
  revenue, gross_profit, operating_profit, profit_before_tax, net_profit,
  eps, dividend_per_share,
  cost_of_goods_sold, operating_expenses, depreciation, finance_cost, tax_expense,
  total_assets, current_assets, non_current_assets, cash_balance,
  trade_receivables, inventory,
  total_liabilities, current_liabilities, non_current_liabilities,
  total_equity, share_capital, long_term_debt,
  operating_cashflow, investing_cashflow, financing_cashflow
"""


def generate_sql(question: str, company: str = "Bestway Cement") -> str:
    years = get_all_years(company)
    latest_year = years[0] if years else 2025

    prompt = f"""You are a PostgreSQL expert converting finance questions to SQL.

{FULL_SCHEMA}

RULES:
- Return ONLY executable SQL, no explanation, no markdown, no backticks
- Company is '{company}' unless stated otherwise
- Latest available year is {latest_year}
- For comparisons, SELECT both years in one query
- Use ORDER BY year DESC for multi-year queries
- If question asks for ratio/calculation, SELECT the raw columns needed

Question: {question}

SQL:"""
    raw = ask_llm(prompt)
    return clean_sql(raw)


def clean_sql(sql: str) -> str:
    sql = sql.strip()
    sql = re.sub(r"```sql|```", "", sql, flags=re.IGNORECASE).strip()
    match = re.search(r"(SELECT\s.+?)(?:;|$)", sql, re.IGNORECASE | re.DOTALL)
    if match:
        sql = match.group(1).strip() + ";"
    return sql


# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_ANALYST = """You are a financial analyst. Answer directly and concisely.
STRICT FORMATTING RULES:
- Never introduce yourself or mention your role
- Never start with phrases like "As a financial analyst" or "I conclude that"
- Never use * or - as bullet points. Use numbered lists (1. 2. 3.) only when listing multiple items
- Currency is always PKR, never use dollar signs
- Only use data explicitly provided to you. NEVER fabricate or estimate figures not in the data
- If a field is not available in the data provided, say clearly "this data is not available"
- Keep answers to 3-5 lines unless user asks for more detail
- Lead directly with the answer, no preamble
- All monetary figures are in PKR thousands unless stated otherwise"""


# ── Level handlers ─────────────────────────────────────────────────────────────

def handle_l1(question: str, company: str, history: str = "") -> str:
    years = get_all_years(company)
    if not years:
        return "No data found for this company."

    year_match = re.search(r"\b(20\d{2})\b", question)
    target_year = int(year_match.group(1)) if year_match else years[0]

    result = run_query(f"""
        SELECT * FROM financials
        WHERE company = '{company}' AND year = {target_year}
        LIMIT 1
    """)

    if "error" in result or not result["rows"]:
        available = ", ".join(str(y) for y in years)
        return f"No data found for {target_year}. Available periods: {available}"

    cols = result["columns"]
    row = result["rows"][0]
    record = dict(zip(cols, row))

    # Fix 1: enrich with derived fields
    enriched = enrich_with_derived(record)

    year = enriched.get("year", target_year)
    period = enriched.get("period", f"FY {year}")

    # Fix 2: only show fields that have actual data
    available = get_available_fields(company, target_year)
    # Add derived fields to available
    for f, formula in DERIVED_FIELDS.items():
        if f not in available:
            derived = formula(record)
            if derived is not None:
                available[f] = derived

    if not available:
        return "No financial data found for this period."

    data_lines = [f"{company} — {period} (PKR thousands):"]
    for col, val in available.items():
        data_lines.append(f"  {col.replace('_', ' ').title()}: PKR {val:,.0f}")

    data_text = "\n".join(data_lines)

    prompt = f"""{data_text}

Question: {question}

Answer in exactly 1 sentence using only the figures shown above.
State the value in PKR thousands and specify the period.
If the exact figure is not listed above, say it is not available — do not guess."""

    return ask_llm(prompt, system=SYSTEM_ANALYST)


def handle_l2(question: str, company: str, history: str = "") -> str:
    # Fix 3: fetch all years directly — no LLM SQL generation
    years = get_all_years(company)
    if not years:
        return "No financial data found."

    year_filter = "(" + ", ".join(str(y) for y in years) + ")"
    result = run_query(f"""
        SELECT year, period, revenue, gross_profit, operating_profit,
               profit_before_tax, net_profit, eps, finance_cost,
               depreciation, total_assets, total_liabilities,
               total_equity, cash_balance, current_assets, non_current_assets,
               current_liabilities, non_current_liabilities
        FROM financials
        WHERE company = '{company}' AND year IN {year_filter}
        ORDER BY year DESC
    """)

    if "error" in result or not result["rows"]:
        return "No comparative data found."

    data_text = format_result(result)

    prompt = f"""Financial data (PKR thousands):
{data_text}

Question: {question}

STRICT RULES:
- Currency is PKR only
- Only reference years present in the data above — never mention years not shown
- State values for BOTH periods being compared
- Always calculate and state the percentage change (e.g. increased by 12.3%)
- State clearly whether the change is an increase or decrease
- Answer in 2-3 sentences maximum

End your response with exactly this line: "Need more detail? Just ask." """

    return ask_llm(prompt, system=SYSTEM_ANALYST)


def handle_l3(question: str, company: str, history: str = "") -> str:
    # Fix 2 + 7: use cached context + comprehensive formula library
    context = get_cached_context(company)

    # Get available fields for hallucination guard
    years = get_all_years(company)
    available_note = ""
    if years:
        available = get_available_fields(company, years[0])
        missing = [f for f in ["revenue", "gross_profit", "operating_profit", "net_profit",
                                "total_assets", "total_liabilities", "total_equity",
                                "current_assets", "current_liabilities", "inventory"]
                   if f not in available]
        if missing:
            available_note = f"\nNote: These fields are NOT available in the data: {', '.join(missing)}"

    prompt = f"""Financial data (all figures in PKR thousands):
{context}{available_note}

Question: {question}

RATIO FORMULAS — use the correct standard formula:
- Gross Profit Margin     = (Gross Profit / Revenue) x 100
- Net Profit Margin       = (Net Profit / Revenue) x 100
- Operating Margin        = (Operating Profit / Revenue) x 100
- ROA                     = (Net Profit / Total Assets) x 100
- ROE                     = (Net Profit / Total Equity) x 100
- Debt-to-Equity          = Total Liabilities / Total Equity
- Current Ratio           = Current Assets / Current Liabilities
- Quick Ratio             = (Current Assets - Inventory) / Current Liabilities
- Asset Turnover          = Revenue / Total Assets
- EPS Growth %            = ((Current EPS - Prior EPS) / Prior EPS) x 100
- Finance Cost % Revenue  = (Finance Cost / Revenue) x 100
- Total Assets            = Non-Current Assets + Current Assets
- Total Liabilities       = Non-Current Liabilities + Current Liabilities
- Total Equity            = Total Assets - Total Liabilities

RULES:
- If a required value is missing from the data, state it clearly — do not estimate
- All monetary values are in PKR thousands

Respond in this exact format:
Formula: [standard formula used]
Values: [exact figures from data with periods]
Calculation: [step by step working]
Result: [final answer with correct units — % or ratio or PKR thousands]
Interpretation: [1 sentence in context of this company]

End your response with exactly this line: "Need more detail? Just ask." """

    return ask_llm(prompt, system=SYSTEM_ANALYST)


def handle_l4(question: str, company: str, history: str = "") -> str:
    context = get_cached_context(company)

    prompt = f"""Financial data (PKR thousands):
{context}

{f"Previous conversation:{chr(10)}{history}{chr(10)}" if history else ""}
Question: {question}

Answer in 3-4 lines. Lead with the conclusion, support with 1-2 specific figures.
Only reference data explicitly shown above. No numbered lists unless listing multiple causes.

End your response with exactly this line: "Need more detail? Just ask." """

    return ask_llm(prompt, system=SYSTEM_ANALYST)


def handle_l5(question: str, company: str, history: str = "") -> str:
    context = get_cached_context(company)

    prompt = f"""Financial data (PKR thousands):
{context}

{f"Previous conversation:{chr(10)}{history}{chr(10)}" if history else ""}
Question: {question}

Answer in 4-5 lines. Lead with a clear verdict supported by the most relevant metrics.
Only use data explicitly shown above.

End your response with exactly this line: "Need more detail? Just ask." """

    return ask_llm(prompt, system=SYSTEM_ANALYST)


def handle_text(question: str, company: str, history: str = "") -> str:
    # Fix 6: look up PDF text by exact company name
    # Use semantic search for relevant chunks if available, else fall back to full text
    chunks = search_pdf_text(company, question, n=5)

    if not chunks:
        # Try full text as last resort
        pdf_text = get_pdf_text(company)
        if not pdf_text:
            return "The PDF text is not available. Please re-upload the report."
        chunks = [pdf_text[:8000]]

    context = "\n\n---\n\n".join(chunks)

    prompt = f"""Report text from {company}'s financial report:

{context}

{f"Previous conversation:{chr(10)}{history}{chr(10)}" if history else ""}
Question: {question}

Answer using only information from the report text above.
Be concise (3-5 lines). If the answer is not in the text, say so clearly.

End your response with exactly this line: "Need more detail? Just ask." """

    return ask_llm(prompt, system=SYSTEM_ANALYST)


def handle_detail(question: str, company: str, history: str = "") -> str:
    context = get_cached_context(company)
    # Use semantic search for the most relevant snippet
    chunks = search_pdf_text(company, question, n=3)
    text_snippet = "\n\n".join(chunks)[:2000] if chunks else ""

    prompt = f"""Financial data (PKR thousands):
{context}

Report text excerpt:
{text_snippet}

Previous conversation:
{history}

The user wants more detail on the previous answer.
Provide a thorough expansion using specific numbers.
Use numbered points for clarity. Only reference data shown above."""

    return ask_llm(prompt, system=SYSTEM_ANALYST)


def handle_l6(question: str, company: str, history: str = "") -> str:
    context = get_cached_context(company)
    # Semantic search for strategic context from PDF
    chunks = search_pdf_text(company, question, n=4)
    text_snippet = "\n\n".join(chunks)[:4000] if chunks else ""

    prompt = f"""Financial data (PKR thousands):
{context}

Report context:
{text_snippet}

{f"Previous conversation:{chr(10)}{history}{chr(10)}" if history else ""}
Question: {question}

Answer directly — no introduction, no self-reference.
Lead with the strategic conclusion. Support with 1-2 specific figures from the data.
Use numbered points only if listing multiple recommendations. Maximum 6 lines.

End your response with exactly this line: "Need more detail? Just ask." """

    return ask_llm(prompt, system=SYSTEM_ANALYST)


def handle_off_topic() -> str:
    return "I can only answer questions related to the financial report and company data. Please ask a finance-related question."


# ── Main dispatcher ────────────────────────────────────────────────────────────

LEVEL_LABELS = {
    "L1": "Basic Retrieval",
    "L2": "Comparative",
    "L3": "Ratio Analysis",
    "L4": "Analytical Reasoning",
    "L5": "Investor Insight",
    "L6": "Strategic Reasoning",
    "TEXT": "Report Text",
    "DETAIL": "Detail Follow-up",
    "OFF_TOPIC": "Off Topic",
}

LEVEL_NUMBERS = {
    "L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5, "L6": 6,
    "TEXT": 3, "DETAIL": 4, "OFF_TOPIC": 1,
}


def answer(question: str, company: str = "Bestway Cement", history: str = "") -> tuple:
    category = route_question(question, history)
    print(f"  [Routed to: {category}]")

    if category == "OFF_TOPIC":
        return handle_off_topic(), 1, LEVEL_LABELS["OFF_TOPIC"]
    elif category == "DETAIL":
        return handle_detail(question, company, history), 4, LEVEL_LABELS["DETAIL"]
    elif category == "TEXT":
        return handle_text(question, company, history), 3, LEVEL_LABELS["TEXT"]
    elif category == "L1":
        return handle_l1(question, company, history), 1, LEVEL_LABELS["L1"]
    elif category == "L2":
        return handle_l2(question, company, history), 2, LEVEL_LABELS["L2"]
    elif category == "L3":
        return handle_l3(question, company, history), 3, LEVEL_LABELS["L3"]
    elif category == "L4":
        return handle_l4(question, company, history), 4, LEVEL_LABELS["L4"]
    elif category == "L5":
        return handle_l5(question, company, history), 5, LEVEL_LABELS["L5"]
    elif category == "L6":
        return handle_l6(question, company, history), 6, LEVEL_LABELS["L6"]
    else:
        return handle_l1(question, company, history), 1, LEVEL_LABELS["L1"]


if __name__ == "__main__":
    company = "Bestway Cement"
    print(f"Financial Analyst — {company}")
    history = ""

    while True:
        question = input("Question: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue
        response, level, label = answer(question, company, history)
        print(f"\n[{label}]\n{response}\n")
        history += f"User: {question}\nAnalyst: {response}\n\n"
        history_lines = history.strip().split("\n")
        if len(history_lines) > 12:
            history = "\n".join(history_lines[-12:])