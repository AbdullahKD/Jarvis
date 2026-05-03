"""
extract_pdf.py — Intelligent PDF extraction using Mistral:7b + deterministic validation
"""

import json
import re
import subprocess
import pdfplumber


# ── Model config ───────────────────────────────────────────────────────────────
EXTRACTION_MODEL = "mistral:7b"
FALLBACK_MODEL   = "llama3:8b"


# ── Ollama call ────────────────────────────────────────────────────────────────

def call_ollama(prompt: str, model: str = EXTRACTION_MODEL) -> str:
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    return result.stdout.decode().strip()


# ── PDF text extraction ────────────────────────────────────────────────────────

def extract_text_only(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def extract_pages(pdf_path: str) -> list:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and len(text.strip()) > 50:
                pages.append({"page": i + 1, "text": text})
    return pages


# ── Page classifier ────────────────────────────────────────────────────────────

FINANCIAL_KEYWORDS = [
    # Revenue & Sales
    "turnover", "revenue", "net revenue", "gross revenue", "net sales", "gross sales",
    "net turnover", "gross turnover", "total revenue", "total sales", "sales revenue",
    "operating revenue", "service revenue", "fee income", "interest income",
    "rental income", "dividend income", "other income", "income from operations",
    "rebates", "discounts", "excise duty", "sales tax",

    # Profitability
    "gross profit", "gross loss", "operating profit", "operating loss",
    "profit from operations", "profit before tax", "profit after tax",
    "profit before taxation", "profit for the period", "profit for the year",
    "net profit", "net loss", "loss for the period", "loss before tax",
    "ebit", "ebitda", "operating income", "operating margin",
    "profit attributable", "earnings", "retained earnings", "retained profit",
    "comprehensive income", "total comprehensive income",

    # Costs & Expenses
    "cost of sales", "cost of goods sold", "cost of revenue", "cost of services",
    "cost of production", "direct costs", "indirect costs",
    "gross expenses", "operating expenses", "total expenses",
    "selling expenses", "selling and distribution", "distribution expenses",
    "administrative expenses", "general and administrative", "general expenses",
    "other operating expenses", "other expenses", "miscellaneous expenses",
    "research and development", "r&d expenses",
    "depreciation", "amortisation", "amortization", "depletion",
    "impairment", "impairment loss", "write-off", "write-down",
    "provision", "provisions", "allowance", "bad debt",
    "finance cost", "finance costs", "financial charges", "interest expense",
    "interest charges", "mark-up", "markup expense", "borrowing costs",
    "exchange loss", "exchange gain", "foreign exchange",
    "royalty", "royalties", "licence fee",
    "insurance", "repair and maintenance", "utilities", "rent expense",
    "employee benefits", "staff costs", "salaries", "wages", "remuneration",
    "gratuity", "provident fund", "pension",
    "taxation", "income tax", "income tax expense", "deferred tax",
    "current tax", "tax charge", "withholding tax",

    # Balance Sheet — Assets
    "total assets", "net assets", "non-current assets", "current assets",
    "fixed assets", "property, plant", "plant and equipment", "ppe",
    "capital work in progress", "cwip", "right of use", "lease asset",
    "intangible assets", "goodwill", "intangibles",
    "investment property", "long term investments", "long-term investments",
    "investments in associates", "investments in subsidiaries",
    "equity accounted", "available for sale", "held to maturity",
    "long term loans", "long term deposits", "long-term deposits",
    "deferred tax asset", "advance tax", "tax refundable",
    "stock in trade", "inventories", "inventory", "stores and spares",
    "spare parts", "work in progress", "finished goods", "raw materials",
    "trade debts", "trade receivables", "accounts receivable",
    "other receivables", "advances", "prepayments", "deposits",
    "short term investments", "short-term investments",
    "cash and bank", "cash and cash equivalents", "bank balances",
    "cash in hand", "bank deposits",

    # Balance Sheet — Liabilities
    "total liabilities", "non-current liabilities", "current liabilities",
    "long term financing", "long-term financing", "long term debt",
    "long-term debt", "long term borrowings", "long-term borrowings",
    "term finance", "sukuk", "bonds payable", "debentures",
    "lease liabilities", "finance lease", "operating lease",
    "deferred tax liability", "deferred income", "government grant",
    "employee benefit obligations", "retirement benefits",
    "short term borrowings", "short-term borrowings", "running finance",
    "export refinancing", "demand finance", "overdraft",
    "trade and other payables", "trade payables", "accounts payable",
    "accrued liabilities", "accrued expenses", "accruals",
    "current portion", "dividend payable", "unclaimed dividend",
    "unpaid dividend", "advance from customers", "contract liabilities",
    "income tax payable", "sales tax payable", "withholding tax payable",
    "contingencies", "commitments",

    # Equity
    "total equity", "shareholders equity", "stockholders equity",
    "owners equity", "net worth",
    "share capital", "paid up capital", "authorised capital",
    "ordinary shares", "preference shares", "share premium",
    "capital reserve", "revenue reserve", "general reserve",
    "statutory reserve", "surplus on revaluation",
    "revaluation surplus", "revaluation reserve",
    "exchange translation reserve", "hedging reserve",
    "unappropriated profit", "accumulated profit", "accumulated deficit",

    # Cash Flow
    "net cash", "cash flows", "cash generated",
    "operating activities", "investing activities", "financing activities",
    "cash from operations", "cash used in operations",
    "capital expenditure", "capex", "purchase of property",
    "proceeds from disposal", "proceeds from sale",
    "dividend paid", "dividend received",
    "proceeds from financing", "repayment of financing",
    "repayment of borrowings", "proceeds from borrowings",
    "interest paid", "finance cost paid", "tax paid",
    "net increase in cash", "net decrease in cash",
    "cash at beginning", "cash at end", "cash equivalents",

    # Per Share & Returns
    "earnings per share", "eps", "basic eps", "diluted eps",
    "dividend per share", "dps", "book value per share",
    "net asset value per share", "nav per share",
    "return on equity", "roe", "return on assets", "roa",
    "return on capital", "roce",

    # Ratios & Analysis
    "current ratio", "quick ratio", "acid test",
    "debt to equity", "gearing ratio", "leverage ratio",
    "interest coverage", "debt service coverage",
    "gross margin", "net margin", "operating margin", "ebitda margin",
    "asset turnover", "inventory turnover", "receivables turnover",
    "price to earnings", "p/e ratio", "price to book",

    # Audit & Governance
    "auditor", "auditors report", "audit report", "audited",
    "unaudited", "reviewed", "independent auditor",
    "chartered accountants", "engagement partner",
    "going concern", "material uncertainty", "emphasis of matter",
    "qualified opinion", "unqualified opinion",
    "board of directors", "directors report", "chairman",
    "chief executive", "managing director", "chief financial officer",
    "company secretary", "audit committee",

    # Notes & Policies
    "accounting policy", "accounting policies", "significant accounting",
    "basis of preparation", "basis of measurement",
    "ifrs", "ias", "gaap", "iasb", "financial reporting standards",
    "fair value", "historical cost", "carrying amount", "carrying value",
    "recoverable amount", "value in use",
    "related party", "related parties", "associated company",
    "subsidiary", "subsidiaries", "holding company", "parent company",
    "segment", "operating segment", "reportable segment",
    "subsequent event", "events after", "post balance sheet",
    "contingent liability", "contingent asset",
    "capital management", "financial risk", "market risk",
    "credit risk", "liquidity risk", "interest rate risk",
    "currency risk", "foreign exchange risk",
]


def is_financial_page(text: str) -> bool:
    t = text.lower()
    hits = sum(1 for kw in FINANCIAL_KEYWORDS if kw in t)
    return hits >= 1


# ── Phase 2: LLM extraction ────────────────────────────────────────────────────

EXTRACTION_PROMPT = """You are a financial data extraction expert. Extract ALL financial figures from the statement below.

RULES:
- Return ONLY valid JSON, no explanation, no markdown, no backticks
- Assign a letter label to each line item: A, B, C, D... AA, AB...
- Extract BOTH current period and prior period values where shown
- CRITICAL: Check the document header for the unit scale (Rupees in thousands / millions / billions)
  - If "Rupees in thousands" or "000": extract numbers as-is
  - If "Rupees in millions" or "Rs. million": multiply all numbers by 1000 before storing
  - If "Rupees in billions": multiply all numbers by 1000000 before storing
  - Always store final values in THOUSANDS regardless of source unit
- Bracketed numbers like (2,901,939) are negative - store as negative
- If a value is missing, use null
- Include ALL line items you can find, do not skip any

Return this exact JSON structure:
{
  "period_current": "e.g. 31 December 2025",
  "period_prior": "e.g. 31 December 2024",
  "currency": "PKR",
  "unit": "thousands",
  "source_unit": "thousands or millions or billions - what the document uses",
  "items": {
    "A": {"name": "line item name exactly as written", "current": number_or_null, "prior": number_or_null},
    "B": {"name": "line item name exactly as written", "current": number_or_null, "prior": number_or_null}
  }
}

Financial statement text:
"""


def extract_page_with_llm(page_text: str, attempt: int = 1) -> dict | None:
    """
    Fix 4: Retry logic — up to 2 attempts.
    Second attempt uses a simpler, more constrained prompt.
    """
    if attempt == 1:
        prompt = EXTRACTION_PROMPT + page_text
    else:
        # Simpler fallback prompt for retry
        prompt = """Extract financial figures from this text. Return ONLY JSON, no explanation.
Format: {"period_current": "date", "period_prior": "date", "items": {"A": {"name": "item name", "current": number, "prior": number}}}
Numbers in brackets are negative. Extract every line item with a number.

Text:
""" + page_text[:2000]  # truncate for retry

    raw = call_ollama(prompt)
    raw = re.sub(r"```json|```", "", raw).strip()
    match = re.search(r"\{.+\}", raw, re.DOTALL)
    if not match:
        if attempt < 2:
            print(f"     Retry attempt {attempt + 1}...")
            return extract_page_with_llm(page_text, attempt + 1)
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        fixed = match.group().replace("'", '"').replace("None", "null").replace("True", "true").replace("False", "false")
        try:
            return json.loads(fixed)
        except Exception:
            if attempt < 2:
                print(f"     JSON parse failed, retry attempt {attempt + 1}...")
                return extract_page_with_llm(page_text, attempt + 1)
            return None


# ── Phase 3: Map LLM items to DB fields ───────────────────────────────────────

FIELD_MAPPING = {
    "gross turnover":                               "gross_turnover",
    "net turnover":                                 "revenue",
    "net revenue":                                  "revenue",
    "net sales":                                    "revenue",
    "total revenue":                                "revenue",
    "revenue":                                      "revenue",
    "gross profit":                                 "gross_profit",
    "operating profit":                             "operating_profit",
    "profit from operations":                       "operating_profit",
    "ebit":                                         "operating_profit",
    "profit before tax":                            "profit_before_tax",
    "profit before taxation":                       "profit_before_tax",
    "profit for the period":                        "net_profit",
    "profit for the year":                          "net_profit",
    "profit after tax":                             "net_profit",
    "net profit":                                   "net_profit",
    "earnings per share":                           "eps",
    "basic and diluted":                            "eps",
    "eps":                                          "eps",
    "dividend per share":                           "dividend_per_share",
    "cost of sales":                                "cost_of_goods_sold",
    "cost of goods sold":                           "cost_of_goods_sold",
    "selling and distribution":                     "operating_expenses",
    "administrative expenses":                      "operating_expenses",
    "depreciation":                                 "depreciation",
    "finance cost":                                 "finance_cost",
    "finance costs":                                "finance_cost",
    "financial charges":                            "finance_cost",
    "income tax expense":                           "tax_expense",
    "taxation":                                     "tax_expense",
    "total assets":                                 "total_assets",
    "current assets":                               "current_assets",
    "non-current assets":                           "non_current_assets",
    "non current assets":                           "non_current_assets",
    "cash and bank balances":                       "cash_balance",
    "cash and cash equivalents":                    "cash_balance",
    "trade debts":                                  "trade_receivables",
    "trade receivables":                            "trade_receivables",
    "stock in trade":                               "inventory",
    "inventories":                                  "inventory",
    "total liabilities":                            "total_liabilities",
    "current liabilities":                          "current_liabilities",
    "non-current liabilities":                      "non_current_liabilities",
    "non current liabilities":                      "non_current_liabilities",
    "total equity":                                 "total_equity",
    "share capital":                                "share_capital",
    "long term financing":                          "long_term_debt",
    "long-term financing":                          "long_term_debt",
    "long term borrowings":                         "long_term_debt",
    "net cash generated from operating activities": "operating_cashflow",
    "net cash from operating activities":           "operating_cashflow",
    "net cash generated from investing":            "investing_cashflow",
    "net cash used in investing":                   "investing_cashflow",
    "net cash used in financing":                   "financing_cashflow",
    "net cash from financing":                      "financing_cashflow",
}


def map_items_to_fields(items: dict) -> tuple:
    current = {}
    prior = {}

    for label, item in items.items():
        name = item.get("name", "").lower().strip()
        cur_val = item.get("current")
        pri_val = item.get("prior")

        matched_field = None
        best_score = 0

        for pattern, field in FIELD_MAPPING.items():
            if pattern in name:
                score = len(pattern)
                if score > best_score:
                    best_score = score
                    matched_field = field

        if matched_field and cur_val is not None:
            if matched_field not in current:
                current[matched_field] = float(cur_val)
        if matched_field and pri_val is not None:
            if matched_field not in prior:
                prior[matched_field] = float(pri_val)

    return current, prior


# ── Phase 4: Deterministic validation ─────────────────────────────────────────

def within_tolerance(a: float, b: float, pct: float = 0.02) -> bool:
    if a == 0 and b == 0:
        return True
    if a == 0 or b == 0:
        return abs(a - b) < 1000
    return abs(a - b) / max(abs(a), abs(b)) <= pct


def validate_financials(data: dict) -> dict:
    passed = []
    failed = []
    warnings = []

    def check(name: str, computed: float, stored: float, tolerance: float = 0.02):
        if within_tolerance(computed, stored, tolerance):
            passed.append(name)
        else:
            failed.append({
                "check": name,
                "expected": round(computed, 2),
                "got": round(stored, 2),
                "diff_pct": round(abs(computed - stored) / max(abs(computed), abs(stored), 1) * 100, 2)
            })

    if all(k in data for k in ["gross_profit", "revenue", "cost_of_goods_sold"]):
        check("Gross Profit = Revenue - COGS",
              data["revenue"] - data["cost_of_goods_sold"],
              data["gross_profit"])

    if all(k in data for k in ["operating_profit", "gross_profit", "operating_expenses"]):
        check("Operating Profit = Gross Profit - OpEx",
              data["gross_profit"] - data["operating_expenses"],
              data["operating_profit"], tolerance=0.05)

    if all(k in data for k in ["net_profit", "profit_before_tax", "tax_expense"]):
        check("Net Profit = PBT - Tax",
              data["profit_before_tax"] - data["tax_expense"],
              data["net_profit"])

    if all(k in data for k in ["total_assets", "total_liabilities", "total_equity"]):
        check("Assets = Liabilities + Equity",
              data["total_liabilities"] + data["total_equity"],
              data["total_assets"])

    if all(k in data for k in ["total_assets", "current_assets", "non_current_assets"]):
        check("Total Assets = Current + Non-Current",
              data["current_assets"] + data["non_current_assets"],
              data["total_assets"])

    if all(k in data for k in ["total_liabilities", "current_liabilities", "non_current_liabilities"]):
        check("Total Liabilities = Current + Non-Current",
              data["current_liabilities"] + data["non_current_liabilities"],
              data["total_liabilities"])

    if "revenue" in data and "net_profit" in data and data["revenue"] != 0:
        margin = data["net_profit"] / data["revenue"]
        if margin > 0.5:
            warnings.append(f"Net profit margin {margin:.1%} seems unusually high — verify extraction")
        if margin < -0.5:
            warnings.append(f"Net profit margin {margin:.1%} seems unusually low — verify extraction")

    if "eps" in data and data["eps"] > 10000:
        warnings.append(f"EPS of {data['eps']} seems very high — may have been extracted in wrong units")

    return {"passed": passed, "failed": failed, "warnings": warnings}


# ── Master extraction function ─────────────────────────────────────────────────

def extract_financials_intelligent(pdf_path: str) -> dict:
    print(f"📄 Reading PDF: {pdf_path}")
    pages = extract_pages(pdf_path)
    full_text = "\n".join(p["text"] for p in pages)

    financial_pages = [p for p in pages if is_financial_page(p["text"])]
    print(f"   Found {len(financial_pages)} financial pages out of {len(pages)} total")

    all_current = {}
    all_prior = {}
    metadata = {
        "period_current": None,
        "period_prior": None,
        "currency": "PKR",
        "pages_processed": 0,
        "validation": None
    }

    for page in financial_pages:
        print(f"   Processing page {page['page']}...")
        result = extract_page_with_llm(page["text"])

        if not result:
            print(f"   ⚠️  Page {page['page']}: LLM extraction failed, skipping")
            continue

        metadata["pages_processed"] += 1

        if not metadata["period_current"] and result.get("period_current"):
            metadata["period_current"] = result["period_current"]
        if not metadata["period_prior"] and result.get("period_prior"):
            metadata["period_prior"] = result["period_prior"]

        items = result.get("items", {})
        cur, pri = map_items_to_fields(items)

        for k, v in cur.items():
            if k not in all_current:
                all_current[k] = v
        for k, v in pri.items():
            if k not in all_prior:
                all_prior[k] = v

    if all_current:
        validation = validate_financials(all_current)
        metadata["validation"] = validation

        print(f"\n📊 Validation results:")
        print(f"   ✅ Passed: {len(validation['passed'])} checks")
        if validation["failed"]:
            print(f"   ❌ Failed: {len(validation['failed'])} checks")
            for f in validation["failed"]:
                print(f"      {f['check']}: expected {f['expected']:,.0f}, got {f['got']:,.0f} ({f['diff_pct']}% diff)")
        if validation["warnings"]:
            for w in validation["warnings"]:
                print(f"   ⚠️  {w}")

    return {
        "current": all_current,
        "prior": all_prior,
        "metadata": metadata,
        "raw_text": full_text
    }


# ── Backwards-compatible wrappers ──────────────────────────────────────────────

def extract_text(pdf_path: str) -> str:
    return extract_text_only(pdf_path)


def extract_financials(text: str) -> dict:
    """Legacy regex fallback."""
    data = {}
    NUM = r"([\d,]+(?:\.\d+)?)"
    patterns = {
        "revenue":           r"Net (?:turnover|revenue|sales)\s+" + NUM,
        "gross_profit":      r"Gross profit\s+" + NUM,
        "profit_before_tax": r"Profit before (?:tax|taxation)\s+" + NUM,
        "net_profit":        r"Profit (?:for the (?:period|year)|after tax)\s+" + NUM,
        "eps":               r"(?:Earnings|EPS) per share[^\d]+" + NUM,
        "depreciation":      r"Depreciation\s+" + NUM,
        "finance_cost":      r"Finance costs?\s+" + NUM,
        "total_assets":      r"Total assets\s+" + NUM,
        "total_liabilities": r"Total liabilities\s+" + NUM,
        "cash_balance":      r"Cash and (?:bank )?balances?\s+" + NUM,
        "operating_cashflow":r"Net cash generated from operating activities\s+" + NUM,
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            data[key] = float(match.group(1).replace(",", ""))
    period_match = re.search(
        r"(?:half year|six month|period|year)\s+ended?\s+(\d{1,2}\s+\w+\s+\d{4}|\w+\s+\d{4})",
        text, re.IGNORECASE
    )
    data["_period"] = period_match.group(1).strip() if period_match else "FY 2025"
    data["_company"] = "Unknown"
    return data


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from db_insert import insert_financials

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "data/Test PDF.pdf"
    company  = sys.argv[2] if len(sys.argv) > 2 else "Bestway Cement"

    result = extract_financials_intelligent(pdf_path)
    current  = result["current"]
    prior    = result["prior"]
    meta     = result["metadata"]

    print(f"\n📋 Extracted {len(current)} current fields, {len(prior)} prior fields")
    print(f"   Period: {meta['period_current']} vs {meta['period_prior']}")

    period_str = meta["period_current"] or "31 December 2025"
    year_match = re.search(r"\d{4}", period_str)
    year = int(year_match.group()) if year_match else 2025

    if current:
        insert_financials(current, company=company, year=year, period=f"H1 FY{year}")
        print(f"✅ Inserted: {company} H1 FY{year}")
    if prior:
        prior_str = meta.get("period_prior", "")
        prior_year_match = re.search(r"\d{4}", prior_str)
        prior_year = int(prior_year_match.group()) if prior_year_match else year - 1
        insert_financials(prior, company=company, year=prior_year, period=f"H1 FY{prior_year}")
        print(f"✅ Inserted: {company} H1 FY{prior_year}")