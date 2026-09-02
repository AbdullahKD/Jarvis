"""
finex/statement_fields.py — shared vocabulary for financial-statement parsing.

This module holds the parts of the extractor that are pure reference data or
pure text-to-number utilities: the line-item alias dictionaries, the sets that
say how a field behaves (always positive, never scaled, never accumulated) and
the cell parsers.

It was split out of ``extract_pdf.py`` so the v3 grid extractor and the v4
statement-scoped extractor share ONE copy. Two copies of an alias dictionary
drift, and a drift here shows up as a field that extracts from one code path
and not the other, which is close to impossible to diagnose from the output.

Nothing in this file knows about pages, grids, LLMs or the database.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set


# ── Numeric patterns ────────────────────────────────
_YEAR_PAT  = re.compile(r'\b(20\d{2}|19\d{2})\b')
_NEG_TEXT  = re.compile(r'\(([\d]{1,3}(?:,[\d]{3})*(?:\.\d+)?)\)')
_POS_TEXT  = re.compile(
    r'(?<!\()\b([\d]{1,3}(?:,[\d]{3})+(?:\.\d+)?|\d{5,}(?:\.\d+)?)\b(?!\))'
)
_EPS_PAT   = re.compile(r'\b(\d{1,3}\.\d{2})\b')



# ── Cell / value utilities ──────────────────────────────────────────────────────

def _parse_cell(cell) -> Optional[float]:
    """Parse a table cell string into a float. Brackets → negative."""
    if cell is None:
        return None
    s = (str(cell).strip()
         .replace(',', '').replace('£', '').replace('$', '')
         .replace('€', '').replace('Rs.', '').replace('%', '').strip())
    if not s or s in ('-', '—', '–', 'n/a', 'nil'):
        return None
    if s.startswith('(') and s.endswith(')'):
        inner = s[1:-1]
        try:
            return -abs(float(inner))
        except ValueError:
            return None
    try:
        return float(s)
    except ValueError:
        return None


def _is_note_ref(val: float, nums_found: List[float]) -> bool:
    """True if val looks like a note-reference integer before any real value."""
    if nums_found:
        return False
    if val != int(val):
        return False
    return 1 <= abs(val) <= 99


def _clean_label(cell) -> str:
    """Lowercase, collapse whitespace — for pattern matching."""
    if cell is None:
        return ''
    return re.sub(r'\s+', ' ', str(cell).strip()).lower()


def _is_likely_label(cell) -> bool:
    """Heuristic: does this cell look like a text label rather than a number?"""
    if cell is None:
        return False
    s = str(cell).strip()
    return bool(s) and _parse_cell(s) is None and len(s) > 2


# ── Field alias dictionaries ────────────────────────────────────────────────────

_PNL_FIELDS: Dict[str, str] = {

    # Revenue / Turnover
    "revenue from contracts with customers":         "revenue",
    "revenue from contract with customers":          "revenue",
    "revenue from sale of goods and services":       "revenue",
    "total revenue":                                 "revenue",
    "net turnover":                                  "revenue",
    "net revenue":                                   "revenue",
    "net sales":                                     "revenue",
    "revenue":                                       "revenue",
    "turnover":                                      "revenue",
    "sales":                                         "revenue",
    "net turnover from operations":                  "revenue",
    "operating revenue":                             "revenue",
    "revenue from operations":                       "revenue",
    "net operating revenue":                         "revenue",
    "income from operations":                        "revenue",
    "total income":                                  "revenue",
    "sales - net":                                   "revenue",
    "net sales revenue":                             "revenue",

    # Gross Turnover
    "gross turnover":                                "gross_turnover",
    "gross sales":                                   "gross_turnover",
    "gross revenue":                                 "gross_turnover",
    "total sales":                                   "gross_turnover",
    "sales - gross":                                 "gross_turnover",

    # Gross Profit
    "gross profit/(loss)":                           "gross_profit",
    "gross (loss)/profit":                           "gross_profit",
    "gross profit":                                  "gross_profit",
    "gross loss":                                    "gross_profit",
    "gross margin":                                  "gross_profit",

    # Operating Profit
    "operating profit/(loss)":                       "operating_profit",
    "operating (loss)/profit":                       "operating_profit",
    "operating profit":                              "operating_profit",
    "operating loss":                                "operating_profit",
    "profit from operations":                        "operating_profit",
    "profit/(loss) from operations":                 "operating_profit",
    "(loss)/profit from operations":                 "operating_profit",
    "results from operating activities":             "operating_profit",
    "operating earnings":                            "operating_profit",
    "operating income":                              "operating_profit",
    "ebit":                                          "operating_profit",
    "profit before other income and finance":        "operating_profit",
    "profit before financing and taxation":          "operating_profit",
    "profit before finance":                         "operating_profit",

    # Profit Before Tax
    "income before taxation":                        "profit_before_tax",
    "income before taxation for the period":         "profit_before_tax",
    "profit before taxation from continuing":        "profit_before_tax",
    "profit before income tax and levy":             "profit_before_tax",
    "profit before income tax":                      "profit_before_tax",
    "profit before tax":                             "profit_before_tax",
    "profit before taxation":                        "profit_before_tax",
    "(loss)/profit before taxation":                 "profit_before_tax",
    "profit/(loss) before taxation":                 "profit_before_tax",
    "(loss)/profit before income tax":               "profit_before_tax",
    "profit/(loss) before income tax":               "profit_before_tax",
    "(loss)/profit before tax":                      "profit_before_tax",
    "loss before taxation":                          "profit_before_tax",
    "loss before tax":                               "profit_before_tax",
    "net profit before tax":                         "profit_before_tax",
    "earnings before tax":                           "profit_before_tax",
    "profit before workers":                         "profit_before_tax",
    "profit before wppf":                            "profit_before_tax",
    "profit before taxation and wppf":               "profit_before_tax",

    # Net Profit
    "income for the period":                         "net_profit",
    "profit for the year":                           "net_profit",
    "profit for the period":                         "net_profit",
    "profit/(loss) for the year":                    "net_profit",
    "(loss)/profit for the year":                    "net_profit",
    "profit/(loss) for the period":                  "net_profit",
    "(loss)/profit for the period":                  "net_profit",
    "(loss)/profit for the financial year":          "net_profit",
    "profit/(loss) for the financial year":          "net_profit",
    "net profit from continuing operations":         "net_profit",
    "total net profit":                              "net_profit",
    "profit after taxation":                         "net_profit",
    "profit after tax":                              "net_profit",
    "net profit":                                    "net_profit",
    "net income":                                    "net_profit",
    "net earnings":                                  "net_profit",
    "profit after income tax":                       "net_profit",
    "profit attributable to owners":                 "net_profit",
    "profit attributable to equity":                 "net_profit",
    "profit attributable to shareholders":           "net_profit",
    "loss for the year":                             "net_profit",
    "loss for the period":                           "net_profit",
    "loss after taxation":                           "net_profit",
    "net loss":                                      "net_profit",

    # COGS
    "material and other cost of sales":              "cost_of_goods_sold",
    "purchases":                                     "cost_of_goods_sold",
    "cost of sales":                                 "cost_of_goods_sold",
    "cost of goods sold":                            "cost_of_goods_sold",
    "cost of revenue":                               "cost_of_goods_sold",
    "cost of services":                              "cost_of_goods_sold",
    "cost of products sold":                         "cost_of_goods_sold",
    "direct costs":                                  "cost_of_goods_sold",
    "production and manufacturing expenses":         "cost_of_goods_sold",
    "cost of goods":                                 "cost_of_goods_sold",
    "cost of operations":                            "cost_of_goods_sold",
    "cost of net revenues":                          "cost_of_goods_sold",
    "cost of goods manufactured":                    "cost_of_goods_sold",

    # Operating Expenses
    "selling, distribution and administrative expenses": "operating_expenses",
    "commercial and administrative costs":           "operating_expenses",
    "selling and distribution expenses":             "operating_expenses",
    "general and administrative expenses":           "operating_expenses",
    "administrative expenses":                       "operating_expenses",
    "administration expenses":                       "operating_expenses",
    "selling, general and administrative":           "operating_expenses",
    "selling and distribution":                      "operating_expenses",
    "selling and marketing":                         "operating_expenses",
    "distribution costs":                            "operating_expenses",
    "distribution expenses":                         "operating_expenses",
    "marketing and distribution":                    "operating_expenses",
    "selling expenses":                              "operating_expenses",
    "marketing expenses":                            "operating_expenses",
    "operating expenses":                            "operating_expenses",
    "other operating expenses":                      "operating_expenses",
    "total operating expenses":                      "operating_expenses",
    "employee costs":                                "operating_expenses",
    "staff costs":                                   "operating_expenses",
    "other expenses":                                "operating_expenses",

    # Finance Cost
    "net finance costs":                             "finance_cost",
    "financing costs":                               "finance_cost",
    "finance costs":                                 "finance_cost",
    "finance cost":                                  "finance_cost",
    "finance expense":                               "finance_cost",
    "finance expense (net)":                         "finance_cost",
    "finance expenses and fees paid":                "finance_cost",
    "financial charges":                             "finance_cost",
    "interest expense":                              "finance_cost",
    "interest expense (net)":                        "finance_cost",
    "interest and other charges":                    "finance_cost",
    "borrowing costs":                               "finance_cost",
    "mark-up expense":                               "finance_cost",
    "markup expense":                                "finance_cost",
    "mark-up on borrowings":                         "finance_cost",
    "finance charges":                               "finance_cost",
    "financial expenses":                            "finance_cost",
    "net financing income/(costs)":                  "finance_cost",
    "net financing costs":                           "finance_cost",

    # Tax
    "taxation charge":                               "tax_expense",
    "income tax (expense)/credit":                   "tax_expense",
    "income tax expense":                            "tax_expense",
    "income tax excluding impact":                   "tax_expense",
    "income tax":                                    "tax_expense",
    "taxation":                                      "tax_expense",
    "tax expense":                                   "tax_expense",
    "provision for taxation":                        "tax_expense",
    "tax charge":                                    "tax_expense",
    "current tax":                                   "tax_expense",
    "tax for the period":                            "tax_expense",
    "tax for the year":                              "tax_expense",

    # EPS
    "basic earnings per share":                      "eps",
    "diluted earnings per share":                    "eps",
    "basic and diluted earnings per share":          "eps",
    "basic and diluted eps":                         "eps",
    "basic and diluted":                             "eps",
    "earnings per share":                            "eps",
    "earnings/(loss) per share":                     "eps",
    "(loss)/earnings per share":                     "eps",
    "loss per share":                                "eps",
    "profit/(loss) per share":                       "eps",
    "net earnings per share":                        "eps",

    # Dividend
    "dividend per share":                            "dividend_per_share",
    "cash dividend per share":                       "dividend_per_share",
    "interim dividend per share":                    "dividend_per_share",
    "final dividend per share":                      "dividend_per_share",
    "dividends per share":                           "dividend_per_share",
}

_BS_FIELDS: Dict[str, str] = {

    # Total Assets
    "total assets":                                  "total_assets",
    "total assets employed":                         "total_assets",
    "assets total":                                  "total_assets",
    "total of assets":                               "total_assets",
    "sum of assets":                                 "total_assets",

    # Non-Current Assets
    "total non-current assets":                      "non_current_assets",
    "total non current assets":                      "non_current_assets",
    "non-current assets":                            "non_current_assets",
    "non current assets":                            "non_current_assets",
    "fixed assets":                                  "non_current_assets",
    "total fixed assets":                            "non_current_assets",
    "long-term assets":                              "non_current_assets",
    "long term assets":                              "non_current_assets",
    "property, plant and equipment and other":       "non_current_assets",
    "intangible and other non-current assets":       "non_current_assets",

    # Current Assets
    "total current assets":                          "current_assets",
    "current assets":                                "current_assets",
    "net current assets":                            "current_assets",
    "current assets total":                          "current_assets",
    "total current assets and equivalents":          "current_assets",

    # Cash
    "cash and cash equivalents":                     "cash_balance",
    "cash and bank balances":                        "cash_balance",
    "cash and balances with banks":                  "cash_balance",
    "cash at bank and in hand":                      "cash_balance",
    "cash and short-term deposits":                  "cash_balance",
    "cash and bank":                                 "cash_balance",
    "bank balances":                                 "cash_balance",
    "cash in hand and at bank":                      "cash_balance",
    "cash and equivalents":                          "cash_balance",
    "cash and cash equivalents and short-term":      "cash_balance",

    # Trade Receivables
    "trade and other receivables":                   "trade_receivables",
    "trade and other current receivables":           "trade_receivables",
    "trade receivables and other assets":            "trade_receivables",
    "trade receivables":                             "trade_receivables",
    "trade debts":                                   "trade_receivables",
    "accounts receivable":                           "trade_receivables",
    "debtors":                                       "trade_receivables",
    "trade debtors":                                 "trade_receivables",
    "net trade receivables":                         "trade_receivables",
    "loans and advances":                            "trade_receivables",
    "advances, deposits and prepayments":            "trade_receivables",

    # Inventory
    "inventories":                                   "inventory",
    "stock in trade":                                "inventory",
    "stock-in-trade":                                "inventory",
    "stores, spares and loose tools":                "inventory",
    "stores and spares":                             "inventory",
    "raw materials":                                 "inventory",
    "finished goods":                                "inventory",
    "work in process":                               "inventory",
    "work-in-progress":                              "inventory",
    "total inventories":                             "inventory",

    # Total Liabilities
    "total liabilities":                             "total_liabilities",
    "liabilities total":                             "total_liabilities",
    "total of liabilities":                          "total_liabilities",
    "total liabilities (current and non-current)":  "total_liabilities",
    "total current and non-current liabilities":     "total_liabilities",
    "total debt and liabilities":                    "total_liabilities",
    "total borrowings and liabilities":              "total_liabilities",

    # Non-Current Liabilities
    "total non-current liabilities":                 "non_current_liabilities",
    "total non current liabilities":                 "non_current_liabilities",
    "non-current liabilities":                       "non_current_liabilities",
    "non current liabilities":                       "non_current_liabilities",
    "long-term liabilities":                         "non_current_liabilities",
    "long term liabilities":                         "non_current_liabilities",
    "non-current liabilities total":                 "non_current_liabilities",
    "total long-term liabilities":                   "non_current_liabilities",
    "total long term liabilities":                   "non_current_liabilities",

    # Current Liabilities
    "total current liabilities":                     "current_liabilities",
    "current liabilities":                           "current_liabilities",
    "current liabilities total":                     "current_liabilities",
    "net current liabilities":                       "current_liabilities",
    "total short-term liabilities":                  "current_liabilities",
    "short-term liabilities":                        "current_liabilities",

    # Total Equity
    "equity attributable to shareholders":           "total_equity",
    "equity attributable to owners":                 "total_equity",
    "total equity attributable to owners":           "total_equity",
    "total shareholders equity":                     "total_equity",
    "shareholders equity":                           "total_equity",
    "stockholders equity":                           "total_equity",
    "total shareholders funds":                      "total_equity",
    "owners equity":                                 "total_equity",
    "total equity":                                  "total_equity",
    "net assets":                                    "total_equity",

    # Share Capital
    "ordinary share capital":                        "share_capital",
    "issued, subscribed and paid":                   "share_capital",
    "issued and paid-up capital":                    "share_capital",
    "paid-up capital":                               "share_capital",
    "called up share capital":                       "share_capital",
    "share capital":                                 "share_capital",
    "common stock":                                  "share_capital",

    # Long-Term Debt
    "long-term borrowings":                          "long_term_debt",
    "long term borrowings":                          "long_term_debt",
    "long-term financing":                           "long_term_debt",
    "long term financing":                           "long_term_debt",
    "long-term loans":                               "long_term_debt",
    "long term loans":                               "long_term_debt",
    "long term debt":                                "long_term_debt",
    "non-current borrowings":                        "long_term_debt",
    "redeemable capital":                            "long_term_debt",
    "term finance certificates":                     "long_term_debt",
    "sukuk":                                         "long_term_debt",
    "sukuk bonds":                                   "long_term_debt",
    "diminishing musharika":                         "long_term_debt",
    "murabaha financing":                            "long_term_debt",
    "lease liabilities":                             "long_term_debt",
    "finance lease liabilities":                     "long_term_debt",
}

_CF_FIELDS: Dict[str, str] = {

    # Operating
    "net cash generated from operating activities":  "operating_cashflow",
    "net cash flow from continuing operating":       "operating_cashflow",
    "inflow from operating activities":              "operating_cashflow",
    "net cash from operating activities":            "operating_cashflow",
    "net cash used in operating activities":         "operating_cashflow",
    "net cash generated from operating":             "operating_cashflow",
    "net cash from operating":                       "operating_cashflow",
    "cash flow from operating activities":           "operating_cashflow",
    "cash flow used in operating activities":        "operating_cashflow",
    "cash flows from operating activities":          "operating_cashflow",
    "net cash inflow from operating":                "operating_cashflow",
    "cash generated from operations":               "operating_cashflow",
    "operating cash flows":                          "operating_cashflow",
    "net cash provided by operating":               "operating_cashflow",
    "cash from operating activities":               "operating_cashflow",

    # Investing
    "net cash used in investing activities":         "investing_cashflow",
    "inflow/(outflow) from investing activities":    "investing_cashflow",
    "net cash generated from investing activities":  "investing_cashflow",
    "net cash from investing activities":            "investing_cashflow",
    "cash flow from investing activities":           "investing_cashflow",
    "cash flow used in investing activities":        "investing_cashflow",
    "cash flows from investing activities":          "investing_cashflow",
    "net cash used in investing":                    "investing_cashflow",
    "net cash generated from investing":             "investing_cashflow",
    "investing cash flows":                          "investing_cashflow",
    "net cash provided by investing":                "investing_cashflow",

    # Financing
    "net cash used in financing activities":         "financing_cashflow",
    "net cash generated from financing activities":  "financing_cashflow",
    "net cash from financing activities":            "financing_cashflow",
    "cash flow from financing activities":           "financing_cashflow",
    "cash flow used in financing activities":        "financing_cashflow",
    "cash flows from financing activities":          "financing_cashflow",
    "net cash used in financing":                    "financing_cashflow",
    "net cash generated from financing":             "financing_cashflow",
    "inflow/(outflow) from financing":               "financing_cashflow",
    "financing cash flows":                          "financing_cashflow",
    "net cash provided by financing":                "financing_cashflow",

    # Depreciation (in CF reconciliation)
    "depreciation, depletion and amortisation":      "depreciation",
    "depreciation, depletion and amortization":      "depreciation",
    "depreciation, amortisation and impairment":     "depreciation",
    "depreciation, amortization and impairment":     "depreciation",
    "depreciation and amortisation":                 "depreciation",
    "depreciation and amortization":                 "depreciation",
    "amortisation and impairment of intangible":     "depreciation",
    "depreciation of property":                      "depreciation",
    "depreciation charge":                           "depreciation",
    "depreciation":                                  "depreciation",
    "amortisation":                                  "depreciation",
    "amortization":                                  "depreciation",
}

# Rows to ignore regardless of which statement they appear on
_SKIP: Set[str] = {
    "total equity and liabilities",
    "total liabilities and equity",
    "total equity and total liabilities",
    "total liabilities and shareholders",
    "contingencies and commitments",
    "authorised share capital",
    "revenue reserves",
    "share premium",
    "unappropriated profit",
    "general reserve",
    "capital reserve",
    "capital redemption reserve",
    "surplus on revaluation",
    "hedging reserve",
    "foreign currency translation",
    "other reserves",
    "retained earnings",
    "retained profit",
    "accumulated losses",
    "non-controlling interests",
    "non-controlling interest",
    "minority interest",
    "total comprehensive income",
    "total comprehensive expense",
    "other comprehensive income",
    "other comprehensive expense",
    "items that will",
    "items that may",
    "remeasurement",
    "adjustments for non-cash",
    "changes in working capital",
    "working capital changes",
    "cashflows generated from operations",
    "cash flows generated",
    "changes in inventories",
    "changes in receivables",
    "changes in payables",
    "deferred tax asset",
    "deferred tax liability",
    "deferred taxation",
    "deferred tax",
    "workers profit participation",
    "workers welfare fund",
    "wppf",
    "wwf",
    "the annexed notes",
    "notes to the",
}

# Fields that are always stored as positive (costs, assets, etc.)
_ALWAYS_POSITIVE: Set[str] = {
    "cost_of_goods_sold", "operating_expenses", "finance_cost", "tax_expense",
    "total_assets", "total_liabilities", "total_equity", "current_assets",
    "non_current_assets", "current_liabilities", "non_current_liabilities",
    "cash_balance", "inventory", "trade_receivables", "share_capital",
    "long_term_debt", "gross_turnover", "revenue", "depreciation",
}

# Fields that are NOT multiplied by the page-level scale factor
_NO_SCALE: Set[str] = {"eps", "dividend_per_share"}

# Fields that should NEVER be accumulated across multiple matching rows.
# These are top-line totals — picking one up from a notes disaggregation
# table and then summing it with sub-items would produce nonsense figures.
_NO_ACCUMULATE: Set[str] = {
    "revenue", "gross_turnover", "gross_profit", "operating_profit",
    "profit_before_tax", "net_profit", "eps", "dividend_per_share",
    "total_assets", "total_liabilities", "total_equity",
    "current_assets", "non_current_assets",
    "current_liabilities", "non_current_liabilities",
    "cash_balance", "trade_receivables", "inventory",
    "long_term_debt", "share_capital",
    "operating_cashflow", "investing_cashflow", "financing_cashflow",
}

# Tax lines only accepted from P&L pages
_PNL_TAX_PATTERN = re.compile(
    r'(?:income tax|taxation charge|taxation\s*$|tax expense|tax charge|'
    r'provision for tax|current tax|tax for the|income tax \(expense\)|'
    r'taxation -levy|taxation excluding)',
    re.IGNORECASE,
)


def _best_match(label: str, mapping: Dict[str, str]) -> Optional[str]:
    """Longest substring match wins (specificity over order)."""
    best, score = None, 0
    for pattern, field in mapping.items():
        if pattern in label and len(pattern) > score:
            best, score = field, len(pattern)
    return best


def _get_mapping(ptype: str) -> Dict[str, str]:
    return {"pnl": _PNL_FIELDS, "bs": _BS_FIELDS, "cf": _CF_FIELDS}.get(ptype, {})
