import psycopg2
from finex.db_insert import get_connection

_conn = None

MONETARY_COLS = {
    "revenue", "gross_profit", "operating_profit", "profit_before_tax",
    "net_profit", "cost_of_goods_sold", "operating_expenses", "depreciation",
    "finance_cost", "tax_expense", "total_assets", "current_assets",
    "non_current_assets", "cash_balance", "trade_receivables", "inventory",
    "total_liabilities", "current_liabilities", "non_current_liabilities",
    "total_equity", "share_capital", "long_term_debt", "gross_turnover",
    "operating_cashflow", "investing_cashflow", "financing_cashflow",
    "dividend_per_share"
}

def get_conn():
    global _conn
    if _conn is None or _conn.closed:
        _conn = get_connection()
    return _conn


def run_query(sql: str):
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(sql)
        colnames = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        cur.close()
        return {"columns": colnames, "rows": rows}
    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        return {"error": str(e)}


def get_all_years(company: str = "Bestway Cement") -> list:
    result = run_query(f"SELECT DISTINCT year FROM financials WHERE company = '{company}' ORDER BY year DESC")
    if "rows" in result:
        return [r[0] for r in result["rows"]]
    return []


def get_financial_context(company: str = "Bestway Cement", years: list = None) -> str:
    """
    Returns structured financial data for LLM prompts.
    All monetary figures are clearly labelled as PKR thousands.
    """
    if years is None:
        years = get_all_years(company)

    if not years:
        return "No financial data found in database."

    year_filter = "(" + ", ".join(str(y) for y in years) + ")"
    result = run_query(f"""
        SELECT * FROM financials
        WHERE company = '{company}' AND year IN {year_filter}
        ORDER BY year DESC, period
    """)

    if "error" in result:
        return f"DB error: {result['error']}"

    cols = result["columns"]
    rows = result["rows"]

    lines = [f"=== Financial Data: {company} (all monetary figures in PKR thousands) ===\n"]

    for row in rows:
        record = dict(zip(cols, row))
        lines.append(f"--- Period: {record.get('period', record['year'])} ---")
        for col in cols:
            if col in ("id",):
                continue
            val = record.get(col)
            if val is not None:
                if isinstance(val, float):
                    if col in MONETARY_COLS:
                        lines.append(f"  {col}: PKR {val:,.2f} thousand")
                    else:
                        lines.append(f"  {col}: {val:,.2f}")
                else:
                    lines.append(f"  {col}: {val}")
        lines.append("")

    return "\n".join(lines)


def format_result(result: dict) -> str:
    """Format a query result dict into readable text."""
    if "error" in result:
        return f"Error: {result['error']}"
    cols = result["columns"]
    rows = result["rows"]
    if not rows:
        return "No results found."
    lines = [" | ".join(cols)]
    lines.append("-" * 60)
    for row in rows:
        lines.append(" | ".join(str(v) if v is not None else "N/A" for v in row))
    return "\n".join(lines)