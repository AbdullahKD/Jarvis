import psycopg2
from finex.db_insert import get_connection

# Read-only connection — kept in AUTOCOMMIT so it never holds locks
# that would block DDL (CREATE TABLE / ALTER TABLE) in db_insert.
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

_SKIP_COLS = {"id"}


def get_conn():
    global _conn
    if _conn is None or _conn.closed:
        _conn = get_connection()
        # AUTOCOMMIT: reads never open an implicit transaction, so they
        # never block DDL locks in the writer connection.
        _conn.set_session(autocommit=True)
    return _conn


def _fresh_conn():
    """One-shot autocommit connection for queries that don't need the cache."""
    c = get_connection()
    c.set_session(autocommit=True)
    return c


def run_query(sql: str, params=None):
    conn = None
    try:
        conn = _fresh_conn()
        cur = conn.cursor()
        if params:
            cur.execute(sql, params)
        else:
            cur.execute(sql)
        colnames = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        cur.close()
        return {"columns": colnames, "rows": rows}
    except Exception as e:
        return {"error": str(e)}
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


def get_all_years(company: str = "Bestway Cement") -> list:
    try:
        conn = _fresh_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT DISTINCT year FROM financials WHERE company = %s ORDER BY year DESC",
            (company,)
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [r[0] for r in rows]
    except Exception:
        return []


def get_financial_context(company: str = "Bestway Cement", years: list = None) -> str:
    """
    Returns structured financial data for LLM prompts.
    Values are stored as absolute currency amounts (raw PDF number × scale_factor).
    """
    if years is None:
        years = get_all_years(company)

    if not years:
        return "No financial data found in database."

    try:
        conn = _fresh_conn()
        cur = conn.cursor()
        placeholders = ", ".join(["%s"] * len(years))
        cur.execute(
            f"SELECT * FROM financials WHERE company = %s AND year IN ({placeholders}) ORDER BY year DESC, period",
            [company] + list(years)
        )
        colnames = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        cur.close()
        conn.close()
    except Exception as e:
        return f"DB error: {e}"

    if not rows:
        return f"No financial data found for {company}."

    sym_map = {"GBP": "£", "USD": "$", "EUR": "€", "PKR": "PKR",
               "AED": "AED", "SAR": "SAR"}

    lines = [f"=== Financial Data: {company} ===\n"]

    for row in rows:
        record = dict(zip(colnames, row))
        currency   = record.get("currency")  or "Unknown"
        unit_label = record.get("unit_label") or ""
        sym = sym_map.get(currency, currency if currency != "Unknown" else "")

        lines.append(f"--- Period: {record.get('period', record['year'])} ---")
        lines.append(f"  [Source unit: {unit_label or 'unknown'}  |  Stored as: absolute {currency}]")

        for col in colnames:
            if col in _SKIP_COLS or col in ("currency", "unit_label"):
                continue
            val = record.get(col)
            if val is None:
                continue
            if isinstance(val, float):
                if col in MONETARY_COLS:
                    if abs(val) >= 1_000_000_000:
                        human = f"({sym}{val/1_000_000_000:,.2f}bn)"
                    elif abs(val) >= 1_000_000:
                        human = f"({sym}{val/1_000_000:,.2f}m)"
                    elif abs(val) >= 1_000:
                        human = f"({sym}{val/1_000:,.2f}k)"
                    else:
                        human = ""
                    lines.append(f"  {col}: {sym}{val:,.2f} {human}")
                else:
                    lines.append(f"  {col}: {val:,.4f}")
            else:
                lines.append(f"  {col}: {val}")

        lines.append("")

    return "\n".join(lines)


def format_result(result: dict) -> str:
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
