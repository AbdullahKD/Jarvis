import psycopg2

# Full schema DDL — run once to set up the database
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS financials (
    id                      SERIAL PRIMARY KEY,
    company                 TEXT NOT NULL,
    year                    INT  NOT NULL,
    period                  TEXT,                  -- e.g. "Q1 2025" or "FY 2025"

    -- Income Statement
    revenue                 FLOAT,                 -- Gross turnover / Net turnover
    gross_profit            FLOAT,
    operating_profit        FLOAT,                 -- EBIT
    profit_before_tax       FLOAT,
    net_profit              FLOAT,                 -- Profit after tax
    eps                     FLOAT,                 -- Earnings per share (PKR)
    dividend_per_share      FLOAT,

    -- Cost items
    cost_of_goods_sold      FLOAT,
    operating_expenses      FLOAT,
    depreciation            FLOAT,
    finance_cost            FLOAT,
    tax_expense             FLOAT,

    -- Balance Sheet — Assets
    total_assets            FLOAT,
    current_assets          FLOAT,
    non_current_assets      FLOAT,
    cash_balance            FLOAT,
    trade_receivables       FLOAT,
    inventory               FLOAT,

    -- Balance Sheet — Liabilities & Equity
    total_liabilities       FLOAT,
    current_liabilities     FLOAT,
    non_current_liabilities FLOAT,
    total_equity            FLOAT,
    share_capital           FLOAT,
    long_term_debt          FLOAT,

    -- Cash Flow
    operating_cashflow      FLOAT,
    investing_cashflow      FLOAT,
    financing_cashflow      FLOAT,

    UNIQUE(company, year, period)
);
"""


def create_schema(conn):
    cur = conn.cursor()
    cur.execute(SCHEMA_SQL)
    conn.commit()
    cur.close()


def get_connection():
    return psycopg2.connect(
        dbname="finance_db",
        user="akd",
        host="localhost"
    )


def insert_financials(data: dict, company: str = "Bestway Cement", year: int = 2025, period: str = "FY 2025"):
    conn = get_connection()
    create_schema(conn)
    cur = conn.cursor()

    fields = [
        "company", "year", "period",
        "revenue", "gross_profit", "operating_profit", "profit_before_tax", "net_profit",
        "eps", "dividend_per_share",
        "cost_of_goods_sold", "operating_expenses", "depreciation", "finance_cost", "tax_expense",
        "total_assets", "current_assets", "non_current_assets", "cash_balance",
        "trade_receivables", "inventory",
        "total_liabilities", "current_liabilities", "non_current_liabilities",
        "total_equity", "share_capital", "long_term_debt",
        "operating_cashflow", "investing_cashflow", "financing_cashflow"
    ]

    values = [company, year, period] + [data.get(f) for f in fields[3:]]
    placeholders = ", ".join(["%s"] * len(fields))
    col_names = ", ".join(fields)

    # Upsert: update on conflict
    update_clause = ", ".join([f"{f} = EXCLUDED.{f}" for f in fields[3:]])

    cur.execute(f"""
        INSERT INTO financials ({col_names})
        VALUES ({placeholders})
        ON CONFLICT (company, year, period) DO UPDATE SET {update_clause}
    """, values)

    conn.commit()
    cur.close()
    conn.close()
    print(f"✅ Inserted/updated: {company} | {period}")