import psycopg2

# Full schema DDL — run once to set up the database
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS financials (
    id                      SERIAL PRIMARY KEY,
    company                 TEXT NOT NULL,
    year                    INT  NOT NULL,
    period                  TEXT,

    -- Currency / unit metadata (from extractor)
    currency                TEXT DEFAULT 'Unknown',
    unit_label              TEXT DEFAULT 'millions (assumed)',

    -- Income Statement
    revenue                 FLOAT,
    gross_profit            FLOAT,
    operating_profit        FLOAT,
    profit_before_tax       FLOAT,
    net_profit              FLOAT,
    eps                     FLOAT,
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

# Safe migration — adds currency/unit_label to tables created before this schema
_MIGRATE_SQL = """
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name='financials' AND column_name='currency'
    ) THEN
        ALTER TABLE financials
            ADD COLUMN currency   TEXT DEFAULT 'Unknown',
            ADD COLUMN unit_label TEXT DEFAULT 'millions (assumed)';
        RAISE NOTICE 'Migrated: added currency + unit_label columns';
    END IF;
END$$;
"""

# Financial data fields (no metadata cols)
_DATA_FIELDS = [
    "revenue", "gross_profit", "operating_profit", "profit_before_tax", "net_profit",
    "eps", "dividend_per_share",
    "cost_of_goods_sold", "operating_expenses", "depreciation", "finance_cost", "tax_expense",
    "total_assets", "current_assets", "non_current_assets", "cash_balance",
    "trade_receivables", "inventory",
    "total_liabilities", "current_liabilities", "non_current_liabilities",
    "total_equity", "share_capital", "long_term_debt",
    "operating_cashflow", "investing_cashflow", "financing_cashflow",
]


_SCHEMA_READY = False


def create_schema(conn=None, force: bool = False):
    """
    Create/migrate the financials table.
    Runs on its OWN autocommit connection so DDL never deadlocks
    with any open read transactions on other connections.

    Idempotent: subsequent calls in the same process are no-ops unless
    force=True. This avoids 2 DDL probes per insert.
    """
    global _SCHEMA_READY
    if _SCHEMA_READY and not force:
        return
    ac = get_connection()
    ac.set_session(autocommit=True)   # DDL outside any transaction block
    try:
        cur = ac.cursor()
        try:
            cur.execute(SCHEMA_SQL)
            cur.execute(_MIGRATE_SQL)
        finally:
            cur.close()
    finally:
        ac.close()
    _SCHEMA_READY = True


def get_connection():
    """Open a connection to the local Postgres instance backing FinEx."""
    return psycopg2.connect(
        dbname="finance_db",
        user="akd",
        host="localhost"
    )


def _col_exists(conn, col: str) -> bool:
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_name='financials' AND column_name=%s",
        (col,)
    )
    result = cur.fetchone() is not None
    cur.close()
    return result


def insert_financials(
    data: dict,
    company: str = "Bestway Cement",
    year: int = 2025,
    period: str = "FY 2025",
    currency: str = "Unknown",
    unit_label: str = "millions (assumed)",
):
    # Step 1 — ensure schema exists (runs on its own autocommit connection)
    try:
        create_schema()
    except Exception as e:
        print(f"⚠ Schema/migration warning (non-fatal): {e}")

    # Step 2 — insert on a fresh dedicated connection
    conn = get_connection()
    cur = conn.cursor()
    # Fail fast (10s) if another connection holds a conflicting lock
    cur.execute("SET lock_timeout = '10s'")
    cur.execute("SET statement_timeout = '30s'")

    # Check if currency/unit_label columns exist (they might not on very old installs)
    has_meta = _col_exists(conn, "currency")

    if has_meta:
        fields = ["company", "year", "period", "currency", "unit_label"] + _DATA_FIELDS
        values = [company, year, period, currency, unit_label] + [data.get(f) for f in _DATA_FIELDS]
    else:
        # Fallback: insert without meta columns so data is never lost
        print("⚠ currency/unit_label columns missing — inserting without them")
        fields = ["company", "year", "period"] + _DATA_FIELDS
        values = [company, year, period] + [data.get(f) for f in _DATA_FIELDS]

    placeholders  = ", ".join(["%s"] * len(fields))
    col_names     = ", ".join(fields)
    update_clause = ", ".join([f"{f} = EXCLUDED.{f}" for f in fields[3:]])

    try:
        cur.execute(f"""
            INSERT INTO financials ({col_names})
            VALUES ({placeholders})
            ON CONFLICT (company, year, period) DO UPDATE SET {update_clause}
        """, values)
        conn.commit()
        print(f"✅ Inserted/updated: {company} | {period} | {currency} | {unit_label}")
    except Exception as e:
        conn.rollback()
        print(f"❌ Insert failed: {e}")
        raise
    finally:
        cur.close()
        conn.close()
