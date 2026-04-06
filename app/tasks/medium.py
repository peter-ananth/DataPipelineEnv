"""
Task 2 — Medium: SQL Query Debugging
Agent receives a broken SQL query + schema. Must fix and submit a working query.
Expanded bug catalog from 4 to 15 robust SQL variants!
"""

from __future__ import annotations

import random
import sqlite3
import textwrap
from pathlib import Path
from typing import Any

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
SCHEMA_DIR = Path(__file__).parent.parent / "schemas"
SCHEMA_SQL = SCHEMA_DIR / "schema_medium.sql"

TASK_ID = "sql_fix"
TASK_NAME = "SQL Query Debugging"
DIFFICULTY = "medium"
MAX_ATTEMPTS = 5

# ─────────────────── Database setup ──────────────────────────────────────────

def setup_database(seed: int | None = None) -> sqlite3.Connection:
    """Create in-memory SQLite DB and procedurally populate with medium task data."""
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA journal_mode=WAL")
    schema = SCHEMA_SQL.read_text()
    conn.executescript(schema)

    rng = random.Random(seed)
    
    names = ["Alice Johnson", "Bob Smith", "Carol White", "David Brown", "Eve Davis", "Frank Miller", "Grace Lee", "Henry Wang", "Null Country Corp"]
    countries = ["United States", "United Kingdom", "Canada", "Germany", "France", "Australia", "Japan", "China"]
    
    customers = []
    for i, name in enumerate(names, start=101):
        c_country = rng.choice(countries) if rng.random() > 0.15 else None
        customers.append((i, name, f"{name.split()[0].lower()}@example.com", c_country))
        
    conn.executemany("INSERT INTO customers VALUES (?, ?, ?, ?)", customers)

    products = [
        (501, "Mouse", "Peripherals", 25.50),
        (502, "Monitor", "Displays", 349.99),
        (503, "Keyboard", "Peripherals", 75.00),
        (504, "Headphones", "Audio", 129.99),
        (505, "Webcam", "Accessories", 89.99),
        (506, "Laptop Stand", "Accessories", 55.00),
        (507, "SSD", "Storage", 89.99),
        (508, "Charger", "Accessories", 29.99),
    ]
    conn.executemany("INSERT INTO products VALUES (?, ?, ?, ?)", products)

    orders = []
    statuses = ["completed", "completed", "completed", "pending", "cancelled"]
    for i in range(1, 45):
        c_id = rng.choice(customers)[0]
        p = rng.choice(products)
        qty = rng.randint(1, 5)
        status = rng.choice(statuses)
        date = f"2024-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}"
        orders.append((1000+i, c_id, p[0], qty, p[3], date, status))
        
    conn.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?)", orders)
    conn.commit()
    return conn

# ─────────────────── Bug variants (15 Variants) ────────────────────────────────

_BUG_VARIANTS: list[dict] = [
    {
        "id": "missing_where",
        "description": "Missing WHERE clause that should filter for 'completed' status only",
        "error_message": "Query runs but returns cancelled and pending orders.",
        "broken_query": "SELECT c.customer_name, SUM(o.quantity * o.unit_price) AS rev \nFROM orders o \nINNER JOIN customers c ON o.customer_id = c.customer_id \nGROUP BY c.customer_name ORDER BY rev DESC",
        "correct_query": "SELECT c.customer_name, SUM(o.quantity * o.unit_price) AS rev \nFROM orders o \nINNER JOIN customers c ON o.customer_id = c.customer_id \nWHERE o.status = 'completed'\nGROUP BY c.customer_name ORDER BY rev DESC",
    },
    {
        "id": "wrong_join",
        "description": "CROSS JOIN used instead of INNER JOIN causing a Cartesian product",
        "error_message": "Returns too many rows (Cartesian explosion).",
        "broken_query": "SELECT c.customer_name, COUNT(o.order_id) AS cnt \nFROM orders o CROSS JOIN customers c \nWHERE o.status = 'completed' GROUP BY c.customer_name",
        "correct_query": "SELECT c.customer_name, COUNT(o.order_id) AS cnt \nFROM orders o INNER JOIN customers c ON o.customer_id = c.customer_id \nWHERE o.status = 'completed' GROUP BY c.customer_name",
    },
    {
        "id": "syntax_error",
        "description": "Syntax error: missing comma between SELECT columns",
        "error_message": "OperationalError: near 'o': syntax error",
        "broken_query": "SELECT o.order_id o.status FROM orders o",
        "correct_query": "SELECT o.order_id, o.status FROM orders o",
    },
    {
        "id": "wrong_column",
        "description": "Incorrect column reference: filtering on wrong column name 'order_status'",
        "error_message": "OperationalError: no such column: o.order_status",
        "broken_query": "SELECT * FROM orders o WHERE o.order_status = 'completed'",
        "correct_query": "SELECT * FROM orders o WHERE o.status = 'completed'",
    },
    {
        "id": "having_vs_where",
        "description": "Using WHERE instead of HAVING to filter aggregate values",
        "error_message": "OperationalError: misuse of aggregate: SUM()",
        "broken_query": "SELECT c.customer_name, SUM(o.quantity) as q \nFROM orders o INNER JOIN customers c ON o.customer_id = c.customer_id \nWHERE SUM(o.quantity) > 5 GROUP BY c.customer_name",
        "correct_query": "SELECT c.customer_name, SUM(o.quantity) as q \nFROM orders o INNER JOIN customers c ON o.customer_id = c.customer_id \nGROUP BY c.customer_name HAVING SUM(o.quantity) > 5",
    },
    {
        "id": "null_equality",
        "description": "Using '= NULL' instead of 'IS NULL'",
        "error_message": "Query runs but returns 0 rows due to NULL equality behavior.",
        "broken_query": "SELECT customer_name FROM customers WHERE country = NULL",
        "correct_query": "SELECT customer_name FROM customers WHERE country IS NULL",
    },
    {
        "id": "logical_precedence",
        "description": "Missing parentheses around OR condition in WHERE clause",
        "error_message": "Returns cancelled Australia orders due to bad AND/OR precedence.",
        "broken_query": "SELECT * FROM orders o INNER JOIN customers c ON o.customer_id=c.customer_id \nWHERE o.status='completed' OR o.status='pending' AND c.country='Australia'",
        "correct_query": "SELECT * FROM orders o INNER JOIN customers c ON o.customer_id=c.customer_id \nWHERE (o.status='completed' OR o.status='pending') AND c.country='Australia'",
    },
    {
        "id": "wildcard_misuse",
        "description": "Missing wildcard % in LIKE clause",
        "error_message": "Returns 0 rows because exact match fails on substring.",
        "broken_query": "SELECT * FROM products WHERE product_name LIKE 'Monitor'",
        "correct_query": "SELECT * FROM products WHERE product_name LIKE '%Monitor%'",
    },
    {
        "id": "join_mismatch",
        "description": "Joining on entirely the wrong foreign key (Semantic Error)",
        "error_message": "Matches customer IDs directly to product IDs, yielding empty/garbage rows.",
        "broken_query": "SELECT o.order_id, p.product_name FROM orders o INNER JOIN products p ON o.customer_id = p.product_id",
        "correct_query": "SELECT o.order_id, p.product_name FROM orders o INNER JOIN products p ON o.product_id = p.product_id",
    },
    {
        "id": "order_direction",
        "description": "Sorting Top-N queries ASC when DESC is required",
        "error_message": "Returns the lowest revenue customers instead of the highest.",
        "broken_query": "SELECT customer_id, SUM(unit_price * quantity) as total FROM orders GROUP BY customer_id ORDER BY total ASC LIMIT 3",
        "correct_query": "SELECT customer_id, SUM(unit_price * quantity) as total FROM orders GROUP BY customer_id ORDER BY total DESC LIMIT 3",
    },
    {
        "id": "missing_group_by",
        "description": "Using aggregate functions without GROUP BY, collapsing all rows to 1",
        "error_message": "Returns only 1 row summing everything, instead of aggregating per product.",
        "broken_query": "SELECT p.category, SUM(o.quantity) FROM orders o INNER JOIN products p ON o.product_id=p.product_id",
        "correct_query": "SELECT p.category, SUM(o.quantity) FROM orders o INNER JOIN products p ON o.product_id=p.product_id GROUP BY p.category",
    },
    {
        "id": "string_case",
        "description": "SQLite string comparisons are case-sensitive by default",
        "error_message": "Query returns 0 rows because 'canada' is lowercase.",
        "broken_query": "SELECT * FROM customers WHERE country = 'canada'",
        "correct_query": "SELECT * FROM customers WHERE country = 'Canada'",
    },
    {
        "id": "aggregate_alias_where",
        "description": "Trying to filter on an alias introduced in the SELECT clause inside the WHERE clause",
        "error_message": "OperationalError: no such column: total_price",
        "broken_query": "SELECT quantity * unit_price AS total_price FROM orders WHERE total_price > 100",
        "correct_query": "SELECT quantity * unit_price AS total_price FROM orders WHERE (quantity * unit_price) > 100",
    },
    {
        "id": "distinct_count",
        "description": "Counting all rows instead of unique values",
        "error_message": "Returns 9 instead of the expected 6 distinct customer_ids.",
        "broken_query": "SELECT COUNT(customer_id) as unique_buyers FROM orders",
        "correct_query": "SELECT COUNT(DISTINCT customer_id) as unique_buyers FROM orders",
    },
    {
        "id": "limit_offset_typo",
        "description": "Incorrect LIMIT / OFFSET syntax parsing",
        "error_message": "OperationalError: near 'OFFSET': syntax error",
        "broken_query": "SELECT * FROM orders ORDER BY order_id DESC LIMIT 5, OFFSET 10",
        "correct_query": "SELECT * FROM orders ORDER BY order_id DESC LIMIT 5 OFFSET 10",
    }
]

def get_random_bug(seed: int | None = None) -> dict:
    rng = random.Random(seed)
    return rng.choice(_BUG_VARIANTS)

def get_expected_df(correct_query: str, conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(correct_query, conn)


DESCRIPTION_TEMPLATE = textwrap.dedent("""
    You are given a broken SQL query and the database schema below.
    Fix the query so it produces the correct result set.

    **Bug description:** {description}
    **Error encountered:** {error_message}

    **Schema:**
    {schema}

    **Broken query:**
    ```sql
    {broken_query}
    ```

    **Action format:**
    {{
        "type": "submit_query",
        "payload": "SELECT ..."
    }}
""").strip()

def get_schema_description() -> str:
    return SCHEMA_SQL.read_text()

def get_initial_observation(bug: dict, conn: sqlite3.Connection) -> dict[str, Any]:
    schema_desc = (
        "customers(customer_id, customer_name, email, country) | "
        "products(product_id, product_name, category, base_price) | "
        "orders(order_id, customer_id, product_id, quantity, unit_price, order_date, status)"
    )
    description = DESCRIPTION_TEMPLATE.format(
        description=bug["description"],
        error_message=bug["error_message"],
        schema=schema_desc,
        broken_query=bug["broken_query"],
    )
    return {
        "task_id": TASK_ID,
        "task_description": description,
        "data_preview": schema_desc,
        "schema": schema_desc,
        "error_message": bug["error_message"],
        "attempt": 1,
        "max_attempts": MAX_ATTEMPTS,
    }
