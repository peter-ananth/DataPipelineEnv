"""
Task 3 — Hard: SQL Query Reverse Engineering
Agent sees only the expected output table. Must write a SQL query that produces it.
Upgraded to feature 12 advanced architectural SQL targeting!
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
SCHEMA_SQL = SCHEMA_DIR / "schema_hard.sql"

TASK_ID = "query_reverse"
TASK_NAME = "SQL Reverse Engineering"
DIFFICULTY = "hard"
MAX_ATTEMPTS = 7

# ─────────────────── Database setup ──────────────────────────────────────────

def setup_database(seed: int | None = None) -> sqlite3.Connection:
    """Create in-memory SQLite DB and procedurally populate with hard task data."""
    conn = sqlite3.connect(":memory:")
    schema = SCHEMA_SQL.read_text()
    conn.executescript(schema)

    rng = random.Random(seed)
    regions = ["North America", "Europe", "Asia", "South America"]
    salespersons = ["Alice", "Bob", "Charlie", "Diana", "Evan", "Fiona", "George", "Hannah"]
    categories = ["Electronics", "Software", "Hardware", "Services"]
    
    sales = []
    for i in range(1, 250):
        region = rng.choice(regions)
        person = rng.choice(salespersons)
        cat = rng.choice(categories)
        units = rng.randint(1, 100)
        revenue = round(units * rng.uniform(50.0, 500.0), 2)
        month = rng.randint(1, 12)
        year = rng.choice([2023, 2024])
        sales.append((10000+i, region, person, cat, revenue, units, month, year))
        
    conn.executemany("INSERT INTO sales VALUES (?, ?, ?, ?, ?, ?, ?, ?)", sales)
    conn.commit()
    
    # Execution Bounding (Timeout)
    def _abort(): return 1
    conn.set_progress_handler(_abort, 50_000_000)
    
    return conn

# ─────────────────── Target queries (what agents must reverse-engineer) ───────

_TARGET_QUERIES: list[dict] = [
    {
        "id": "top_revenue_by_region",
        "hint": "The expected output shows regions with their aggregated Electronics revenue for 2024, ordered highest to lowest.",
        "query": "SELECT region, SUM(revenue) AS total_revenue, SUM(units_sold) AS total_units FROM sales WHERE product_category = 'Electronics' AND sale_year = 2024 GROUP BY region ORDER BY total_revenue DESC",
    },
    {
        "id": "monthly_software_totals",
        "hint": "The expected output shows monthly aggregates for Software category only, ordered by year then revenue.",
        "query": "SELECT sale_month, sale_year, COUNT(*) AS num_transactions, SUM(revenue) AS monthly_revenue FROM sales WHERE product_category = 'Software' GROUP BY sale_year, sale_month ORDER BY sale_year, monthly_revenue DESC",
    },
    {
        "id": "top_salespersons",
        "hint": "The expected output shows only the top 3 performers of all time with their revenue and distinct category count.",
        "query": "SELECT salesperson, SUM(revenue) AS total_revenue, COUNT(DISTINCT product_category) AS categories_sold FROM sales GROUP BY salesperson ORDER BY total_revenue DESC LIMIT 3",
    },
    {
        "id": "revenue_per_unit_by_region",
        "hint": "Calculate the average revenue per unit sold for each region, sorted by the highest unit value.",
        "query": "SELECT region, SUM(revenue)/SUM(units_sold) AS avg_revenue_per_unit FROM sales GROUP BY region ORDER BY avg_revenue_per_unit DESC",
    },
    {
        "id": "top_salesperson_per_region",
        "hint": "Use a Window Function to find the single highest grossing salesperson for EACH region based on total revenue.",
        "query": "WITH Ranked AS (SELECT region, salesperson, SUM(revenue) as rev, ROW_NUMBER() OVER(PARTITION BY region ORDER BY SUM(revenue) DESC) as rnk FROM sales GROUP BY region, salesperson) SELECT region, salesperson, rev FROM Ranked WHERE rnk = 1",
    },
    {
        "id": "cumulative_running_total",
        "hint": "Calculate a running total (cumulative sum) of revenue by month for the year 2024.",
        "query": "SELECT sale_month, SUM(revenue) AS monthly_rev, SUM(SUM(revenue)) OVER (ORDER BY sale_month) AS cumulative_revenue FROM sales WHERE sale_year = 2024 GROUP BY sale_month",
    },
    {
        "id": "bottom_3_months",
        "hint": "Identify the 3 worst performing calendar months of all time by total revenue.",
        "query": "SELECT sale_year, sale_month, SUM(revenue) AS total_revenue FROM sales GROUP BY sale_year, sale_month ORDER BY total_revenue ASC LIMIT 3",
    },
    {
        "id": "above_average_categories",
        "hint": "Find product categories whose total revenue is STRICTLY GREATER than the global average revenue per category.",
        "query": "WITH CatTotal AS (SELECT product_category, SUM(revenue) as cat_rev FROM sales GROUP BY product_category), GlobalAvg AS (SELECT AVG(cat_rev) as g_avg FROM CatTotal) SELECT product_category, cat_rev FROM CatTotal, GlobalAvg WHERE cat_rev > g_avg ORDER BY cat_rev DESC",
    },
    {
        "id": "yoy_growth",
        "hint": "Compare total revenue of 2023 vs 2024 per region. You must pivot the years into columns.",
        "query": "SELECT region, SUM(CASE WHEN sale_year=2023 THEN revenue ELSE 0 END) AS rev_2023, SUM(CASE WHEN sale_year=2024 THEN revenue ELSE 0 END) AS rev_2024 FROM sales GROUP BY region ORDER BY region",
    },
    {
        "id": "salesperson_category_matrix",
        "hint": "Pivot table: Calculate total revenue for each salesperson across 'Hardware' and 'Software' specifically.",
        "query": "SELECT salesperson, SUM(CASE WHEN product_category='Hardware' THEN revenue ELSE 0 END) AS hardware_rev, SUM(CASE WHEN product_category='Software' THEN revenue ELSE 0 END) AS software_rev FROM sales GROUP BY salesperson ORDER BY hardware_rev DESC",
    },
    {
        "id": "high_ticket_sales",
        "hint": "Find all individual sales instances where the revenue strictly exceeds 5000.",
        "query": "SELECT sale_id, salesperson, region, revenue FROM sales WHERE revenue > 5000 ORDER BY revenue DESC",
    },
    {
        "id": "year_over_year_overall",
        "hint": "Total volume and revenue grouped strictly by year to see macro growth.",
        "query": "SELECT sale_year, SUM(units_sold) as total_volume, SUM(revenue) as gross_revenue FROM sales GROUP BY sale_year ORDER BY sale_year ASC",
    }
]

def get_target(seed: int | None = None) -> dict:
    import random
    rng = random.Random(seed)
    return rng.choice(_TARGET_QUERIES)

def get_expected_df(target: dict, conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(target["query"], conn)


SCHEMA_DESCRIPTION = (
    "sales(sale_id, region, salesperson, product_category, revenue, units_sold, sale_month, sale_year)"
)

DESCRIPTION_TEMPLATE = textwrap.dedent("""
    You are given a database with sales data and an expected output table.
    Your task is to write a SQL query that produces **exactly** this output.

    **Database schema:**
    {schema}

    **Expected output (first rows shown):**
    {expected_preview}

    **Hint:** {hint}

    **Action format:**
    {{
        "type": "submit_query",
        "payload": "SELECT ..."
    }}

    The grader will compare your result set order-independently against the expected output.
    Extra columns incur a penalty. Aim for an exact match.
""").strip()

def get_initial_observation(target: dict, conn: sqlite3.Connection) -> dict[str, Any]:
    expected_df = get_expected_df(target, conn)
    preview = expected_df.head(8).to_csv(index=False)
    description = DESCRIPTION_TEMPLATE.format(
        schema=SCHEMA_DESCRIPTION,
        expected_preview=preview,
        hint=target["hint"],
    )
    return {
        "task_id": TASK_ID,
        "task_description": description,
        "data_preview": preview,
        "schema": SCHEMA_DESCRIPTION,
        "error_message": None,
        "attempt": 1,
        "max_attempts": MAX_ATTEMPTS,
    }
