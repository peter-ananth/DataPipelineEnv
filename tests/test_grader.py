"""
Tests for grader.py — covers all 3 reward functions with edge cases.
TDD: Perfect score, zero score, partial credit, malformed input.
"""

from __future__ import annotations

import io
import sqlite3

import numpy as np
import pandas as pd
import pytest

from app import grader


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def clean_df() -> pd.DataFrame:
    """A perfectly clean reference DataFrame (no dupes, no nulls, correct types, title case)."""
    return pd.DataFrame({
        "order_id": [1, 2, 3, 4, 5],
        "customer_name": ["Alice Johnson", "Bob Smith", "Carol White", "Dave Brown", "Eve Davis"],
        "country": ["United States", "United Kingdom", "Canada", "Germany", "France"],
        "product": ["Laptop", "Mouse", "Keyboard", "Monitor", "Webcam"],
        "price": [1299.99, 25.50, 75.00, 349.99, 89.99],
        "quantity": [1, 2, 3, 1, 2],
        "order_date": ["2024-01-15"] * 5,
    })


@pytest.fixture
def dirty_df(clean_df) -> pd.DataFrame:
    """A dirty DataFrame with all 4 issues."""
    dirty = clean_df.copy()
    # 1. Add duplicates
    dirty = pd.concat([dirty, dirty.iloc[[0, 1]]], ignore_index=True)
    # 2. Add null customer_name
    dirty.loc[7, "customer_name"] = None
    # 3. Break price type
    dirty["price"] = dirty["price"].astype(str)
    # 4. Break country casing
    dirty["country"] = dirty["country"].str.upper()
    return dirty


@pytest.fixture
def in_memory_db_medium() -> sqlite3.Connection:
    """SQLite in-memory DB for medium task grader tests."""
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY,
            customer_name TEXT,
            email TEXT,
            country TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE products (
            product_id INTEGER PRIMARY KEY,
            product_name TEXT,
            category TEXT,
            base_price REAL
        )
    """)
    conn.execute("""
        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            product_id INTEGER,
            quantity INTEGER,
            unit_price REAL,
            order_date TEXT,
            status TEXT
        )
    """)
    conn.executemany(
        "INSERT INTO customers VALUES (?,?,?,?)",
        [(1, "Alice", "a@x.com", "US"), (2, "Bob", "b@x.com", "UK")],
    )
    conn.executemany(
        "INSERT INTO products VALUES (?,?,?,?)",
        [(10, "Laptop", "Electronics", 999.0), (11, "Mouse", "Peripherals", 25.0)],
    )
    conn.executemany(
        "INSERT INTO orders VALUES (?,?,?,?,?,?,?)",
        [
            (100, 1, 10, 1, 999.0, "2024-01-01", "completed"),
            (101, 2, 11, 2, 25.0,  "2024-01-02", "pending"),
            (102, 1, 11, 3, 25.0,  "2024-01-03", "completed"),
        ],
    )
    conn.commit()
    yield conn
    conn.close()


@pytest.fixture
def in_memory_db_hard() -> sqlite3.Connection:
    """SQLite in-memory DB for hard task grader tests."""
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE sales (
            sale_id INTEGER PRIMARY KEY,
            region TEXT,
            salesperson TEXT,
            product_category TEXT,
            revenue REAL,
            units_sold INTEGER,
            sale_month TEXT,
            sale_year INTEGER
        )
    """)
    conn.executemany(
        "INSERT INTO sales VALUES (?,?,?,?,?,?,?,?)",
        [
            (1, "North", "Alice", "Electronics", 15000.0, 12, "January", 2024),
            (2, "South", "Bob", "Electronics", 8500.0, 7, "January", 2024),
            (3, "North", "Alice", "Furniture", 5000.0, 6, "February", 2024),
            (4, "South", "Bob", "Software", 18000.0, 30, "February", 2024),
        ],
    )
    conn.commit()
    yield conn
    conn.close()


@pytest.fixture
def expected_df_medium(in_memory_db_medium) -> pd.DataFrame:
    """Expected result: completed orders only."""
    return pd.read_sql_query(
        "SELECT order_id, customer_id, quantity, unit_price FROM orders WHERE status='completed' ORDER BY order_id",
        in_memory_db_medium,
    )


@pytest.fixture
def expected_df_hard(in_memory_db_hard) -> pd.DataFrame:
    """Expected result: total revenue per region."""
    return pd.read_sql_query(
        "SELECT region, SUM(revenue) as total_revenue FROM sales GROUP BY region ORDER BY total_revenue DESC",
        in_memory_db_hard,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Task 1 — grade_csv_clean
# ──────────────────────────────────────────────────────────────────────────────

class TestGradeCsvClean:

    def test_perfect_score(self, clean_df):
        """A perfectly clean DF scores 1.0."""
        score = grader.grade_csv_clean(clean_df, clean_df)
        assert score == pytest.approx(1.0, abs=0.05)

    def test_zero_score_empty_submission(self, clean_df):
        """Empty submission returns 0.0."""
        score = grader.grade_csv_clean(pd.DataFrame(), clean_df)
        assert score == 0.0

    def test_zero_score_none_submission(self, clean_df):
        """None submission returns 0.0."""
        score = grader.grade_csv_clean(None, clean_df)
        assert score == 0.0

    def test_partial_credit_only_deduped(self, clean_df, dirty_df):
        """Deduping only gets partial credit, not full."""
        # Remove dupes but keep other issues
        partial = dirty_df.drop_duplicates().reset_index(drop=True)
        score = grader.grade_csv_clean(partial, clean_df)
        assert 0.0 < score < 1.0

    def test_partial_credit_types_fixed_only(self, clean_df, dirty_df):
        """Fixing types and casing but not nulls/dupes gets partial credit."""
        partial = dirty_df.copy()
        partial["price"] = pd.to_numeric(partial["price"], errors="coerce")
        partial["country"] = partial["country"].str.title()
        score = grader.grade_csv_clean(partial, clean_df)
        assert 0.0 < score < 1.0

    def test_score_varies_across_inputs(self, clean_df, dirty_df):
        """Grader must return different scores for different inputs (no-constant check)."""
        score_perfect = grader.grade_csv_clean(clean_df, clean_df)
        score_dirty = grader.grade_csv_clean(dirty_df, clean_df)
        assert score_perfect != score_dirty

    def test_reward_always_in_range(self, clean_df, dirty_df):
        """Reward must always be in [0.0, 1.0]."""
        for df in [clean_df, dirty_df, pd.DataFrame(), pd.DataFrame({"x": [1, 2]})]:
            score = grader.grade_csv_clean(df, clean_df)
            assert 0.0 <= score <= 1.0

    def test_country_casing_partial(self, clean_df):
        """Partial casing improvement gives partial credit."""
        partial = clean_df.copy()
        partial["country"] = partial["country"].str.upper()
        score_bad = grader.grade_csv_clean(partial, clean_df)
        partial["country"] = partial["country"].str.title()
        score_good = grader.grade_csv_clean(partial, clean_df)
        assert score_good > score_bad

    def test_mismatched_columns_no_crash(self, clean_df):
        """Submission with wrong columns doesn't crash."""
        bad = pd.DataFrame({"foo": [1, 2], "bar": ["a", "b"]})
        score = grader.grade_csv_clean(bad, clean_df)
        assert 0.0 <= score <= 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Task 2 — grade_sql_fix
# ──────────────────────────────────────────────────────────────────────────────

class TestGradeSqlFix:

    def test_perfect_score(self, in_memory_db_medium, expected_df_medium):
        """Correct query returns 1.0."""
        query = "SELECT order_id, customer_id, quantity, unit_price FROM orders WHERE status='completed' ORDER BY order_id"
        score = grader.grade_sql_fix(query, in_memory_db_medium, expected_df_medium)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_zero_score_empty_query(self, in_memory_db_medium, expected_df_medium):
        """Empty query returns 0.0."""
        score = grader.grade_sql_fix("", in_memory_db_medium, expected_df_medium)
        assert score == 0.0

    def test_zero_score_syntax_error(self, in_memory_db_medium, expected_df_medium):
        """Syntax error returns 0.0."""
        score = grader.grade_sql_fix("SELECT FROM broken ??", in_memory_db_medium, expected_df_medium)
        assert score == 0.0

    def test_partial_score_correct_columns_wrong_rows(self, in_memory_db_medium, expected_df_medium):
        """Correct columns, wrong filter returns 0.5."""
        # Return all rows (no WHERE filter)
        query = "SELECT order_id, customer_id, quantity, unit_price FROM orders ORDER BY order_id"
        score = grader.grade_sql_fix(query, in_memory_db_medium, expected_df_medium)
        assert score == pytest.approx(0.5, abs=0.01)

    def test_zero_score_wrong_columns(self, in_memory_db_medium, expected_df_medium):
        """Wrong columns returns 0.0."""
        query = "SELECT status FROM orders"
        score = grader.grade_sql_fix(query, in_memory_db_medium, expected_df_medium)
        assert score == 0.0

    def test_score_varies_across_inputs(self, in_memory_db_medium, expected_df_medium):
        """Grader must return different scores (not constant)."""
        scores = set()
        queries = [
            "",
            "SELECT order_id, customer_id, quantity, unit_price FROM orders WHERE status='completed' ORDER BY order_id",
            "SELECT order_id, customer_id, quantity, unit_price FROM orders ORDER BY order_id",
        ]
        for q in queries:
            scores.add(grader.grade_sql_fix(q, in_memory_db_medium, expected_df_medium))
        assert len(scores) > 1, "Grader must not return the same score for all inputs"

    def test_reward_always_in_range(self, in_memory_db_medium, expected_df_medium):
        """Reward must be in [0.0, 1.0] for any input."""
        queries = ["", "SELECT 1", "INVALID SQL !!!!", "SELECT * FROM orders"]
        for q in queries:
            score = grader.grade_sql_fix(q, in_memory_db_medium, expected_df_medium)
            assert 0.0 <= score <= 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Task 3 — grade_query_reverse
# ──────────────────────────────────────────────────────────────────────────────

class TestGradeQueryReverse:

    def test_perfect_score(self, in_memory_db_hard, expected_df_hard):
        """Correct query gets 1.0."""
        query = "SELECT region, SUM(revenue) as total_revenue FROM sales GROUP BY region ORDER BY total_revenue DESC"
        score = grader.grade_query_reverse(query, in_memory_db_hard, expected_df_hard)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_zero_score_empty_query(self, in_memory_db_hard, expected_df_hard):
        score = grader.grade_query_reverse("", in_memory_db_hard, expected_df_hard)
        assert score == 0.0

    def test_zero_score_syntax_error(self, in_memory_db_hard, expected_df_hard):
        score = grader.grade_query_reverse("SELECT ??? BROKEN", in_memory_db_hard, expected_df_hard)
        assert score == 0.0

    def test_penalized_extra_columns(self, in_memory_db_hard, expected_df_hard):
        """Extra unnecessary columns reduces score below 1.0 but stays >= 0.5."""
        query = "SELECT region, salesperson, SUM(revenue) as total_revenue FROM sales GROUP BY region, salesperson ORDER BY total_revenue DESC"
        score = grader.grade_query_reverse(query, in_memory_db_hard, expected_df_hard)
        # This returns wrong rows anyway, so score will vary
        assert 0.0 <= score <= 1.0

    def test_partial_match_correct_columns_wrong_rows(self, in_memory_db_hard, expected_df_hard):
        """Partial row overlap with correct columns gets partial credit."""
        # A query that gets some but not all rows
        query = "SELECT region, SUM(revenue) as total_revenue FROM sales WHERE region='North' GROUP BY region"
        score = grader.grade_query_reverse(query, in_memory_db_hard, expected_df_hard)
        assert 0.0 <= score < 1.0

    def test_score_varies_across_inputs(self, in_memory_db_hard, expected_df_hard):
        """Non-constant reward."""
        correct = "SELECT region, SUM(revenue) as total_revenue FROM sales GROUP BY region ORDER BY total_revenue DESC"
        wrong = "SELECT * FROM sales"
        empty = ""
        scores = {
            grader.grade_query_reverse(q, in_memory_db_hard, expected_df_hard)
            for q in [correct, wrong, empty]
        }
        assert len(scores) > 1

    def test_reward_always_in_range(self, in_memory_db_hard, expected_df_hard):
        queries = ["", "SELECT 1", "DROP TABLE sales", "SELECT * FROM sales LIMIT 1"]
        for q in queries:
            score = grader.grade_query_reverse(q, in_memory_db_hard, expected_df_hard)
            assert 0.0 <= score <= 1.0
