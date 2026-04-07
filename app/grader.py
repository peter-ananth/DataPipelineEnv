"""
DataPipelineEnv — Grader Module
Deterministic, pure reward functions for all 3 tasks.
All functions return float in [0.0, 1.0].
All exceptions are caught: never crashes, returns 0.0 on failures.
"""

from __future__ import annotations

import io
import sqlite3
from typing import Optional

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# TASK 1: CSV Cleaning
# ──────────────────────────────────────────────────────────────────────────────

def grade_csv_clean(submitted_df: pd.DataFrame, reference_df: pd.DataFrame) -> float:
    """
    Grade a cleaned CSV submission against the reference DataFrame.
    All criteria are derived dynamically from reference_df.

    Partial credit breakdown:
      +0.25  Row count matches reference (deduplication + any row filters)
      +0.25  Null count matches reference (across all common columns)
      +0.25  Numeric types correct (price -> float, quantity -> int where applicable)
      +0.25  Country casing matches reference (upper / lower / title)

    Returns float in [0.0, 1.0].
    """
    try:
        if submitted_df is None or submitted_df.empty:
            return 0.0
        if reference_df is None or reference_df.empty:
            return 0.0

        score = 0.0

        # ── Criterion 1: Duplicates removed ────────────────────────────────
        ref_rows = len(reference_df)
        sub_rows = len(submitted_df)
        # Allow ±2 tolerance for minor differences
        if abs(sub_rows - ref_rows) <= 2:
            score += 0.25
        elif sub_rows <= ref_rows:
            # Partial: fewer rows than reference but still reduced some dupes
            score += 0.10

        # ── Criterion 2: Null handling — compare against reference ───────────
        common_cols = [c for c in reference_df.columns if c in submitted_df.columns]
        ref_total_nulls = reference_df[common_cols].isna().sum().sum()
        sub_total_nulls = submitted_df[common_cols].isna().sum().sum()
        if sub_total_nulls == ref_total_nulls:
            score += 0.25
        elif sub_total_nulls <= ref_total_nulls + 2:
            score += 0.10

        # ── Criterion 3: Data types ─────────────────────────────────────────
        type_score = 0.0
        if "price" in submitted_df.columns:
            try:
                _ = submitted_df["price"].astype(float)
                if submitted_df["price"].dtype in [np.float64, np.float32, float]:
                    type_score += 0.5
                else:
                    # Values are numeric but maybe object type
                    pd.to_numeric(submitted_df["price"], errors="raise")
                    type_score += 0.5
            except Exception:
                pass

        if "quantity" in submitted_df.columns and "quantity" in reference_df.columns:
            try:
                sub_qty = submitted_df["quantity"].dropna()
                is_int_type = submitted_df["quantity"].dtype in [np.int64, np.int32, int]
                is_nullable_int = str(submitted_df["quantity"].dtype) == "Int64"
                is_float_int = (
                    submitted_df["quantity"].dtype in [np.float64, np.float32, float]
                    and sub_qty.apply(float.is_integer).all()
                )
                if is_int_type or is_nullable_int or (is_float_int and not sub_qty.empty):
                    type_score += 0.5
                else:
                    pd.to_numeric(submitted_df["quantity"], errors="raise")
                    type_score += 0.3
            except Exception:
                pass

        score += 0.25 * type_score

        # ── Criterion 4: Country casing normalization ───────────────────────
        if "country" in submitted_df.columns and "country" in reference_df.columns:
            valid_sub = submitted_df["country"].dropna()
            valid_ref_str = reference_df["country"].dropna().astype(str)
            if len(valid_sub) > 0 and len(valid_ref_str) > 0:
                is_upper = (valid_ref_str == valid_ref_str.str.upper()).all()
                is_lower = (valid_ref_str == valid_ref_str.str.lower()).all()
                
                if is_upper:
                    cased = valid_sub.apply(lambda x: x == x.upper() if isinstance(x, str) else False)
                elif is_lower:
                    cased = valid_sub.apply(lambda x: x == x.lower() if isinstance(x, str) else False)
                else:
                    cased = valid_sub.apply(lambda x: x == x.title() if isinstance(x, str) else False)
                
                casing_ratio = cased.mean()
                if casing_ratio >= 0.95:
                    score += 0.25
                elif casing_ratio >= 0.70:
                    score += 0.12

        return float(np.clip(score, 0.0, 1.0))

    except Exception:
        return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# TASK 2: SQL Fix (Medium)
# ──────────────────────────────────────────────────────────────────────────────

def grade_sql_fix(
    query_str: str,
    db_conn: sqlite3.Connection,
    expected_df: pd.DataFrame,
) -> float:
    """
    Execute the submitted query and compare result set against expected_df.

    Scoring:
      1.0 — Result set matches exactly (order-independent)
      0.5 — Correct columns, rows differ (wrong WHERE/JOIN filter)
      0.0 — Wrong output, syntax error, or exception

    Returns float in [0.0, 1.0].
    """
    try:
        if not query_str or not query_str.strip():
            return 0.0

        result_df = _execute_query_safe(query_str, db_conn)
        if result_df is None:
            return 0.0
        if result_df.empty and not expected_df.empty:
            return 0.0

        # Normalize both for comparison
        result_norm = _normalize_df(result_df)
        expected_norm = _normalize_df(expected_df)

        # Exact match (order-independent)
        if _dataframes_equal(result_norm, expected_norm):
            return 1.0

        # Partial: correct columns, wrong rows
        if set(result_norm.columns) == set(expected_norm.columns):
            return 0.5

        return 0.0

    except Exception:
        return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# TASK 3: Query Reverse Engineering (Hard)
# ──────────────────────────────────────────────────────────────────────────────

def grade_query_reverse(
    query_str: str,
    db_conn: sqlite3.Connection,
    expected_df: pd.DataFrame,
) -> float:
    """
    Execute the submitted query and compare against expected_df.
    Penalizes extra unnecessary columns.

    Scoring:
      1.0 — Result set matches exactly (order-independent, correct columns)
      0.7 — Correct rows, extra unnecessary columns
      0.4 — Correct columns, wrong rows (partial filter match, >50% row overlap)
      0.0 — Wrong output, syntax error, or exception

    Returns float in [0.0, 1.0].
    """
    try:
        if not query_str or not query_str.strip():
            return 0.0

        result_df = _execute_query_safe(query_str, db_conn)
        if result_df is None:
            return 0.0

        result_norm = _normalize_df(result_df)
        expected_norm = _normalize_df(expected_df)

        # Exact match
        if _dataframes_equal(result_norm, expected_norm):
            return 1.0

        expected_cols = set(expected_norm.columns)
        result_cols = set(result_norm.columns)
        extra_cols = result_cols - expected_cols
        missing_cols = expected_cols - result_cols

        # Case: correct rows, extra unnecessary columns
        if not missing_cols and extra_cols:
            # Check rows match on expected columns
            result_subset = result_norm[list(expected_cols)]
            if _dataframes_equal(result_subset, expected_norm):
                # Penalize -0.1 per extra column, min 0.5
                penalty = min(0.2, 0.1 * len(extra_cols))
                return float(np.clip(0.7 - penalty + 0.3 * (1 - penalty), 0.5, 0.7))

        # Case: correct columns, wrong rows (partial match)
        if not missing_cols and not extra_cols:
            overlap = _row_overlap_ratio(result_norm, expected_norm)
            if overlap >= 0.5:
                return 0.4
            elif overlap > 0:
                return 0.2

        return 0.0

    except Exception:
        return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _execute_query_safe(
    query_str: str, db_conn: sqlite3.Connection
) -> Optional[pd.DataFrame]:
    """Execute a SQL query safely; return None on any error."""
    try:
        result = pd.read_sql_query(query_str, db_conn)
        return result
    except Exception:
        return None


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize DataFrame for comparison: lower column names, sort rows."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    # Convert numeric columns
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    # Sort by all columns for order-independent comparison
    try:
        df = df.sort_values(by=list(df.columns)).reset_index(drop=True)
    except Exception:
        df = df.reset_index(drop=True)
    return df


def _dataframes_equal(df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
    """Compare two normalized DataFrames for equality."""
    try:
        if set(df1.columns) != set(df2.columns):
            return False
        df1 = df1[sorted(df1.columns)].reset_index(drop=True)
        df2 = df2[sorted(df2.columns)].reset_index(drop=True)
        if df1.shape != df2.shape:
            return False
        return df1.equals(df2)
    except Exception:
        return False


def _row_overlap_ratio(df1: pd.DataFrame, df2: pd.DataFrame) -> float:
    """Return fraction of df2 rows that exist in df1."""
    try:
        if df1.empty or df2.empty:
            return 0.0
        # Convert to sets of tuples for overlap check
        set1 = set(map(tuple, df1.values.tolist()))
        set2 = set(map(tuple, df2.values.tolist()))
        overlap = len(set1 & set2)
        return overlap / max(len(set2), 1)
    except Exception:
        return 0.0
