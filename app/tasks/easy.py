"""
Task 1 — Easy: CSV Data Cleaning
Agent receives a dirty orders CSV, must clean it and return the fixed DataFrame.
Now featuring deeply randomized cleaning variants based on the environment seed!
"""

from __future__ import annotations

import io
import random
import textwrap
from pathlib import Path
from typing import Any

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"


_CLEANING_VARIANTS = [
    {
        "id": "standard",
        "description": "1. **Duplicate rows**: Remove exact duplicate records.\n2. **Missing values**: Drop rows where `customer_name` is null/empty.\n3. **Type errors**: Convert `price` to float and `quantity` to integer.\n4. **Inconsistent casing**: Normalize `country` to Title Case.",
        "operations": ["drop_dup", "drop_null_name", "type_fix", "title_country"],
    },
    {
        "id": "uppercase_countries",
        "description": "1. **Duplicate rows**: Remove exact duplicate records.\n2. **Missing values**: Fill null/empty `customer_name` values with the exact string 'GUEST'.\n3. **Type errors**: Convert `price` to float and `quantity` to integer.\n4. **Inconsistent casing**: Normalize `country` to strictly UPPERCASE.",
        "operations": ["drop_dup", "fill_null_guest", "type_fix", "upper_country"],
    },
    {
        "id": "lowercase_filter_price",
        "description": "1. **Duplicate rows**: Remove exact duplicate records.\n2. **Pricing Filter**: Drop all rows where `price` is less than 50.0.\n3. **Type errors**: Convert `price` to float and `quantity` to integer.\n4. **Inconsistent casing**: Normalize `country` strictly to lowercase.",
        "operations": ["drop_dup", "filter_price_50", "type_fix", "lower_country"],
    },
    {
        "id": "fill_quantity_zero",
        "description": "1. **Duplicate rows**: Remove exact duplicate records.\n2. **Missing values**: Drop rows where `customer_name` is null/empty.\n3. **Type errors**: Convert `price` to float. Fill missing/invalid `quantity` values with 0 (zero) and cast to integer.\n4. **Inconsistent casing**: Normalize `country` to Title Case.",
        "operations": ["drop_dup", "drop_null_name", "type_fix_fill_0", "title_country"],
    },
    {
        "id": "upper_drop_quantity",
        "description": "1. **Duplicate rows**: Remove exact duplicate records.\n2. **Quantity Filter**: Drop any rows where `quantity` is missing or invalid.\n3. **Type errors**: Convert `price` to float.\n4. **Inconsistent casing**: Normalize `country` to strictly UPPERCASE.",
        "operations": ["drop_dup", "drop_null_quantity", "type_fix", "upper_country"],
    }
]


def get_variant(seed: int | None = None) -> dict:
    """Return a deterministically random cleaning specification."""
    rng = random.Random(seed)
    return rng.choice(_CLEANING_VARIANTS)


# ─────────────────── Reference (ground-truth) solution ───────────────────────

def generate_dirty_csv_string(seed: int | None = None) -> str:
    """Procedurally generate messy CSV transactions directly via random seed."""
    rng = random.Random(seed)
    
    names = ["Alice Johnson", "Bob Smith", "Carol White", "David Brown", "Eve Davis", "Frank Miller", "Grace Lee", "Henry Wang", "Ivy Chen", "Jack Doe"]
    countries = ["United States", "uk", "CANADA", "germany", "France", "AUSTRALIA", "japan", "China", "Brazil", "india"]
    products = ["Laptop", "Mouse", "Monitor", "Keyboard", "Headphones", "Webcam", "SSD", "Charger"]
    
    rows = ["order_id,customer_name,country,product,price,quantity,order_date"]
    
    for i in range(1, 35):
        order_id = 1000 + i
        name = rng.choice(names) if rng.random() > 0.15 else "" # 15% null name
        country = rng.choice(countries)
        
        if rng.random() > 0.5:
            country = country.lower() if rng.random() > 0.5 else country.upper()
            
        product = rng.choice(products)
        base_price = round(rng.uniform(10.0, 1500.0), 2)
        
        if rng.random() > 0.4:
            price_str = f'"{base_price:,.2f}"'
        else:
            price_str = str(base_price)
            
        r_q = rng.random()
        if r_q > 0.9:
            qty = ""
        elif r_q > 0.8:
            qty = "invalid"
        else:
            qty = str(rng.randint(1, 5))
            
        date = f"2024-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}"
        
        row_str = f'{order_id},{" " + name if name else name},{country},{product},{price_str},{qty},{date}'
        rows.append(row_str)
        
        if rng.random() > 0.9:
            rows.append(row_str) # Duplicate row injection
            
    return "\n".join(rows)

def get_reference_df(seed: int | None = None) -> pd.DataFrame:
    """Load and return the fully-cleaned reference DataFrame based on the seed."""
    csv_string = generate_dirty_csv_string(seed)
    df = pd.read_csv(io.StringIO(csv_string), dtype={"price": str, "quantity": str})
    
    variant = get_variant(seed)
    ops = variant["operations"]

    if "drop_dup" in ops:
        df = df.drop_duplicates()

    if "drop_null_name" in ops:
        df = df.dropna(subset=["customer_name"])
    if "fill_null_guest" in ops:
        
        df["customer_name"] = df["customer_name"].fillna("GUEST")

    # Clean price string for numeric conversion
    df["price"] = pd.to_numeric(df["price"].str.replace(",", "").str.strip(), errors="coerce")
    
    if "filter_price_50" in ops:
        df = df[df["price"] >= 50.0]

    if "type_fix" in ops or "drop_null_quantity" in ops:
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").astype("Int64")
        
    if "type_fix_fill_0" in ops:
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0).astype("Int64")
        
    if "drop_null_quantity" in ops:
        df = df.dropna(subset=["quantity"])

    if "title_country" in ops:
        df["country"] = df["country"].str.strip().str.title()
    if "upper_country" in ops:
        df["country"] = df["country"].str.strip().str.upper()
    if "lower_country" in ops:
        df["country"] = df["country"].str.strip().str.lower()

    df = df.reset_index(drop=True)
    return df


def get_dirty_csv_string() -> str:
    """Return the raw dirty CSV content as a string."""
    return DIRTY_CSV.read_text()


# ─────────────────── Task metadata ───────────────────────────────────────────

TASK_ID = "csv_cleaning"
TASK_NAME = "CSV Data Cleaning"
DIFFICULTY = "easy"
MAX_ATTEMPTS = 5


def get_initial_observation(seed: int | None = None) -> dict[str, Any]:
    """Return the dynamic initial observation for Task 1."""
    variant = get_variant(seed)
    csv_string = generate_dirty_csv_string(seed)
    dirty_preview = pd.read_csv(io.StringIO(csv_string)).head(10).to_csv(index=False)
    
    description = textwrap.dedent(f"""
        You are given a dirty e-commerce orders CSV. Your job is to clean it and return
        the corrected CSV as a string.

        Issues to fix:
        {variant['description']}

        **Action format:**
        {{
            "type": "clean_csv",
            "payload": "<your cleaned CSV as a string>"
        }}

        Return a valid CSV with the same columns as the original (order_id, customer_name,
        country, product, price, quantity, order_date).
    """).strip()
    
    return {
        "task_id": TASK_ID,
        "task_description": description,
        "data_preview": dirty_preview,
        "schema": "order_id (int), customer_name (str), country (str), product (str), price (float), quantity (int), order_date (str)",
        "error_message": None,
        "attempt": 1,
        "max_attempts": MAX_ATTEMPTS,
    }


def parse_submission(payload: str) -> pd.DataFrame:
    """Parse a CSV string submission into a DataFrame."""
    try:
        return pd.read_csv(io.StringIO(payload.strip()))
    except Exception as e:
        raise ValueError(f"Could not parse submitted CSV: {e}") from e
