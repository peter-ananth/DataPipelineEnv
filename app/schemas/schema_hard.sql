-- Schema for Task 3: Query Reverse Engineering (Hard)
-- Tables: sales, regions, salespersons

CREATE TABLE IF NOT EXISTS regions (
    region_id INTEGER PRIMARY KEY,
    region_name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS salespersons (
    salesperson_id INTEGER PRIMARY KEY,
    salesperson_name TEXT NOT NULL,
    region_id INTEGER,
    FOREIGN KEY (region_id) REFERENCES regions(region_id)
);

CREATE TABLE IF NOT EXISTS sales (
    sale_id INTEGER PRIMARY KEY,
    region TEXT NOT NULL,
    salesperson TEXT NOT NULL,
    product_category TEXT NOT NULL,
    revenue REAL NOT NULL,
    units_sold INTEGER NOT NULL,
    sale_month TEXT NOT NULL,
    sale_year INTEGER NOT NULL
);
