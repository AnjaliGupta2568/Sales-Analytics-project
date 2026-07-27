"""
generate_data.py
-----------------
Generates a synthetic retail transactions dataset (100,000+ records) used
as the raw input for the Sales & Business Analytics Dashboard project.

In a real-world version of this project, this script would be replaced by
a connection to a POS / e-commerce database. Here we simulate realistic,
slightly messy data (missing values, duplicate rows, inconsistent casing,
occasional negative/zero quantities) so the cleaning pipeline in
`clean_data.py` has genuine work to do.

Run:
    python src/generate_data.py
Output:
    data/raw_sales_data.csv
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG_SEED = 42
N_RECORDS = 100_000
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2026, 6, 30)

REGIONS = ["North", "South", "East", "West", "Central"]

CATEGORIES = {
    "Electronics": ["Wireless Earbuds", "Smart Watch", "Bluetooth Speaker", "Power Bank", "Laptop Stand"],
    "Apparel": ["Cotton T-Shirt", "Denim Jacket", "Running Shoes", "Wool Sweater", "Formal Shirt"],
    "Home & Kitchen": ["Non-stick Pan", "Air Fryer", "LED Desk Lamp", "Storage Organizer", "Coffee Maker"],
    "Beauty": ["Face Serum", "Sunscreen SPF50", "Hair Dryer", "Lipstick Set", "Body Lotion"],
    "Sports": ["Yoga Mat", "Dumbbell Set", "Cricket Bat", "Football", "Resistance Bands"],
}

PAYMENT_METHODS = ["Credit Card", "Debit Card", "UPI", "Net Banking", "Cash on Delivery"]


def random_dates(n, start, end, rng):
    delta_days = (end - start).days
    offsets = rng.integers(0, delta_days, size=n)
    return [start + timedelta(days=int(o)) for o in offsets]


def generate(n_records: int = N_RECORDS, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    categories = list(CATEGORIES.keys())
    category_choices = rng.choice(categories, size=n_records)
    product_names = [
        rng.choice(CATEGORIES[cat]) for cat in category_choices
    ]

    df = pd.DataFrame({
        "order_id": np.arange(1, n_records + 1),
        "order_date": random_dates(n_records, START_DATE, END_DATE, rng),
        "customer_id": rng.integers(1000, 9000, size=n_records),
        "region": rng.choice(REGIONS, size=n_records),
        "product_category": category_choices,
        "product_name": product_names,
        "quantity": rng.integers(1, 6, size=n_records),
        "unit_price": np.round(rng.uniform(199, 4999, size=n_records), 2),
        "payment_method": rng.choice(PAYMENT_METHODS, size=n_records),
        "customer_rating": rng.integers(1, 6, size=n_records),
    })

    # Simulate a customer churn flag (1 = churned, i.e. no repeat purchase in 90 days)
    df["is_churned"] = rng.choice([0, 1], size=n_records, p=[0.78, 0.22])

    # --- Inject realistic messiness for the cleaning pipeline to handle ---

    # 1. Missing values in a few columns
    for col, frac in [("customer_rating", 0.03), ("payment_method", 0.02), ("unit_price", 0.01)]:
        missing_idx = rng.choice(df.index, size=int(frac * n_records), replace=False)
        df.loc[missing_idx, col] = np.nan

    # 2. Inconsistent text casing in region / category (common real-world issue)
    messy_idx = rng.choice(df.index, size=int(0.05 * n_records), replace=False)
    df.loc[messy_idx, "region"] = df.loc[messy_idx, "region"].str.upper()

    # 3. A handful of duplicate rows
    dup_rows = df.sample(n=200, random_state=seed)
    df = pd.concat([df, dup_rows], ignore_index=True)

    # 4. A few invalid quantities (data entry errors)
    bad_idx = rng.choice(df.index, size=50, replace=False)
    df.loc[bad_idx, "quantity"] = rng.choice([-1, 0], size=50)

    df["order_date"] = pd.to_datetime(df["order_date"]).dt.date
    return df


if __name__ == "__main__":
    data = generate()
    data.to_csv("data/raw_sales_data.csv", index=False)
    print(f"Generated {len(data):,} rows -> data/raw_sales_data.csv")
