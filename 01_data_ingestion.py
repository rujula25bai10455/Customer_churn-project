import sqlite3
import pandas as pd

# Connect to database
conn = sqlite3.connect("churn_warehouse.db")
cursor = conn.cursor()

# 1. Clear any existing tables so there are no duplicate key conflicts
cursor.executescript("""
DROP TABLE IF EXISTS churn_records;
DROP TABLE IF EXISTS usage;
DROP TABLE IF EXISTS subscription;
DROP TABLE IF EXISTS customers;
""")

# 2. Create Relational Tables
cursor.execute("""
CREATE TABLE customers (
    customer_id TEXT PRIMARY KEY,
    age INTEGER,
    gender TEXT,
    city TEXT,
    tenure_months INTEGER
);
""")

cursor.execute("""
CREATE TABLE subscription (
    customer_id TEXT PRIMARY KEY,
    plan_type TEXT,
    monthly_charges REAL,
    contract_type TEXT,
    payment_method TEXT
);
""")

cursor.execute("""
CREATE TABLE usage (
    customer_id TEXT PRIMARY KEY,
    monthly_usage REAL,
    login_frequency INTEGER,
    support_tickets INTEGER,
    avg_session_duration REAL
);
""")

cursor.execute("""
CREATE TABLE churn_records (
    customer_id TEXT PRIMARY KEY,
    churn INTEGER
);
""")
conn.commit()

# 3. Read and normalize raw data
raw_df = pd.read_csv("data/customer_data.csv").drop_duplicates(subset=["customer_id"])

# Insert data into tables
raw_df[["customer_id", "age", "gender", "city", "tenure_months"]].to_sql(
    "customers", conn, if_exists="append", index=False
)
raw_df[["customer_id", "plan_type", "monthly_charges", "contract_type", "payment_method"]].to_sql(
    "subscription", conn, if_exists="append", index=False
)
raw_df[["customer_id", "monthly_usage", "login_frequency", "support_tickets", "avg_session_duration"]].to_sql(
    "usage", conn, if_exists="append", index=False
)
raw_df[["customer_id", "churn"]].to_sql(
    "churn_records", conn, if_exists="append", index=False
)

print("Relational tables populated successfully.")

# 4. Extract Analytical ML Dataset using SQL CTE
query = """
WITH valid_customers AS (
    SELECT 
        customer_id,
        COALESCE(age, 0) AS age,
        COALESCE(gender, 'Unknown') AS gender,
        city,
        tenure_months
    FROM customers
    WHERE customer_id IS NOT NULL
),
valid_usage AS (
    SELECT 
        customer_id,
        monthly_usage,
        login_frequency,
        COALESCE(support_tickets, 0) AS support_tickets,
        avg_session_duration
    FROM usage
)
SELECT 
    c.customer_id,
    c.age,
    c.gender,
    c.city,
    c.tenure_months,
    s.plan_type,
    s.monthly_charges,
    s.contract_type,
    s.payment_method,
    u.monthly_usage,
    u.login_frequency,
    u.support_tickets,
    u.avg_session_duration,
    cr.churn
FROM valid_customers c
INNER JOIN subscription s ON c.customer_id = s.customer_id
INNER JOIN valid_usage u ON c.customer_id = u.customer_id
INNER JOIN churn_records cr ON c.customer_id = cr.customer_id;
"""

df_analytical = pd.read_sql_query(query, conn)
conn.close()

# Save final dataset
df_analytical.to_csv("data/analytical_churn_dataset.csv", index=False)
print("Step 1 Complete! Output saved to data/analytical_churn_dataset.csv")
print(f"Shape: {df_analytical.shape}")