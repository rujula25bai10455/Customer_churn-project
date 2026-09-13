import numpy as np
import pandas as pd

np.random.seed(42)
n = 1000

data = {
    "customer_id": [f"CUST_{i:04d}" for i in range(1, n + 1)],
    "age": np.random.randint(18, 70, size=n),
    "gender": np.random.choice(["Male", "Female"], size=n),
    "city": np.random.choice(
        ["New York", "Los Angeles", "Chicago", "Houston", "Miami"], size=n
    ),
    "tenure_months": np.random.randint(1, 72, size=n),
    "plan_type": np.random.choice(["Basic", "Standard", "Premium"], size=n),
    "monthly_charges": np.round(np.random.uniform(20.0, 120.0, size=n), 2),
    "contract_type": np.random.choice(
        ["Month-to-Month", "One Year", "Two Year"], size=n, p=[0.5, 0.3, 0.2]
    ),
    "payment_method": np.random.choice(
        ["Credit Card", "Bank Transfer", "Electronic Check", "Mailed Check"],
        size=n,
    ),
    "monthly_usage": np.round(np.random.uniform(50.0, 500.0, size=n), 1),
    "login_frequency": np.random.randint(1, 30, size=n),
    "support_tickets": np.random.poisson(lam=1.5, size=n),
    "avg_session_duration": np.round(
        np.random.uniform(5.0, 60.0, size=n), 1
    ),
    "churn": np.random.choice([0, 1], size=n, p=[0.73, 0.27]),
}

df = pd.DataFrame(data)
df.to_csv("data/customer_data.csv", index=False)
print("Successfully generated data/customer_data.csv")