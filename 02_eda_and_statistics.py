import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# 1. Load Extracted Data from Step 1
df = pd.read_csv("data/analytical_churn_dataset.csv")

# Create a folder to store generated plots
os.makedirs("reports/figures", exist_ok=True)

print("=" * 60)
print("EXPLORATORY DATA ANALYSIS & STATISTICAL HYPOTHESIS TESTING")
print("=" * 60)

# Check class balance
churn_rate = df["churn"].mean() * 100
print(f"\nOverall Churn Rate: {churn_rate:.2f}%")
print(f"Total Customers: {len(df)}")

# -------------------------------------------------------------
# 2. Statistical Hypothesis Testing: Numerical Drivers
# -------------------------------------------------------------
# We test whether churned vs non-churned distributions differ significantly
numerical_cols = [
    "monthly_charges",
    "tenure_months",
    "monthly_usage",
    "login_frequency",
    "support_tickets",
    "avg_session_duration",
]

print("\n--- 1. Numerical Variables (Mann-Whitney U Test) ---")
print(f"{'Feature':<25} | {'p-value':<12} | {'Significant (p < 0.05)'}")
print("-" * 60)

for col in numerical_cols:
    retained = df[df["churn"] == 0][col]
    churned = df[df["churn"] == 1][col]
    stat, p_val = stats.mannwhitneyu(retained, churned, alternative="two-sided")
    is_sig = "YES" if p_val < 0.05 else "NO"
    print(f"{col:<25} | {p_val:<12.5e} | {is_sig}")

# -------------------------------------------------------------
# 3. Statistical Hypothesis Testing: Categorical Drivers
# -------------------------------------------------------------
# Chi-Square Test of Independence for categorical features
categorical_cols = ["gender", "city", "plan_type", "contract_type", "payment_method"]

print("\n--- 2. Categorical Variables (Chi-Square Independence Test) ---")
print(f"{'Feature':<20} | {'Chi2 Stat':<10} | {'p-value':<12} | {'Significant'}")
print("-" * 60)

for col in categorical_cols:
    contingency_table = pd.crosstab(df[col], df["churn"])
    chi2, p_val, dof, _ = stats.chi2_contingency(contingency_table)
    is_sig = "YES" if p_val < 0.05 else "NO"
    print(f"{col:<20} | {chi2:<10.2f} | {p_val:<12.5e} | {is_sig}")

# -------------------------------------------------------------
# 4. Information Value (IV) for Feature Predictive Power
# -------------------------------------------------------------
def calculate_woe_iv(data, feature, target):
    lst = []
    for val in data[feature].unique():
        total_good = (data[target] == 0).sum()
        total_bad = (data[target] == 1).sum()
        good = ((data[feature] == val) & (data[target] == 0)).sum()
        bad = ((data[feature] == val) & (data[target] == 1)).sum()
        
        if good == 0 or bad == 0:
            continue
        
        dist_good = good / total_good
        dist_bad = bad / total_bad
        woe = np.log(dist_good / dist_bad)
        iv = (dist_good - dist_bad) * woe
        lst.append(iv)
    return sum(lst)

print("\n--- 3. Information Value (Predictive Strength) ---")
for cat in categorical_cols:
    iv_score = calculate_woe_iv(df, cat, "churn")
    strength = "Strong" if iv_score > 0.3 else "Medium" if iv_score > 0.1 else "Weak/None"
    print(f"{cat:<20} | IV: {iv_score:.4f} ({strength})")

# -------------------------------------------------------------
# 5. Visual EDA Generation
# -------------------------------------------------------------
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot A: Contract Type vs Churn Rate
contract_churn = df.groupby("contract_type")["churn"].mean().reset_index()
sns.barplot(data=contract_churn, x="contract_type", y="churn", ax=axes[0, 0], palette="Blues_r")
axes[0, 0].set_title("Churn Rate by Contract Type", fontsize=12, fontweight="bold")
axes[0, 0].set_ylabel("Churn Proportion")

# Plot B: Monthly Charges Distribution
sns.boxplot(data=df, x="churn", y="monthly_charges", ax=axes[0, 1], palette="Set2")
axes[0, 1].set_title("Monthly Charges vs Churn Status", fontsize=12, fontweight="bold")
axes[0, 1].set_xticklabels(["Retained", "Churned"])

# Plot C: Support Tickets Impact
sns.barplot(data=df, x="support_tickets", y="churn", ax=axes[1, 0], palette="Reds")
axes[1, 0].set_title("Support Tickets Count vs Churn Rate", fontsize=12, fontweight="bold")
axes[1, 0].set_ylabel("Churn Proportion")

# Plot D: Correlation Heatmap
corr = df[numerical_cols + ["churn"]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1, 1], cbar=False)
axes[1, 1].set_title("Correlation Matrix", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig("reports/figures/eda_summary.png", dpi=300)
print("\nGenerated summary visual: reports/figures/eda_summary.png")