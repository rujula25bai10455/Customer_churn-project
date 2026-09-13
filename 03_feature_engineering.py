import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# 1. Load Extracted Data from Step 1
df = pd.read_csv("data/analytical_churn_dataset.csv")

# -------------------------------------------------------------
# 2. Domain-Driven Feature Engineering
# -------------------------------------------------------------
# Prevent division by zero with np.maximum()
df["support_ticket_rate"] = df["support_tickets"] / np.maximum(df["tenure_months"], 1)
df["clv_proxy"] = df["tenure_months"] * df["monthly_charges"]
df["usage_to_cost_ratio"] = df["monthly_usage"] / np.maximum(df["monthly_charges"], 1)
df["engagement_intensity"] = df["monthly_usage"] / np.maximum(df["login_frequency"], 1)

# Tenure Cohort Bins
df["tenure_cohort"] = pd.cut(
    df["tenure_months"],
    bins=[-1, 12, 24, 48, 120],
    labels=["0-12m", "13-24m", "25-48m", "49m+"],
)

print("Engineered features created:")
print(["support_ticket_rate", "clv_proxy", "usage_to_cost_ratio", "engagement_intensity", "tenure_cohort"])

# -------------------------------------------------------------
# 3. Stratified Train-Test Split (Strictly Before Scaling)
# -------------------------------------------------------------
X = df.drop(columns=["customer_id", "churn"])
y = df["churn"]

# Stratify ensures train and test have the exact same churn ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")
print(f"Train Churn Rate: {y_train.mean():.2%}")
print(f"Test Churn Rate:  {y_test.mean():.2%}")

# -------------------------------------------------------------
# 4. Leakage-Free Preprocessing Pipeline
# -------------------------------------------------------------
# Identify numerical and categorical columns
num_features = [
    "age",
    "tenure_months",
    "monthly_charges",
    "monthly_usage",
    "login_frequency",
    "support_tickets",
    "avg_session_duration",
    "support_ticket_rate",
    "clv_proxy",
    "usage_to_cost_ratio",
    "engagement_intensity",
]

cat_features = [
    "gender",
    "city",
    "plan_type",
    "contract_type",
    "payment_method",
    "tenure_cohort",
]

# Numerical Pipeline: Median Imputation + Robust Scaling (handles outliers)
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", RobustScaler())
])

# Categorical Pipeline: Most Frequent Imputation + One-Hot Encoding
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# Full Column Transformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipeline, num_features),
        ("cat", cat_pipeline, cat_features),
    ],
    remainder="drop"
)

# Fit ONLY on training data, then transform both (prevents data leakage)
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Extract encoded feature names for explainability
cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
encoded_cat_names = list(cat_encoder.get_feature_names_out(cat_features))
all_feature_names = num_features + encoded_cat_names

# Convert back to DataFrame for inspection and modeling
X_train_df = pd.DataFrame(X_train_processed, columns=all_feature_names)
X_test_df = pd.DataFrame(X_test_processed, columns=all_feature_names)

# -------------------------------------------------------------
# 5. Persist Processed Sets and Artifacts
# -------------------------------------------------------------
X_train_df.to_csv("data/X_train.csv", index=False)
X_test_df.to_csv("data/X_test.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

# Save preprocessor pipeline for inference
joblib.dump(preprocessor, "src/preprocessor.joblib")

print("\nStep 3 Complete!")
print("Saved artifacts: data/X_train.csv, data/X_test.csv, data/y_train.csv, data/y_test.csv")
print("Saved fitted pipeline: src/preprocessor.joblib")
print(f"Final feature count for modeling: {len(all_feature_names)}")