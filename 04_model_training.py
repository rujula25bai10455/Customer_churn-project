import pandas as pd
import numpy as np
import joblib
import os

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# 1. Load Processed Feature Matrices
X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv").values.ravel()
y_test = pd.read_csv("data/y_test.csv").values.ravel()

# Compute scale_pos_weight for XGBoost to handle class imbalance
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
imbalance_ratio = neg_count / max(pos_count, 1)

# -------------------------------------------------------------
# 2. Define Model Suite with Cost-Sensitive Weighting
# -------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced", max_depth=5, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        class_weight="balanced", n_estimators=150, max_depth=8, random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42
    ),
    "XGBoost": XGBClassifier(
        scale_pos_weight=imbalance_ratio,
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        eval_metric="logloss",
        random_state=42,
    ),
}

# -------------------------------------------------------------
# 3. Stratified 5-Fold Cross-Validation
# -------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}

print("=" * 70)
print("BENCHMARKING MODELS ACROSS 5-FOLD STRATIFIED CV")
print("=" * 70)

cv_results = []
for name, model in models.items():
    scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
    cv_results.append({
        "Model": name,
        "CV ROC-AUC": np.mean(scores["test_roc_auc"]),
        "CV Recall": np.mean(scores["test_recall"]),
        "CV F1": np.mean(scores["test_f1"]),
        "CV Precision": np.mean(scores["test_precision"]),
        "CV Accuracy": np.mean(scores["test_accuracy"]),
    })

results_df = pd.DataFrame(cv_results).sort_values(by="CV ROC-AUC", ascending=False)
print(results_df.to_string(index=False))

# -------------------------------------------------------------
# 4. Hyperparameter Tuning on the Top Performer (Random Forest)
# -------------------------------------------------------------
print("\n" + "=" * 70)
print("HYPERPARAMETER OPTIMIZATION (RANDOM FOREST)")
print("=" * 70)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 8, 12],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}

rf_base = RandomForestClassifier(class_weight="balanced", random_state=42)
grid_search = GridSearchCV(
    estimator=rf_base,
    param_grid=param_grid,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=0,
)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"Optimal Parameters: {grid_search.best_params_}")
print(f"Optimized Train ROC-AUC: {grid_search.best_score_:.4f}")

# -------------------------------------------------------------
# 5. Out-of-Sample Test Set Evaluation
# -------------------------------------------------------------
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print("\n" + "=" * 70)
print("OUT-OF-SAMPLE TEST EVALUATION (FINAL BENCHMARK)")
print("=" * 70)
print(f"Test Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Test Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Test Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"Test F1-Score:  {f1_score(y_test, y_pred):.4f}")
print(f"Test ROC-AUC:   {roc_auc_score(y_test, y_proba):.4f}")

# -------------------------------------------------------------
# 6. Save Model Artifact and Predictions
# -------------------------------------------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/best_churn_model.joblib")

# Save test predictions for evaluation and explainability stages
test_predictions = pd.DataFrame({
    "y_true": y_test,
    "y_pred": y_pred,
    "churn_probability": y_proba,
})
test_predictions.to_csv("data/test_predictions.csv", index=False)

print("\nStep 4 Complete!")
print("Saved best model to: models/best_churn_model.joblib")
print("Saved test predictions to: data/test_predictions.csv")