import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import os
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    auc,
)

# 1. Load Processed Test Sets and Trained Model
X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv").values.ravel()
best_model = joblib.load("models/best_churn_model.joblib")

y_proba = best_model.predict_proba(X_test)[:, 1]

os.makedirs("reports/figures", exist_ok=True)

print("=" * 70)
print("FINANCIAL UTILITY & THRESHOLD OPTIMIZATION")
print("=" * 70)

# -------------------------------------------------------------
# 2. Business Cost-Benefit Matrix Simulation
# -------------------------------------------------------------
# Scenario assumptions:
# - CLV saved if churner is retained: $150
# - Retention campaign cost per outreach: $25
# - Lost customer cost (churned without intervention): $200
# - Contacting a non-churner: costs only the retention offer ($25)
RETAIN_BENEFIT = 150
RETAIN_COST = 25
LOST_CHURN_PENALTY = 200

thresholds = np.linspace(0.1, 0.9, 81)
net_profits = []

for thresh in thresholds:
    preds = (y_proba >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

    # Financial formula:
    # Profit = (TP * net benefit of retention) - (FP * cost of outreach) - (FN * penalty of lost customer)
    profit = (tp * (RETAIN_BENEFIT - RETAIN_COST)) - (fp * RETAIN_COST) - (fn * LOST_CHURN_PENALTY)
    net_profits.append(profit)

optimal_idx = np.argmax(net_profits)
optimal_threshold = thresholds[optimal_idx]
max_profit = net_profits[optimal_idx]

default_preds = (y_proba >= 0.5).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, default_preds).ravel()
default_profit = (tp * (RETAIN_BENEFIT - RETAIN_COST)) - (fp * RETAIN_COST) - (fn * LOST_CHURN_PENALTY)

print(f"Default Threshold (0.50) Expected Net Profit: ${default_profit:,.2f}")
print(f"Optimized Threshold ({optimal_threshold:.2f}) Expected Net Profit: ${max_profit:,.2f}")
print(f"Net Profit Gain from Tuning: +${(max_profit - default_profit):,.2f}")

# -------------------------------------------------------------
# 3. Comprehensive Visual Diagnostics
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Profit Curve Across Thresholds
axes[0].plot(thresholds, net_profits, color="green", lw=2, label="Net Business Profit")
axes[0].axvline(optimal_threshold, color="red", linestyle="--", label=f"Optimal: {optimal_threshold:.2f}")
axes[0].axvline(0.5, color="gray", linestyle=":", label="Default: 0.50")
axes[0].set_title("Financial Utility vs Decision Threshold", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Probability Decision Threshold")
axes[0].set_ylabel("Expected Net Profit ($)")
axes[0].legend()

# Plot 2: Optimized Confusion Matrix
optimized_preds = (y_proba >= optimal_threshold).astype(int)
cm = confusion_matrix(y_test, optimized_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1],
            xticklabels=["Retained", "Churned"], yticklabels=["Retained", "Churned"])
axes[1].set_title(f"Confusion Matrix (Thresh = {optimal_threshold:.2f})", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

# Plot 3: Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_proba)
pr_auc = auc(recall, precision)
axes[2].plot(recall, precision, color="purple", lw=2, label=f"PR-AUC = {pr_auc:.3f}")
axes[2].set_title("Precision-Recall Curve", fontsize=12, fontweight="bold")
axes[2].set_xlabel("Recall")
axes[2].set_ylabel("Precision")
axes[2].legend()

plt.tight_layout()
plt.savefig("reports/figures/model_evaluation_metrics.png", dpi=300)
print("\nEvaluation graphs saved to: reports/figures/model_evaluation_metrics.png")

# -------------------------------------------------------------
# 4. Model Explainability (SHAP TreeExplainer)
# -------------------------------------------------------------
print("\n" + "=" * 70)
print("COMPUTING GLOBAL SHAP EXPLANATIONS")
print("=" * 70)

explainer = shap.TreeExplainer(best_model)
# For binary classification, take the SHAP values corresponding to the positive class (churn = 1)
shap_values = explainer.shap_values(X_test)
if isinstance(shap_values, list):
    shap_matrix = shap_values[1]
elif len(shap_values.shape) == 3:
    shap_matrix = shap_values[:, :, 1]
else:
    shap_matrix = shap_values

# Save SHAP Summary Bar Plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_matrix, X_test, plot_type="bar", show=False)
plt.title("Top Feature Drivers Ranked by SHAP Importance", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("reports/figures/shap_feature_importance.png", dpi=300)
plt.close()

# Save SHAP Detailed Beeswarm Plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_matrix, X_test, show=False)
plt.title("SHAP Beeswarm: Impact on Churn Probability", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("reports/figures/shap_beeswarm.png", dpi=300)
plt.close()

print("SHAP plots saved:")
print(" - reports/figures/shap_feature_importance.png")
print(" - reports/figures/shap_beeswarm.png")

# Save optimal threshold for deployment
with open("models/optimal_threshold.txt", "w") as f:
    f.write(str(optimal_threshold))
print(f"Optimal threshold saved: {optimal_threshold:.4f}")