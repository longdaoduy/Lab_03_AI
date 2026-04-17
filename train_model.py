"""
Train AI model on League of Legends high_diamond_ranked_10min dataset.
Target: blueWins (1 = blue team wins, 0 = red team wins)
"""

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("=" * 60)
print("1. LOAD DATA")
print("=" * 60)

df = pd.read_csv("high_diamond_ranked_10min.csv")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nTarget distribution:\n{df['blueWins'].value_counts()}")
print(f"\nNull values: {df.isnull().sum().sum()}")

# ── 2. Preprocess ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. PREPROCESS")
print("=" * 60)

# Drop non-feature columns
DROP_COLS = ["gameId"]
X = df.drop(columns=DROP_COLS + ["blueWins"])
y = df["blueWins"]

print(f"Features ({len(X.columns)}): {list(X.columns)}")

# ── 3. Split ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. TRAIN / TEST SPLIT  (80 / 20)")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

# Scale features (needed for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ── 4. Train models ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. TRAIN MODELS")
print("=" * 60)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
}

results = {}
for name, model in models.items():
    # Logistic Regression benefits from scaled data
    X_tr = X_train_scaled if name == "Logistic Regression" else X_train
    X_te = X_test_scaled if name == "Logistic Regression" else X_test

    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    acc = accuracy_score(y_test, y_pred)

    # 5-fold CV on training set
    cv_scores = cross_val_score(
        model, X_tr, y_train, cv=5, scoring="accuracy", n_jobs=-1
    )

    results[name] = {
        "model": model,
        "y_pred": y_pred,
        "accuracy": acc,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
    }
    print(
        f"  {name:<25}  Test Acc: {acc:.4f}  "
        f"CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
    )

# ── 5. Evaluate best model ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. EVALUATION")
print("=" * 60)

best_name = max(results, key=lambda n: results[n]["accuracy"])
best = results[best_name]
print(f"\nBest model: {best_name}  (Accuracy: {best['accuracy']:.4f})\n")
print(classification_report(y_test, best["y_pred"], target_names=["Red Wins", "Blue Wins"]))

# Confusion matrix
cm = confusion_matrix(y_test, best["y_pred"])
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Red Wins", "Blue Wins"])
disp.plot(ax=axes[0], colorbar=False)
axes[0].set_title(f"Confusion Matrix – {best_name}")

# ── 6. Feature importance ─────────────────────────────────────────────────────
model_obj = best["model"]
if hasattr(model_obj, "feature_importances_"):
    importances = pd.Series(model_obj.feature_importances_, index=X.columns)
    top20 = importances.sort_values(ascending=False).head(20)
    top20.sort_values().plot(kind="barh", ax=axes[1], color="steelblue")
    axes[1].set_title(f"Top-20 Feature Importances – {best_name}")
    axes[1].set_xlabel("Importance")
elif hasattr(model_obj, "coef_"):
    coefs = pd.Series(model_obj.coef_[0], index=X.columns)
    top20 = coefs.abs().sort_values(ascending=False).head(20)
    coefs[top20.index].sort_values().plot(kind="barh", ax=axes[1], color="steelblue")
    axes[1].set_title(f"Top-20 Coefficients – {best_name}")
    axes[1].set_xlabel("Coefficient")

plt.tight_layout()
plt.savefig("model_evaluation.png", dpi=150)
print("\nPlot saved → model_evaluation.png")

# ── 7. Save best model ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. SAVE MODEL")
print("=" * 60)

save_data = {
    "model_name": best_name,
    "model": model_obj,
    "scaler": scaler,
    "feature_columns": list(X.columns),
}

with open("model.pkl", "wb") as f:
    pickle.dump(save_data, f)

print(f"Model saved → model.pkl")
print("\nDone ✓")
