"""
Diabetes Prediction Using Machine Learning
==========================================
Authors: Tanya Batra, Vanshika Kohli
Department of Computer Science & Engineering
Chitkara University, Punjab, India

This script trains and evaluates five supervised classifiers:
  - Logistic Regression (LR)
  - Decision Tree (DT)
  - Random Forest (RF)
  - Naive Bayes (NB)
  - K-Nearest Neighbours (KNN)
on a 100,001-record diabetes dataset from Kaggle.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)

# ─────────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────────
def load_data(filepath: str) -> pd.DataFrame:
    """Load the diabetes dataset from a CSV file."""
    df = pd.read_csv(filepath)
    print(f"[INFO] Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"[INFO] Missing values:\n{df.isnull().sum()}\n")
    return df


# ─────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────
def exploratory_analysis(df: pd.DataFrame) -> None:
    """Generate EDA plots: class distribution, correlation heatmap, and feature distributions."""
    print("[INFO] Running exploratory data analysis...")

    # Class distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    class_counts = df["diabetes"].value_counts()
    axes[0].bar(["Non-Diabetic (0)", "Diabetic (1)"], class_counts.values,
                color=["steelblue", "tomato"], edgecolor="black")
    axes[0].set_title("Class Distribution")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(class_counts.values):
        axes[0].text(i, v + 200, f"{v:,}", ha="center", fontweight="bold")

    # Pearson correlation heatmap
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                ax=axes[1], linewidths=0.5)
    axes[1].set_title("Pearson Correlation Heatmap")

    plt.tight_layout()
    plt.savefig("eda_overview.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[INFO] Saved: eda_overview.png\n")

    # Feature correlation with target
    target_corr = numeric_df.corr()["diabetes"].drop("diabetes").sort_values(ascending=False)
    print("[INFO] Feature correlation with diabetes label:")
    print(target_corr.to_string(), "\n")


# ─────────────────────────────────────────────
# 3. PREPROCESSING
# ─────────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    """
    - One-hot encode or binary-encode categorical columns if present.
    - Apply StandardScaler to continuous features.
    - Stratified 70:30 train-test split.
    Returns: X_train, X_test, y_train, y_test, scaler
    """
    # Encode gender if it is a string
    if df["gender"].dtype == object:
        df["gender"] = df["gender"].map({"Male": 1, "Female": 0, "Other": 0})

    # Encode smoking_history if it is a string
    if "smoking_history" in df.columns and df["smoking_history"].dtype == object:
        df["smoking_history"] = df["smoking_history"].apply(
            lambda x: 1 if x in ["current", "ever", "formerly"] else 0
        )

    continuous_features = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
    X = df.drop("diabetes", axis=1)
    y = df["diabetes"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train[continuous_features] = scaler.fit_transform(X_train[continuous_features])
    X_test[continuous_features] = scaler.transform(X_test[continuous_features])

    print(f"[INFO] Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
    print(f"[INFO] Class ratio in train — "
          f"Non-diabetic: {(y_train==0).sum()} | Diabetic: {(y_train==1).sum()}\n")

    return X_train, X_test, y_train, y_test, scaler


# ─────────────────────────────────────────────
# 4. MODEL TRAINING & EVALUATION
# ─────────────────────────────────────────────
def evaluate_model(name: str, model, X_train, X_test, y_train, y_test) -> dict:
    """Train a model, print results, and return a summary dict."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # AUC-ROC (use predict_proba if available, else decision_function)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)
    auc = roc_auc_score(y_test, y_prob)

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    print(f"{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Accuracy : {acc*100:.2f}%")
    print(f"  AUC-ROC  : {auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred)}")
    print(f"  Confusion Matrix:\n{cm}\n")

    return {
        "name": name,
        "model": model,
        "y_prob": y_prob,
        "accuracy": acc,
        "auc": auc,
        "precision_macro": report["macro avg"]["precision"],
        "recall_macro": report["macro avg"]["recall"],
        "f1_macro": report["macro avg"]["f1-score"],
        "cm": cm,
    }


def run_all_models(X_train, X_test, y_train, y_test) -> list:
    """Instantiate and evaluate all five classifiers."""
    models = [
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
        ("Decision Tree",        DecisionTreeClassifier(random_state=42)),
        ("Random Forest",        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ("Naive Bayes",          GaussianNB()),
        ("KNN (k=5)",            KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
    ]
    results = []
    for name, model in models:
        results.append(evaluate_model(name, model, X_train, X_test, y_train, y_test))
    return results


# ─────────────────────────────────────────────
# 5. VISUALISATION
# ─────────────────────────────────────────────
def plot_comparison(results: list) -> None:
    """Bar chart comparing accuracy, AUC, and macro F1 across models."""
    names   = [r["name"] for r in results]
    acc     = [r["accuracy"] * 100 for r in results]
    auc     = [r["auc"] for r in results]
    f1      = [r["f1_macro"] for r in results]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(13, 6))
    bars1 = ax.bar(x - width, acc,  width, label="Accuracy (%)",   color="steelblue",  edgecolor="black")
    bars2 = ax.bar(x,         [a*100 for a in auc], width, label="AUC-ROC (%)", color="darkorange", edgecolor="black")
    bars3 = ax.bar(x + width, [f*100 for f in f1],  width, label="Macro F1 (%)", color="mediumseagreen", edgecolor="black")

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(80, 102)
    ax.set_ylabel("Score (%)")
    ax.set_title("Model Comparison — Accuracy, AUC-ROC, Macro F1")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[INFO] Saved: model_comparison.png")


def plot_roc_curves(results: list, y_test) -> None:
    """Overlay ROC curves for all models."""
    plt.figure(figsize=(9, 7))
    for r in results:
        fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
        plt.plot(fpr, tpr, lw=2, label=f"{r['name']} (AUC={r['auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — All Models")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[INFO] Saved: roc_curves.png")


def plot_confusion_matrices(results: list) -> None:
    """Grid of confusion matrices, one per model."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    for ax, r in zip(axes, results):
        sns.heatmap(r["cm"], annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["No DM", "DM"], yticklabels=["No DM", "DM"])
        ax.set_title(r["name"], fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.suptitle("Confusion Matrices", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig("confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[INFO] Saved: confusion_matrices.png")


def plot_feature_importance(rf_result: dict, feature_names: list) -> None:
    """Bar chart of Random Forest feature importances."""
    importances = rf_result["model"].feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in indices]
    sorted_imp   = importances[indices]

    plt.figure(figsize=(9, 5))
    plt.bar(sorted_names, sorted_imp, color="mediumseagreen", edgecolor="black")
    plt.title("Random Forest — Feature Importances")
    plt.ylabel("Importance")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[INFO] Saved: feature_importance.png")


# ─────────────────────────────────────────────
# 6. SUMMARY TABLE
# ─────────────────────────────────────────────
def print_summary_table(results: list) -> None:
    rows = []
    for r in results:
        rows.append({
            "Algorithm":        r["name"],
            "Accuracy (%)":     f"{r['accuracy']*100:.2f}",
            "Precision (Macro)":f"{r['precision_macro']:.2f}",
            "Recall (Macro)":   f"{r['recall_macro']:.2f}",
            "F1 (Macro)":       f"{r['f1_macro']:.2f}",
            "AUC-ROC":          f"{r['auc']:.4f}",
        })
    summary = pd.DataFrame(rows)
    print("\n" + "="*70)
    print("FINAL PERFORMANCE SUMMARY")
    print("="*70)
    print(summary.to_string(index=False))
    print("="*70 + "\n")
    summary.to_csv("results_summary.csv", index=False)
    print("[INFO] Saved: results_summary.csv")


# ─────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    # Accept dataset path as CLI argument or use default
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "diabetes_prediction_dataset.csv"

    print("\n🔬 Diabetes Prediction — ML Classifier Comparison")
    print("="*55)

    # Load
    df = load_data(dataset_path)

    # EDA
    exploratory_analysis(df)

    # Preprocess
    X_train, X_test, y_train, y_test, scaler = preprocess(df)
    feature_names = X_train.columns.tolist()

    # Train & evaluate all models
    results = run_all_models(X_train, X_test, y_train, y_test)

    # Plots
    plot_comparison(results)
    plot_roc_curves(results, y_test)
    plot_confusion_matrices(results)

    # Random Forest feature importance
    rf_result = next(r for r in results if "Random Forest" in r["name"])
    plot_feature_importance(rf_result, feature_names)

    # Summary
    print_summary_table(results)
    print("✅ All outputs saved. Run complete.\n")
