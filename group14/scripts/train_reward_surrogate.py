import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor


def load_reward_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df_reward = df[df["use_for_reward"] == True].copy()
    # Add domain feature: minibatches per update
    df_reward["minibatches"] = df_reward["buffer_size"] / df_reward["batch_size"]
    return df_reward


def evaluate_regression(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    models = {
        "Dummy Mean Baseline": DummyRegressor(strategy="mean"),
        "Ridge Regression": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for name, model in models.items():
        scores = cross_validate(
            model, X, y, cv=cv, scoring=["r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"]
        )
        rows.append({
            "Model": name,
            "R2_mean": round(scores["test_r2"].mean(), 3),
            "R2_std": round(scores["test_r2"].std(), 3),
            "RMSE_mean": round(-scores["test_neg_root_mean_squared_error"].mean(), 3),
            "MAE_mean": round(-scores["test_neg_mean_absolute_error"].mean(), 3),
        })
    return pd.DataFrame(rows)


def evaluate_classification(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    y_int = y.astype(int)
    models = {
        "Dummy Majority Baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(random_state=42)),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for name, model in models.items():
        scores = cross_validate(model, X, y_int, cv=cv, scoring=["accuracy", "f1", "roc_auc"])
        rows.append({
            "Model": name,
            "Accuracy": round(scores["test_accuracy"].mean(), 3),
            "F1": round(scores["test_f1"].mean(), 3),
            "ROC_AUC": round(scores["test_roc_auc"].mean(), 3),
        })
    return pd.DataFrame(rows)


def save_feature_importance(X: pd.DataFrame, y: pd.Series, out_dir: str) -> None:
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
    plt.figure(figsize=(7, 4))
    importances.plot(kind="barh", color="#2b5c8f")
    plt.title("Feature Importance for Predicting Final Reward (n=79)")
    plt.xlabel("Relative Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "reward_feature_importance.png"), dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train Reward & Convergence Surrogate Models")
    parser.add_argument("--data", default="group14/data/master/runs_master_v3.csv", help="Path to master CSV")
    parser.add_argument("--out", default="group14/results/surrogate_reward", help="Directory for output metrics & plots")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df_reward = load_reward_data(args.data)
    print(f"Loaded {len(df_reward)} valid reward runs from {args.data}")

    feature_cols = ["learning_rate", "batch_size", "buffer_size", "minibatches"]
    X = df_reward[feature_cols]

    # 1. Final Reward Regression
    reg_results = evaluate_regression(X, df_reward["final_reward"])
    print("\n--- Final Reward Regression (5-Fold CV) ---")
    print(reg_results.to_string(index=False))
    reg_results.to_csv(os.path.join(args.out, "reward_regression_metrics.csv"), index=False)

    # 2. Convergence Classification
    if df_reward["converged"].nunique() > 1:
        clf_results = evaluate_classification(X, df_reward["converged"])
        print("\n--- Convergence Classification (5-Fold CV) ---")
        print(clf_results.to_string(index=False))
        clf_results.to_csv(os.path.join(args.out, "convergence_classification_metrics.csv"), index=False)

    # 3. Feature Importance Plot
    save_feature_importance(X, df_reward["final_reward"], args.out)
    print(f"\nSaved metrics and plot to {args.out}/")


if __name__ == "__main__":
    main()
