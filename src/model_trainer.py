"""Train an XGBoost model to predict post-earnings price direction.

Uses time-series–aware splitting (earlier data for training, later for testing)
and evaluates with accuracy, precision, recall, F1, and AUC.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_CSV = PROJECT_ROOT / "data" / "processed" / "features.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "sentiment_pos_mean",
    "sentiment_pos_std",
    "sentiment_neg_mean",
    "sentiment_neg_std",
    "sentiment_neu_mean",
    "sentiment_spread",
    "num_qa_pairs",
    "pct_positive",
    "pct_negative",
    "pre_earnings_vol",
    "pre_earnings_momentum",
    "overnight_gap",
]


def train_model(target_col="target_5d"):
    """Train XGBoost with time-series cross-validation and a final holdout eval."""
    print("Loading features...")
    df = pd.read_csv(FEATURES_CSV)
    df["earnings_date"] = pd.to_datetime(df["earnings_date"])
    df = df.sort_values("earnings_date").reset_index(drop=True)

    X = df[FEATURE_COLS].values
    y = df[target_col].values

    # --- Time-series cross-validation ---
    tscv = TimeSeriesSplit(n_splits=3)
    cv_scores = []
    print(f"\nTime-series CV (3 folds) for target={target_col}:")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        model = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
        )
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[val_idx])
        acc = accuracy_score(y[val_idx], preds)
        cv_scores.append(acc)
        print(f"  Fold {fold + 1}: accuracy={acc:.3f} (train={len(train_idx)}, val={len(val_idx)})")

    print(f"  Mean CV accuracy: {np.mean(cv_scores):.3f} +/- {np.std(cv_scores):.3f}")

    # --- Final train/test split (80/20 chronological) ---
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"\nFinal holdout: train={len(X_train)}, test={len(X_test)}")

    final_model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
    )
    final_model.fit(X_train, y_train)

    y_pred = final_model.predict(X_test)
    y_prob = final_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc = float("nan")

    print(f"\nHoldout results:")
    print(f"  Accuracy:  {acc:.3f}")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall:    {rec:.3f}")
    print(f"  F1 Score:  {f1:.3f}")
    print(f"  AUC-ROC:   {auc:.3f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Down', 'Up'], zero_division=0)}")

    # Feature importance
    importances = dict(zip(FEATURE_COLS, final_model.feature_importances_.tolist()))
    importances_sorted = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
    print("Feature importances:")
    for feat, imp in importances_sorted.items():
        bar = "#" * int(imp * 50)
        print(f"  {feat:<25} {imp:.4f}  {bar}")

    # Save model and results
    model_path = MODEL_DIR / f"xgb_{target_col}.json"
    final_model.save_model(str(model_path))
    print(f"\nModel saved to {model_path.relative_to(PROJECT_ROOT)}")

    results = {
        "target": target_col,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "cv_accuracy_mean": float(np.mean(cv_scores)),
        "cv_accuracy_std": float(np.std(cv_scores)),
        "holdout_accuracy": float(acc),
        "holdout_precision": float(prec),
        "holdout_recall": float(rec),
        "holdout_f1": float(f1),
        "holdout_auc": float(auc),
        "feature_importances": importances_sorted,
    }
    results_path = MODEL_DIR / f"results_{target_col}.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"Results saved to {results_path.relative_to(PROJECT_ROOT)}")

    # Also store predictions on the test set for backtesting
    test_df = df.iloc[split_idx:].copy()
    test_df["prediction"] = y_pred
    test_df["pred_prob"] = y_prob
    test_path = PROJECT_ROOT / "data" / "processed" / f"test_predictions_{target_col}.csv"
    test_df.to_csv(test_path, index=False)
    print(f"Test predictions saved to {test_path.name}")

    return final_model, results


if __name__ == "__main__":
    train_model("target_5d")
    print("\n" + "=" * 60)
    train_model("target_1d")
