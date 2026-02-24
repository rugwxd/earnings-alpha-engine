"""Train an XGBoost model to predict post-earnings price direction.

Uses EXPANDING-WINDOW WALK-FORWARD validation:
  - Sort all earnings events chronologically.
  - Start with a minimum training window (first N events).
  - Predict the next quarter's events out-of-sample.
  - Expand the training window by one quarter, retrain, repeat.

This ensures every prediction is truly out-of-sample and the model
is retrained as new data arrives — matching how it would work live.
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
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_CSV = PROJECT_ROOT / "data" / "processed" / "features.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "sentiment_pos_mean",
    "sentiment_neg_mean",
    "sentiment_spread",
    "pct_positive",
    "pct_negative",
    "pre_earnings_vol",
    "pre_earnings_momentum",
]

MIN_TRAIN_QUARTERS = 6  # need at least 6 quarters before first prediction


def _make_model():
    """Create an XGBoost classifier with conservative hyperparameters for small data."""
    return XGBClassifier(
        n_estimators=50,          # fewer trees to reduce overfitting
        max_depth=2,              # shallow trees for ~80 samples
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,       # regularization for small samples
        reg_alpha=0.1,            # L1 regularization
        reg_lambda=1.0,           # L2 regularization
        random_state=42,
        eval_metric="logloss",
    )


def walk_forward(target_col="target_5d"):
    """Expanding-window walk-forward validation by quarter.

    For each quarter Q (starting after MIN_TRAIN_QUARTERS):
      1. Train on all data from quarters < Q.
      2. Predict on all events in quarter Q.
      3. Record predictions.

    Returns a DataFrame with one row per earnings event that received
    an out-of-sample prediction, plus the trained model from the final fold.
    """
    print("Loading features...")
    df = pd.read_csv(FEATURES_CSV)
    df["earnings_date"] = pd.to_datetime(df["earnings_date"])
    df = df.sort_values("earnings_date").reset_index(drop=True)

    # Get chronologically ordered unique quarters
    quarters = df["q"].unique().tolist()
    # Quarters are like '2019-Q1' — sort properly
    quarters = sorted(quarters, key=lambda q: (int(q[:4]), int(q[-1])))

    print(f"Total quarters: {len(quarters)}")
    print(f"Min training quarters: {MIN_TRAIN_QUARTERS}")
    print(f"Walk-forward folds: {len(quarters) - MIN_TRAIN_QUARTERS}\n")

    X = df[FEATURE_COLS].values
    y = df[target_col].values

    all_predictions = []
    fold_results = []
    final_model = None

    for i in range(MIN_TRAIN_QUARTERS, len(quarters)):
        train_quarters = quarters[:i]
        test_quarter = quarters[i]

        train_mask = df["q"].isin(train_quarters)
        test_mask = df["q"] == test_quarter

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        if len(X_test) == 0:
            continue

        model = _make_model()
        model.fit(X_train, y_train)
        final_model = model

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, preds)

        fold_results.append({
            "fold": i - MIN_TRAIN_QUARTERS + 1,
            "test_quarter": test_quarter,
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
            "accuracy": float(acc),
        })
        print(f"  Fold {fold_results[-1]['fold']:2d}: "
              f"train Q<={train_quarters[-1]} ({len(X_train):3d}) → "
              f"test {test_quarter} ({len(X_test):2d}): "
              f"acc={acc:.3f}")

        # Store per-event predictions
        test_df = df[test_mask].copy()
        test_df["prediction"] = preds
        test_df["pred_prob"] = probs
        test_df["fold"] = fold_results[-1]["fold"]
        all_predictions.append(test_df)

    # Combine all out-of-sample predictions
    oos_df = pd.concat(all_predictions, ignore_index=True)

    # Overall OOS metrics
    y_true = oos_df[target_col].values
    y_pred = oos_df["prediction"].values
    y_prob = oos_df["pred_prob"].values

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    print(f"\n{'=' * 55}")
    print(f"  WALK-FORWARD OOS RESULTS — {target_col}")
    print(f"{'=' * 55}")
    print(f"  Total OOS predictions: {len(oos_df)}")
    print(f"  Folds:                 {len(fold_results)}")
    print(f"  Accuracy:              {acc:.3f}")
    print(f"  Precision:             {prec:.3f}")
    print(f"  Recall:                {rec:.3f}")
    print(f"  F1 Score:              {f1:.3f}")
    print(f"  AUC-ROC:               {auc:.3f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=['Down', 'Up'], zero_division=0)}")

    # Feature importance (from final model)
    importances = dict(zip(FEATURE_COLS, final_model.feature_importances_.tolist()))
    importances_sorted = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
    print("Feature importances (final fold):")
    for feat, imp in importances_sorted.items():
        bar = "#" * int(imp * 40)
        print(f"  {feat:<25} {imp:.4f}  {bar}")

    # Save model (final fold) and results
    model_path = MODEL_DIR / f"xgb_{target_col}.json"
    final_model.save_model(str(model_path))

    results = {
        "target": target_col,
        "validation": "expanding_window_walk_forward",
        "min_train_quarters": MIN_TRAIN_QUARTERS,
        "total_oos_predictions": int(len(oos_df)),
        "n_folds": len(fold_results),
        "oos_accuracy": float(acc),
        "oos_precision": float(prec),
        "oos_recall": float(rec),
        "oos_f1": float(f1),
        "oos_auc": float(auc),
        "per_fold": fold_results,
        "feature_importances": importances_sorted,
    }
    results_path = MODEL_DIR / f"results_{target_col}.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nModel saved to {model_path.relative_to(PROJECT_ROOT)}")
    print(f"Results saved to {results_path.relative_to(PROJECT_ROOT)}")

    # Save OOS predictions for backtesting
    oos_path = PROJECT_ROOT / "data" / "processed" / f"oos_predictions_{target_col}.csv"
    oos_df.to_csv(oos_path, index=False)
    print(f"OOS predictions saved to {oos_path.name}")

    return final_model, results, oos_df


if __name__ == "__main__":
    walk_forward("target_5d")
    print("\n" + "=" * 60 + "\n")
    walk_forward("target_1d")
