"""Tests for the model_trainer module (walk-forward validation)."""

import json
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"


class TestWalkForwardResults:
    @pytest.mark.parametrize("target", ["target_5d", "target_1d"])
    def test_model_file_exists(self, target):
        model_path = MODEL_DIR / f"xgb_{target}.json"
        if not model_path.exists():
            pytest.skip("Model not trained yet — run the pipeline first")
        assert model_path.stat().st_size > 0

    @pytest.mark.parametrize("target", ["target_5d", "target_1d"])
    def test_results_use_walk_forward(self, target):
        results_path = MODEL_DIR / f"results_{target}.json"
        if not results_path.exists():
            pytest.skip("Results not generated yet")
        results = json.loads(results_path.read_text())
        assert results["validation"] == "expanding_window_walk_forward"
        assert results["n_folds"] > 1

    @pytest.mark.parametrize("target", ["target_5d", "target_1d"])
    def test_oos_predictions_larger_than_old_holdout(self, target):
        """Walk-forward should produce more OOS predictions than a static 80/20 split."""
        results_path = MODEL_DIR / f"results_{target}.json"
        if not results_path.exists():
            pytest.skip("Results not generated yet")
        results = json.loads(results_path.read_text())
        assert results["total_oos_predictions"] > 20, \
            f"Only {results['total_oos_predictions']} OOS predictions — expected >20"

    @pytest.mark.parametrize("target", ["target_5d", "target_1d"])
    def test_accuracy_above_random(self, target):
        results_path = MODEL_DIR / f"results_{target}.json"
        if not results_path.exists():
            pytest.skip("Results not generated yet")
        results = json.loads(results_path.read_text())
        assert results["oos_accuracy"] > 0.30

    @pytest.mark.parametrize("target", ["target_5d", "target_1d"])
    def test_oos_predictions_file_exists(self, target):
        pred_path = PROJECT_ROOT / "data" / "processed" / f"oos_predictions_{target}.csv"
        if not pred_path.exists():
            pytest.skip("Predictions not generated yet")
        df = pd.read_csv(pred_path)
        assert len(df) > 0
        assert "prediction" in df.columns
        assert "pred_prob" in df.columns
        assert "fold" in df.columns

    @pytest.mark.parametrize("target", ["target_5d", "target_1d"])
    def test_no_lookahead_in_features(self, target):
        results_path = MODEL_DIR / f"results_{target}.json"
        if not results_path.exists():
            pytest.skip("Results not generated yet")
        results = json.loads(results_path.read_text())
        feature_names = list(results["feature_importances"].keys())
        banned = ["overnight_gap", "price_at_open", "price_reaction_close",
                   "return_1d", "return_5d", "return_10d", "target_1d", "target_5d"]
        for feat in feature_names:
            assert feat not in banned, f"Lookahead feature in model: {feat}"
