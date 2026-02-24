"""Tests for the model_trainer module."""

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"


class TestModelArtifacts:
    @pytest.mark.parametrize("target", ["target_5d", "target_1d"])
    def test_model_file_exists(self, target):
        model_path = MODEL_DIR / f"xgb_{target}.json"
        if not model_path.exists():
            pytest.skip("Model not trained yet — run the pipeline first")
        assert model_path.stat().st_size > 0

    @pytest.mark.parametrize("target", ["target_5d", "target_1d"])
    def test_results_file_exists(self, target):
        results_path = MODEL_DIR / f"results_{target}.json"
        if not results_path.exists():
            pytest.skip("Results not generated yet — run the pipeline first")
        results = json.loads(results_path.read_text())
        assert "holdout_accuracy" in results
        assert "holdout_f1" in results
        assert "feature_importances" in results

    @pytest.mark.parametrize("target", ["target_5d", "target_1d"])
    def test_accuracy_above_random(self, target):
        results_path = MODEL_DIR / f"results_{target}.json"
        if not results_path.exists():
            pytest.skip("Results not generated yet")
        results = json.loads(results_path.read_text())
        # CV accuracy should be better than 30% (generous lower bound)
        assert results["cv_accuracy_mean"] > 0.30

    @pytest.mark.parametrize("target", ["target_5d", "target_1d"])
    def test_predictions_file_exists(self, target):
        pred_path = PROJECT_ROOT / "data" / "processed" / f"test_predictions_{target}.csv"
        if not pred_path.exists():
            pytest.skip("Predictions not generated yet")
        import pandas as pd
        df = pd.read_csv(pred_path)
        assert len(df) > 0
        assert "prediction" in df.columns
        assert "pred_prob" in df.columns
