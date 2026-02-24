"""Tests for the feature_engineer module."""

from pathlib import Path

import pandas as pd
import pytest

from src.feature_engineer import FEATURE_COLS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_CSV = PROJECT_ROOT / "data" / "processed" / "features.csv"


@pytest.fixture(scope="module")
def features_df():
    """Load the generated features dataset."""
    if not FEATURES_CSV.exists():
        pytest.skip("features.csv not found — run the pipeline first")
    return pd.read_csv(FEATURES_CSV)


class TestFeaturesDataset:
    def test_has_rows(self, features_df):
        assert len(features_df) > 0

    def test_required_columns_exist(self, features_df):
        required = FEATURE_COLS + ["ticker", "q", "earnings_date", "return_1d", "return_5d", "target_1d", "target_5d"]
        for col in required:
            assert col in features_df.columns, f"Missing column: {col}"

    def test_no_lookahead_features(self, features_df):
        """overnight_gap and price_at_open must NOT be in the feature set."""
        banned = ["overnight_gap", "price_at_open"]
        for col in banned:
            assert col not in FEATURE_COLS, f"Lookahead feature found: {col}"

    def test_targets_are_binary(self, features_df):
        assert set(features_df["target_1d"].unique()).issubset({0, 1})
        assert set(features_df["target_5d"].unique()).issubset({0, 1})

    def test_sentiment_scores_in_range(self, features_df):
        for col in ["sentiment_pos_mean", "sentiment_neg_mean"]:
            assert features_df[col].min() >= 0, f"{col} has values below 0"
            assert features_df[col].max() <= 1, f"{col} has values above 1"

    def test_no_null_features(self, features_df):
        for col in FEATURE_COLS:
            assert features_df[col].isna().sum() == 0, f"{col} has NaN values"

    def test_sorted_by_date(self, features_df):
        dates = pd.to_datetime(features_df["earnings_date"])
        assert dates.is_monotonic_increasing

    def test_feature_count_is_small(self, features_df):
        """With ~89 samples, we should have fewer than 10 features to limit overfitting."""
        assert len(FEATURE_COLS) < 10, f"Too many features ({len(FEATURE_COLS)}) for {len(features_df)} samples"
