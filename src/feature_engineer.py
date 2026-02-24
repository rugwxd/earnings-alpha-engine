"""Merge sentiment features with price data to create the ML training dataset.

Joins the per-call sentiment aggregates with post-earnings return data,
creates the binary target (price up/down), and saves the final feature matrix.
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SENTIMENT_CSV = PROJECT_ROOT / "data" / "processed" / "sentiment_by_call.csv"
PRICE_CSV = PROJECT_ROOT / "data" / "processed" / "price_data.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "features.csv"


def build_features():
    """Merge sentiment and price data into a single feature matrix."""
    print("Loading sentiment and price data...")
    sentiment = pd.read_csv(SENTIMENT_CSV)
    prices = pd.read_csv(PRICE_CSV)

    print(f"  Sentiment calls: {len(sentiment)}")
    print(f"  Price rows:      {len(prices)}")

    # Merge on ticker + quarter
    df = sentiment.merge(prices, on=["ticker", "q"], how="inner")
    print(f"  Merged rows:     {len(df)}")

    # Create binary targets
    df["target_1d"] = (df["return_1d"] > 0).astype(int)
    df["target_5d"] = (df["return_5d"] > 0).astype(int)

    # Feature columns
    feature_cols = [
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

    # Fill any NaN in std columns (single-row groups) with 0
    df["sentiment_pos_std"] = df["sentiment_pos_std"].fillna(0)
    df["sentiment_neg_std"] = df["sentiment_neg_std"].fillna(0)
    df["pre_earnings_vol"] = df["pre_earnings_vol"].fillna(df["pre_earnings_vol"].median())
    df["pre_earnings_momentum"] = df["pre_earnings_momentum"].fillna(0)

    # Sort by date for proper time-series splitting later
    df["earnings_date"] = pd.to_datetime(df["earnings_date"])
    df = df.sort_values("earnings_date").reset_index(drop=True)

    print(f"\nFeature matrix: {len(df)} rows x {len(feature_cols)} features")
    print(f"Target distribution (5-day):")
    print(f"  Up:   {df['target_5d'].sum()} ({df['target_5d'].mean():.1%})")
    print(f"  Down: {(1 - df['target_5d']).sum()} ({1 - df['target_5d'].mean():.1%})")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved to {OUTPUT_CSV.name}")
    return df, feature_cols


if __name__ == "__main__":
    build_features()
