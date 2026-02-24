"""Merge sentiment features with price data to create the ML training dataset.

Joins the per-call sentiment aggregates with post-earnings return data,
creates the binary target (price up/down), and saves the final feature matrix.

LOOKAHEAD BIAS POLICY:
Every feature must be knowable BEFORE the market reacts to earnings.
- Sentiment features: derived from the call transcript (available during/after the call,
  before the next trading session opens).
- Pre-earnings price features: computed from closes before the call date.
- EXCLUDED: overnight_gap, price_reaction_close, or any post-call price data.
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SENTIMENT_CSV = PROJECT_ROOT / "data" / "processed" / "sentiment_by_call.csv"
PRICE_CSV = PROJECT_ROOT / "data" / "processed" / "price_data.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "features.csv"

# Only features available before the market reacts.
# Kept intentionally small (7 features) to reduce overfitting on ~80 samples.
FEATURE_COLS = [
    "sentiment_pos_mean",      # mean positive sentiment across Q&A answers
    "sentiment_neg_mean",      # mean negative sentiment
    "sentiment_spread",        # pos_mean - neg_mean (net tone)
    "pct_positive",            # fraction of answers classified positive
    "pct_negative",            # fraction of answers classified negative
    "pre_earnings_vol",        # 5-day price volatility before the call
    "pre_earnings_momentum",   # 5-day price momentum before the call
]


def build_features():
    """Merge sentiment and price data into a single feature matrix."""
    print("Loading sentiment and price data...")
    sentiment = pd.read_csv(SENTIMENT_CSV)
    prices = pd.read_csv(PRICE_CSV)

    print(f"  Sentiment calls: {len(sentiment)}")
    print(f"  Price rows:      {len(prices)}")

    df = sentiment.merge(prices, on=["ticker", "q"], how="inner")
    print(f"  Merged rows:     {len(df)}")

    # Binary targets
    df["target_1d"] = (df["return_1d"] > 0).astype(int)
    df["target_5d"] = (df["return_5d"] > 0).astype(int)

    # Fill NaNs in pre-earnings features
    df["pre_earnings_vol"] = df["pre_earnings_vol"].fillna(df["pre_earnings_vol"].median())
    df["pre_earnings_momentum"] = df["pre_earnings_momentum"].fillna(0)

    # Sort chronologically for time-series splitting
    df["earnings_date"] = pd.to_datetime(df["earnings_date"])
    df = df.sort_values("earnings_date").reset_index(drop=True)

    print(f"\nFeature matrix: {len(df)} rows x {len(FEATURE_COLS)} features")
    print(f"Features (all pre-reaction):")
    for col in FEATURE_COLS:
        print(f"  {col}")
    print(f"\nTarget distribution (5-day):")
    print(f"  Up:   {df['target_5d'].sum()} ({df['target_5d'].mean():.1%})")
    print(f"  Down: {(1 - df['target_5d']).sum()} ({1 - df['target_5d'].mean():.1%})")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved to {OUTPUT_CSV.name}")
    return df, FEATURE_COLS


if __name__ == "__main__":
    build_features()
