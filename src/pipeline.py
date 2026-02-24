"""End-to-end pipeline: sentiment analysis -> price data -> features -> model -> backtest.

Run this script to execute the full Earnings Alpha Engine pipeline.
Assumes filtered transcripts already exist at data/processed/filtered_transcripts.csv.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_pipeline():
    """Execute all pipeline stages in order."""
    print("=" * 60)
    print("  EARNINGS ALPHA ENGINE — Full Pipeline")
    print("=" * 60)

    # Stage 1: Sentiment Analysis
    print("\n[Stage 1/5] Running FinBERT sentiment analysis...")
    from src.sentiment_analyzer import aggregate_sentiment, run_sentiment
    df_sentiment = run_sentiment()
    aggregate_sentiment(df_sentiment)

    # Stage 2: Price Data
    print("\n[Stage 2/5] Fetching stock price data...")
    from src.price_fetcher import fetch_prices
    fetch_prices()

    # Stage 3: Feature Engineering
    print("\n[Stage 3/5] Building feature matrix...")
    from src.feature_engineer import build_features
    build_features()

    # Stage 4: Model Training
    print("\n[Stage 4/5] Training XGBoost models...")
    from src.model_trainer import train_model
    train_model("target_5d")
    train_model("target_1d")

    # Stage 5: Backtest
    print("\n[Stage 5/5] Running backtest...")
    from src.backtester import run_backtest
    results_5d = run_backtest("target_5d", "return_5d")
    results_1d = run_backtest("target_1d", "return_1d")

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  5-day signal: {results_5d['excess_return']:+.1%} excess, Sharpe {results_5d['sharpe']:.2f}")
    print(f"  1-day signal: {results_1d['excess_return']:+.1%} excess, Sharpe {results_1d['sharpe']:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
