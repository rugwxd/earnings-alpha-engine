# Earnings Alpha Engine

A financial ML pipeline that analyzes earnings call transcripts using **FinBERT** sentiment analysis and predicts post-earnings stock price movements with **XGBoost**.

## How It Works

```
Earnings Transcripts → FinBERT Sentiment → Feature Engineering → XGBoost Model → Trading Signal
```

1. **Data ingestion** — Loads ~21K earnings call Q&A pairs from the [lamini/earnings-calls-qa](https://huggingface.co/datasets/lamini/earnings-calls-qa) HuggingFace dataset covering 10 major tickers (AAPL, GOOGL, AMZN, META, TSLA, WMT, KO, IBM, JPM, DIS).
2. **Sentiment analysis** — Runs [ProsusAI/FinBERT](https://huggingface.co/ProsusAI/finbert) on each Q&A answer to extract positive/negative/neutral sentiment scores.
3. **Price data** — Fetches historical stock prices via `yfinance` and computes post-earnings returns (1-day, 5-day, 10-day).
4. **Feature engineering** — Aggregates per-call sentiment features (mean, std, spread, % positive/negative) and merges with price-based features (pre-earnings volatility, momentum, overnight gap).
5. **Model training** — Trains XGBoost classifiers with time-series cross-validation to predict whether a stock goes up or down after earnings.
6. **Backtesting** — Simulates a long/short strategy on held-out data and computes Sharpe ratio, win rate, and cumulative returns.

## Results

| Signal | Excess Return | Sharpe | Win Rate | Test Trades |
|--------|--------------|--------|----------|-------------|
| 1-day  | +42.9%       | 1.28   | 58.8%   | 17          |
| 5-day  | +6.3%        | -0.12  | 47.1%   | 17          |

> **Note:** Results are on a small holdout set (17 trades). This is a research/learning project — not financial advice.

## Project Structure

```
earnings-alpha-engine/
├── src/
│   ├── sentiment_analyzer.py    # FinBERT sentiment scoring + aggregation
│   ├── price_fetcher.py         # yfinance price data + return calculation
│   ├── feature_engineer.py      # Merge sentiment + price into ML features
│   ├── model_trainer.py         # XGBoost training + evaluation
│   ├── backtester.py            # Long/short strategy backtest + plots
│   ├── pipeline.py              # End-to-end pipeline runner
│   ├── transcript_fetcher.py    # SEC EDGAR 10-Q downloader
│   ├── transcript_fetcher_v2.py # HuggingFace dataset loader
│   ├── earnings_transcript_fetcher.py  # FMP API fetcher (needs paid key)
│   └── finbert_test.py          # FinBERT quick test
├── tests/
│   ├── test_sentiment_analyzer.py
│   ├── test_feature_engineer.py
│   └── test_model_trainer.py
├── models/                      # Trained XGBoost models + results JSON
├── plots/                       # Backtest performance charts
├── notebooks/
│   ├── verify_setup.py          # Environment verification
│   └── test_finbert.py          # FinBERT prototype
├── data/                        # Raw + processed data (gitignored)
├── requirements.txt
└── CLAUDE.md
```

## Setup

```bash
# Clone the repo
git clone https://github.com/rugwxd/earnings-alpha-engine.git
cd earnings-alpha-engine

# Create and activate virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# macOS only: XGBoost needs OpenMP
brew install libomp
```

## Usage

### Run the full pipeline

```bash
python src/pipeline.py
```

This runs all stages: sentiment analysis → price fetching → feature engineering → model training → backtesting.

### Run individual stages

```bash
# 1. Fetch and filter earnings transcripts
python src/transcript_fetcher_v2.py

# 2. Run FinBERT sentiment on transcripts
python src/sentiment_analyzer.py

# 3. Fetch stock price data
python src/price_fetcher.py

# 4. Build feature matrix
python src/feature_engineer.py

# 5. Train models
python src/model_trainer.py

# 6. Run backtest
python src/backtester.py
```

### Run tests

```bash
python -m pytest tests/ -v
```

## Key Dependencies

- **pandas / numpy** — Data manipulation
- **yfinance** — Market data retrieval
- **torch / transformers** — FinBERT NLP model
- **xgboost** — Gradient-boosted tree classifier
- **scikit-learn** — Model evaluation and CV
- **matplotlib** — Backtest plots

## Top Predictive Features

| Feature | Importance |
|---------|-----------|
| Pre-earnings volatility | 12.1% |
| Sentiment spread (pos - neg) | 11.0% |
| Pre-earnings momentum | 11.0% |
| % positive answers | 8.9% |
| Negative sentiment std | 8.8% |

## License

MIT
