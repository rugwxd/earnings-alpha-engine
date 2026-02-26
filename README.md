# Earnings Alpha Engine

A financial ML pipeline that analyzes earnings call transcripts using **FinBERT** sentiment analysis and predicts post-earnings stock price movements with **XGBoost**, validated with expanding-window walk-forward backtesting.

## Hypothesis

Earnings calls contain forward-looking language (guidance, tone, hedging) that may predict short-term price drift beyond the initial market reaction. If FinBERT can quantify the sentiment of analyst Q&A answers, and that sentiment carries information not yet priced in by the close of the reaction day, then a classifier trained on historical sentiment-to-return mappings should generate positive out-of-sample alpha.

## Pipeline

```
Earnings Transcripts → FinBERT Sentiment → Feature Engineering → Walk-Forward XGBoost → Backtest
```

1. **Data ingestion** — Loads ~21K earnings call Q&A pairs from the [lamini/earnings-calls-qa](https://huggingface.co/datasets/lamini/earnings-calls-qa) HuggingFace dataset covering 10 tickers (AAPL, GOOGL, AMZN, META, TSLA, WMT, KO, IBM, JPM, DIS) across 19 quarters (2019-Q1 to 2023-Q3).
2. **Sentiment analysis** — Runs [ProsusAI/FinBERT](https://huggingface.co/ProsusAI/finbert) on each Q&A answer to extract positive/negative/neutral probabilities. Aggregates per earnings call (mean, spread, % positive/negative).
3. **Price data** — Fetches historical stock prices via `yfinance`. Returns are computed from the **first trading day after** the earnings date (reaction day), avoiding lookahead bias from the earnings-day price move itself.
4. **Feature engineering** — 7 features, all available before the market reacts:
   - Sentiment: `pos_mean`, `neg_mean`, `spread`, `pct_positive`, `pct_negative`
   - Price: `pre_earnings_vol` (5-day), `pre_earnings_momentum` (5-day)
5. **Walk-forward training** — Expanding-window validation: train on all quarters up to Q, predict quarter Q+1, retrain with Q included, repeat. Every prediction is truly out-of-sample.
6. **Backtesting** — Long/short strategy on OOS predictions with bootstrap confidence intervals, proper Sharpe annualization, and max drawdown.

## Lookahead Bias Policy

| Information | Available at prediction time? | Used as feature? |
|---|---|---|
| Earnings call transcript text | Yes (call has happened) | Yes (via FinBERT) |
| Pre-earnings stock price | Yes (closes before call date) | Yes (vol, momentum) |
| Earnings-day close / overnight gap | No (market hasn't opened yet) | **No — excluded** |
| Post-earnings returns (target) | No | **No — target only** |
| Future quarters' data | No | **No — walk-forward split** |

## Results

### Walk-Forward Out-of-Sample Performance (71 trades, 13 folds)

| Signal | OOS Accuracy | Excess Return | Sharpe | Win Rate | Max Drawdown | 95% CI (mean) | Significant? |
|--------|-------------|--------------|--------|----------|-------------|---------------|-------------|
| 5-day  | 60.6%       | +32.7%       | 1.18   | 60.6%   | -17.1%      | [-0.10%, +2.17%] | No |
| 1-day  | 53.5%       | +17.1%       | 0.25   | 53.5%   | -10.2%      | [-0.24%, +0.60%] | No |

### Interpretation

The 5-day signal shows promising OOS accuracy (60.6%) and excess return (+32.7% over buy-and-hold), with a Sharpe ratio above 1. However, the **95% bootstrap confidence interval for mean trade return includes zero**, meaning we cannot reject the null hypothesis that the strategy's performance is due to chance at the 95% level. This is expected given the small sample size (71 trades).

## Limitations

- **Small sample size** — 89 total earnings events, 71 OOS predictions. Not enough to establish statistical significance. A production system would need 500+ events.
- **No transaction costs** — Backtest assumes zero commissions, zero bid-ask spread, and instant fills at close prices.
- **No slippage** — Real after-hours/pre-market execution on earnings events has significant slippage.
- **Survivorship bias** — All 10 tickers are large-cap stocks that survived the full period. Distressed/delisted companies are excluded.
- **Single dataset** — All transcripts come from one HuggingFace dataset. Cross-validation with a second source would strengthen confidence.
- **No regime awareness** — The model treats bull markets (2019-2021) and bear markets (2022) the same. Feature drift is likely.

## Project Structure

```
earnings-alpha-engine/
├── src/
│   ├── sentiment_analyzer.py    # FinBERT sentiment scoring + aggregation
│   ├── price_fetcher.py         # yfinance price data + return calculation
│   ├── feature_engineer.py      # Merge sentiment + price into ML features
│   ├── model_trainer.py         # Walk-forward XGBoost training
│   ├── backtester.py            # Strategy backtest + confidence intervals
│   ├── pipeline.py              # End-to-end pipeline runner
│   ├── transcript_fetcher_v2.py # HuggingFace dataset loader
│   └── finbert_test.py          # FinBERT quick test
├── tests/
│   ├── test_sentiment_analyzer.py
│   ├── test_feature_engineer.py
│   └── test_model_trainer.py
├── models/                      # Trained XGBoost models + results JSON
├── plots/                       # Backtest performance charts
├── data/                        # Raw + processed data (gitignored)
├── requirements.txt
└── CLAUDE.md
```

## Setup

```bash
# Clone the repo
git clone https://github.com/rugwed9/earnings-alpha-engine.git
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

### Run individual stages

```bash
python src/transcript_fetcher_v2.py    # 1. Fetch transcripts
python src/sentiment_analyzer.py       # 2. FinBERT sentiment
python src/price_fetcher.py            # 3. Price data
python src/feature_engineer.py         # 4. Features
python src/model_trainer.py            # 5. Walk-forward training
python src/backtester.py               # 6. Backtest
```

### Run tests

```bash
python -m pytest tests/ -v
```

## Key Dependencies

| Package | Purpose |
|---------|---------|
| pandas / numpy | Data manipulation |
| yfinance | Market data retrieval |
| torch / transformers | FinBERT NLP model |
| xgboost | Gradient-boosted tree classifier |
| scikit-learn | Metrics and evaluation |
| matplotlib | Backtest plots |

## License

MIT
