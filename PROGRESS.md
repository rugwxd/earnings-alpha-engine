# PROGRESS.md

## Project Status: COMPLETE

All pipeline stages are built, tested, and producing results.

## What We Built

### Phase 1: Data Foundation

1. **Project Scaffold & Environment** — Python 3.11 virtualenv, `requirements.txt` with 145+ packages, git repo with `.gitignore`.

2. **FinBERT Sentiment Model** — `ProsusAI/finbert` loaded from HuggingFace, tested on sample sentences. Classifies financial text as positive/negative/neutral.

3. **Earnings Transcript Data** — Loaded `lamini/earnings-calls-qa` dataset (860K rows) from HuggingFace. Filtered to 21,229 Q&A pairs across 10 tickers (AAPL, GOOGL, AMZN, META, TSLA, WMT, KO, IBM, JPM, DIS).

### Phase 2: ML Pipeline

4. **Sentiment Analysis** (`src/sentiment_analyzer.py`) — Ran FinBERT on all 21,229 answers in batches of 32 using MPS acceleration. Produced per-answer sentiment scores, then aggregated to 90 earnings calls with mean/std/spread features.

5. **Price Data** (`src/price_fetcher.py`) — Fetched historical prices for all 10 tickers via `yfinance`. Computed post-earnings returns (1d, 5d, 10d), pre-earnings volatility, momentum, and overnight gap for 81 earnings events.

6. **Feature Engineering** (`src/feature_engineer.py`) — Merged 12 features (sentiment + price-based) with binary targets. 81 samples, ~52/48 class balance.

7. **XGBoost Models** (`src/model_trainer.py`) — Trained with 3-fold time-series CV and 80/20 chronological holdout. Two models: 1-day and 5-day post-earnings direction prediction.

8. **Backtesting** (`src/backtester.py`) — Long/short strategy simulation on held-out test data with performance plots.

### Phase 3: Testing & Docs

9. **Test Suite** — 20 pytest tests covering sentiment scoring, feature integrity, and model artifacts. All passing.

10. **Documentation** — README with setup instructions, architecture, results. CLAUDE.md for Claude Code context.

## Results

| Signal | Excess Return | Sharpe | Win Rate |
|--------|--------------|--------|----------|
| 1-day  | +42.9%       | 1.28   | 58.8%   |
| 5-day  | +6.3%        | -0.12  | 47.1%   |

Top features: pre-earnings volatility, sentiment spread, pre-earnings momentum.

## File Structure

```
earnings-alpha-engine/
├── src/
│   ├── __init__.py
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
│   ├── __init__.py
│   ├── test_sentiment_analyzer.py
│   ├── test_feature_engineer.py
│   └── test_model_trainer.py
├── models/
│   ├── xgb_target_1d.json       # Trained XGBoost model (1-day)
│   ├── xgb_target_5d.json       # Trained XGBoost model (5-day)
│   ├── results_target_1d.json   # Evaluation metrics
│   └── results_target_5d.json   # Evaluation metrics
├── plots/
│   ├── backtest_target_1d.png   # 1-day backtest chart
│   └── backtest_target_5d.png   # 5-day backtest chart
├── notebooks/
│   ├── verify_setup.py
│   └── test_finbert.py
├── data/                         # Raw + processed data (gitignored)
├── requirements.txt
├── CLAUDE.md
├── PROGRESS.md
└── README.md
```
