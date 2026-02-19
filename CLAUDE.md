# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Earnings Alpha Engine is a financial ML project that analyzes earnings data to generate trading signals (alpha). It combines market data retrieval, NLP-based earnings analysis, and gradient-boosted/deep-learning models.

## Environment Setup

- Python 3.14 virtualenv at `venv/`
- Activate: `source venv/bin/activate`
- Install deps: `pip install -r requirements.txt`

## Commands

```bash
# Activate environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run all tests
python -m pytest tests/

# Run a single test file
python -m pytest tests/test_example.py

# Run a single test by name
python -m pytest tests/test_example.py::test_function_name -v

# Run scripts
python notebooks/verify_setup.py
```

## Architecture

```
src/           — Core library modules (data ingestion, feature engineering, models, signals)
notebooks/     — Standalone scripts and exploratory analysis
tests/         — pytest test suite mirroring src/ structure
data/raw/      — Raw downloaded data (not committed)
data/processed/ — Transformed/feature-engineered data (not committed)
```

## Key Dependencies

- **pandas / numpy** — Data manipulation
- **yfinance** — Market data retrieval
- **torch / transformers** — Deep learning and NLP (earnings text analysis)
- **xgboost** — Gradient-boosted tree models for signal generation
