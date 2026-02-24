"""Fetch earnings call transcripts from the HuggingFace lamini/earnings-calls-qa dataset.

This dataset contains ~860K rows of earnings call Q&A pairs across many
public companies, with full transcript snippets, tickers, dates, and quarters.
"""

from pathlib import Path

import pandas as pd
from datasets import load_dataset

# Project paths for saving output.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_CSV = PROJECT_ROOT / "data" / "raw" / "earnings_transcripts.csv"
FILTERED_CSV = PROJECT_ROOT / "data" / "processed" / "filtered_transcripts.csv"
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
FILTERED_CSV.parent.mkdir(parents=True, exist_ok=True)


def fetch_dataset():
    """Download the full dataset from HuggingFace and save as CSV."""

    # Load the dataset from HuggingFace Hub.
    # This downloads and caches the data locally (~200MB) on first run.
    print("Loading lamini/earnings-calls-qa dataset from HuggingFace...")
    ds = load_dataset("lamini/earnings-calls-qa", split="train")
    print(f"Total records available: {len(ds):,}\n")

    # Preview the first 3 rows: show ticker (company), date, and transcript snippet.
    print("First 3 rows:")
    print("=" * 70)
    for i in range(3):
        row = ds[i]
        ticker = row["ticker"]
        date = row["date"]
        quarter = row["q"]
        transcript = row["transcript"]

        print(f"  Company: {ticker}  |  Quarter: {quarter}  |  Date: {date}")
        print(f"  Transcript: {transcript[:300]}...")
        print("-" * 70)

    # Convert the full dataset to a pandas DataFrame and save as CSV.
    # This makes it easy to load later with pandas for feature engineering.
    print(f"\nSaving full dataset to {OUTPUT_CSV.relative_to(PROJECT_ROOT)}...")
    df = ds.to_pandas()
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df):,} rows to {OUTPUT_CSV.name}")


def filter_companies():
    """Filter the raw dataset to keep only select big-tech tickers."""

    TARGET_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN"]

    # Load the full raw dataset from the CSV saved by fetch_dataset().
    print(f"\nLoading raw data from {OUTPUT_CSV.relative_to(PROJECT_ROOT)}...")
    df = pd.read_csv(OUTPUT_CSV)
    print(f"Loaded {len(df):,} rows.")

    # Keep only rows whose ticker is in our target list.
    filtered = df[df["ticker"].isin(TARGET_TICKERS)].copy()
    print(f"Filtered to {len(filtered):,} rows for {TARGET_TICKERS}.\n")

    # Count distinct earnings calls per company.
    # Each unique (ticker, date) pair represents one earnings call;
    # multiple rows share the same call because each row is a Q&A pair.
    calls = filtered.groupby("ticker")["date"].nunique().reset_index()
    calls.columns = ["Ticker", "Num Calls"]

    # Compute the date range per ticker for the summary table.
    date_ranges = (
        filtered.groupby("ticker")["date"]
        .agg(["min", "max"])
        .reset_index()
    )
    date_ranges.columns = ["Ticker", "Earliest", "Latest"]

    summary = calls.merge(date_ranges, on="Ticker")

    # Print the summary table.
    print("Summary:")
    print(f"  {'Ticker':<10} {'Calls':>6}   {'Earliest':<30} {'Latest':<30}")
    print(f"  {'-'*10} {'-'*6}   {'-'*30} {'-'*30}")
    for _, row in summary.iterrows():
        print(
            f"  {row['Ticker']:<10} {row['Num Calls']:>6}   "
            f"{row['Earliest']:<30} {row['Latest']:<30}"
        )

    # Save the filtered data for downstream use.
    filtered.to_csv(FILTERED_CSV, index=False)
    print(f"\nSaved {len(filtered):,} rows to {FILTERED_CSV.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    fetch_dataset()
    filter_companies()
