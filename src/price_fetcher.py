"""Fetch stock price data around earnings dates using yfinance.

For each earnings call in the sentiment dataset, pulls price data and
computes post-earnings returns (1-day, 5-day, 10-day) as prediction targets.
"""

from pathlib import Path

import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SENTIMENT_CSV = PROJECT_ROOT / "data" / "processed" / "sentiment_by_call.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "price_data.csv"

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "WMT", "KO", "IBM", "JPM", "DIS"]


def parse_earnings_date(date_str):
    """Parse date strings like 'Oct 29, 2020, 5:00 p.m. ET' into datetime."""
    # Strip time and timezone — keep only 'Month Day, Year'
    import re
    match = re.match(r"([A-Za-z]+\s+\d{1,2},\s*\d{4})", date_str)
    if match:
        return pd.to_datetime(match.group(1))
    return pd.NaT


def fetch_prices():
    """Download price data for all tickers and compute post-earnings returns."""
    print("Loading sentiment by call...")
    sentiment = pd.read_csv(SENTIMENT_CSV)
    sentiment["earnings_date"] = sentiment["date"].apply(parse_earnings_date)
    sentiment = sentiment.dropna(subset=["earnings_date"])
    print(f"Parsed {len(sentiment)} earnings dates.")

    # Determine overall date range needed (with buffer)
    min_date = sentiment["earnings_date"].min() - pd.Timedelta(days=30)
    max_date = sentiment["earnings_date"].max() + pd.Timedelta(days=30)

    # Download price data for all tickers at once
    tickers = sorted(sentiment["ticker"].unique().tolist())
    print(f"Fetching price data for {tickers} from {min_date.date()} to {max_date.date()}...")
    price_data = yf.download(tickers, start=min_date, end=max_date, auto_adjust=True)

    # yf.download returns MultiIndex columns (metric, ticker) for multiple tickers
    close = price_data["Close"]

    results = []
    for _, row in sentiment.iterrows():
        ticker = row["ticker"]
        ed = row["earnings_date"]

        if ticker not in close.columns:
            continue

        series = close[ticker].dropna()

        # Find the next trading day on or after earnings
        future_dates = series.index[series.index >= ed]
        if len(future_dates) < 11:
            continue
        t0 = future_dates[0]

        # Find the previous trading day (day before earnings)
        past_dates = series.index[series.index < ed]
        if len(past_dates) < 6:
            continue
        t_minus1 = past_dates[-1]

        price_at_earnings = series[t0]
        price_before = series[t_minus1]

        # Post-earnings returns
        ret_1d = (series[future_dates[1]] / price_at_earnings) - 1 if len(future_dates) > 1 else None
        ret_5d = (series[future_dates[min(5, len(future_dates) - 1)]] / price_at_earnings) - 1
        ret_10d = (series[future_dates[min(10, len(future_dates) - 1)]] / price_at_earnings) - 1

        # Pre-earnings features: 5-day volatility and momentum
        pre_window = series[past_dates[-5:]]
        pre_vol = pre_window.pct_change().std() if len(pre_window) >= 2 else None
        pre_momentum = (price_before / series[past_dates[-5]]) - 1 if len(past_dates) >= 5 else None

        # Overnight gap (earnings reaction)
        overnight_gap = (price_at_earnings / price_before) - 1

        results.append({
            "ticker": ticker,
            "q": row["q"],
            "earnings_date": ed,
            "price_before": float(price_before),
            "price_at_open": float(price_at_earnings),
            "overnight_gap": float(overnight_gap),
            "return_1d": float(ret_1d) if ret_1d is not None else None,
            "return_5d": float(ret_5d),
            "return_10d": float(ret_10d),
            "pre_earnings_vol": float(pre_vol) if pre_vol is not None else None,
            "pre_earnings_momentum": float(pre_momentum) if pre_momentum is not None else None,
        })

    price_df = pd.DataFrame(results)
    price_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(price_df)} rows with price data to {OUTPUT_CSV.name}")
    print(f"\nReturn stats:")
    print(price_df[["return_1d", "return_5d", "return_10d"]].describe().to_string())
    return price_df


if __name__ == "__main__":
    fetch_prices()
