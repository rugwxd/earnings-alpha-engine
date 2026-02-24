"""Fetch earnings call transcripts for AAPL from Financial Modeling Prep API.

Requires an FMP API key with access to the earning-call-transcript endpoint.
The key is loaded from the .env file in the project root.
"""

import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

# Load environment variables from the .env file at the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Read the FMP API key. Abort early if it's missing.
API_KEY = os.getenv("FMP_API_KEY")
if not API_KEY:
    sys.exit("Error: FMP_API_KEY not found in .env file.")

# Configuration for the request.
TICKER = "AAPL"
NUM_TRANSCRIPTS = 8
BASE_URL = "https://financialmodelingprep.com/stable/earning-call-transcript"

# Directory where transcript text files will be saved.
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / TICKER
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Fetch transcripts from the FMP API.
# The endpoint returns a list of transcript objects sorted by most recent first.
print(f"Fetching {NUM_TRANSCRIPTS} earnings call transcripts for {TICKER}...")
response = requests.get(
    BASE_URL,
    params={"symbol": TICKER, "limit": NUM_TRANSCRIPTS, "apikey": API_KEY},
)

# Check for HTTP errors and report them clearly.
if response.status_code == 402:
    sys.exit(
        "Error: Earnings transcript endpoint requires a paid FMP subscription.\n"
        "Upgrade at https://financialmodelingprep.com/"
    )
if response.status_code != 200:
    sys.exit(f"Error: API returned status {response.status_code}\n{response.text}")

transcripts = response.json()
if not transcripts:
    sys.exit("No transcripts returned — check the ticker or API key.")

print(f"Received {len(transcripts)} transcript(s).\n")

# Process each transcript: print a preview and save to disk.
for t in transcripts:
    date = t.get("date", "unknown-date")
    quarter = t.get("quarter", "?")
    year = t.get("year", "?")
    content = t.get("content", "")

    # Print the date and first 500 characters as a preview.
    label = f"Q{quarter} {year} — {date}"
    print(f"{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(content[:500])
    print("...\n")

    # Save the full transcript to a text file named by quarter and year.
    filename = f"transcript_Q{quarter}_{year}.txt"
    filepath = OUTPUT_DIR / filename
    filepath.write_text(content, encoding="utf-8")

# List all saved files.
saved = sorted(OUTPUT_DIR.glob("transcript_*.txt"))
print(f"Saved {len(saved)} transcript(s) to {OUTPUT_DIR.relative_to(PROJECT_ROOT)}/")
for f in saved:
    print(f"  {f.name}")
