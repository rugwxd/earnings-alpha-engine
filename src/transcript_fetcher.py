"""Download the latest quarterly SEC filings for Apple (AAPL) from EDGAR.

Note: SEC EDGAR does not host earnings call transcripts (those are proprietary).
This script downloads 10-Q filings, which are the official quarterly reports
containing the financial data discussed on earnings calls.
"""

from pathlib import Path
from sec_edgar_downloader import Downloader

# SEC EDGAR requires a company name and email for its User-Agent header.
# This is a fair-access compliance requirement, not authentication.
COMPANY_NAME = "EarningsAlphaEngine"
EMAIL = "research@example.com"

# Ticker and filing configuration.
TICKER = "AAPL"
FILING_TYPE = "10-Q"
NUM_FILINGS = 4

# Save filings under data/raw/AAPL/ relative to the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOWNLOAD_DIR = PROJECT_ROOT / "data" / "raw"

# Initialize the downloader with our identity and target directory.
dl = Downloader(COMPANY_NAME, EMAIL, DOWNLOAD_DIR)

# Download the most recent 10-Q filings from EDGAR.
# download_details=True fetches the human-readable HTML alongside the raw text.
print(f"Downloading {NUM_FILINGS} latest {FILING_TYPE} filings for {TICKER}...")
count = dl.get(FILING_TYPE, TICKER, limit=NUM_FILINGS, download_details=True)
print(f"Downloaded {count} filing(s).\n")

# Walk the output directory to list all downloaded files.
filing_dir = DOWNLOAD_DIR / "sec-edgar-filings" / TICKER / FILING_TYPE
if filing_dir.exists():
    files = sorted(filing_dir.rglob("*"))
    files = [f for f in files if f.is_file()]
    print(f"Total files on disk: {len(files)}")
    for f in files:
        print(f"  {f.relative_to(PROJECT_ROOT)}")
else:
    print("No filings found — the download may have failed.")
