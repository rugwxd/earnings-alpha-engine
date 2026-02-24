"""Run FinBERT sentiment analysis on earnings call Q&A answers.

Processes the filtered transcripts CSV, scores each answer with FinBERT
(positive / negative / neutral), and saves the results. Then aggregates
sentiment per earnings call (ticker + quarter) for downstream feature engineering.
"""

from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "filtered_transcripts.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "transcripts_with_sentiment.csv"
AGG_CSV = PROJECT_ROOT / "data" / "processed" / "sentiment_by_call.csv"

MODEL_NAME = "ProsusAI/finbert"
BATCH_SIZE = 32
MAX_LENGTH = 512


def load_model():
    """Load the FinBERT tokenizer and model."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def score_batch(texts, tokenizer, model, device):
    """Run FinBERT on a batch of texts. Returns list of (pos, neg, neu) tuples."""
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    # FinBERT labels: 0=positive, 1=negative, 2=neutral
    return [(float(p[0]), float(p[1]), float(p[2])) for p in probs]


def run_sentiment(df=None):
    """Score every answer in the filtered transcripts with FinBERT sentiment."""
    if df is None:
        print(f"Loading transcripts from {INPUT_CSV.name}...")
        df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df):,} rows.")

    tokenizer, model, device = load_model()
    print(f"FinBERT loaded on {device}. Scoring {len(df):,} answers in batches of {BATCH_SIZE}...")

    answers = df["answer"].fillna("").astype(str).tolist()
    all_scores = []

    for i in range(0, len(answers), BATCH_SIZE):
        batch = answers[i : i + BATCH_SIZE]
        scores = score_batch(batch, tokenizer, model, device)
        all_scores.extend(scores)
        if (i // BATCH_SIZE) % 50 == 0:
            print(f"  Processed {min(i + BATCH_SIZE, len(answers)):,} / {len(answers):,}")

    df["sentiment_positive"] = [s[0] for s in all_scores]
    df["sentiment_negative"] = [s[1] for s in all_scores]
    df["sentiment_neutral"] = [s[2] for s in all_scores]
    df["sentiment_label"] = df[["sentiment_positive", "sentiment_negative", "sentiment_neutral"]].idxmax(axis=1).str.replace("sentiment_", "")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df):,} rows with sentiment to {OUTPUT_CSV.name}")
    return df


def aggregate_sentiment(df=None):
    """Aggregate per-answer sentiment to per-call (ticker + quarter) level."""
    if df is None:
        print(f"Loading sentiment data from {OUTPUT_CSV.name}...")
        df = pd.read_csv(OUTPUT_CSV)

    agg = df.groupby(["ticker", "q", "date"]).agg(
        sentiment_pos_mean=("sentiment_positive", "mean"),
        sentiment_pos_std=("sentiment_positive", "std"),
        sentiment_neg_mean=("sentiment_negative", "mean"),
        sentiment_neg_std=("sentiment_negative", "std"),
        sentiment_neu_mean=("sentiment_neutral", "mean"),
        sentiment_spread=("sentiment_positive", lambda x: x.mean() - df.loc[x.index, "sentiment_negative"].mean()),
        num_qa_pairs=("answer", "count"),
        pct_positive=("sentiment_label", lambda x: (x == "positive").mean()),
        pct_negative=("sentiment_label", lambda x: (x == "negative").mean()),
    ).reset_index()

    agg.to_csv(AGG_CSV, index=False)
    print(f"Aggregated to {len(agg)} earnings calls. Saved to {AGG_CSV.name}")
    return agg


if __name__ == "__main__":
    df = run_sentiment()
    aggregate_sentiment(df)
