"""Download FinBERT and run sentiment analysis on sample sentences."""

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

MODEL_NAME = "ProsusAI/finbert"

print(f"Loading FinBERT from '{MODEL_NAME}' (will download on first run)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
print("Model loaded and cached.\n")

nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Placeholder sentences — replace with user-provided ones
sentences = [
    "The company reported record earnings, beating analyst expectations by 20%.",
    "Revenue declined sharply due to weak consumer demand and rising costs.",
    "The firm maintained its quarterly dividend at $0.50 per share.",
]

print("FinBERT Sentiment Analysis Results:")
print("-" * 60)
for sentence in sentences:
    result = nlp(sentence)[0]
    print(f"  Sentence: {sentence}")
    print(f"  Label: {result['label']:<10s}  Score: {result['score']:.4f}")
    print()
