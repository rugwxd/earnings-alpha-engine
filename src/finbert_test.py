"""FinBERT sentiment analysis on sample earnings call sentences."""

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load the FinBERT tokenizer and model from HuggingFace.
# FinBERT is a BERT model fine-tuned on financial text for sentiment classification
# into three labels: positive, negative, neutral.
MODEL_NAME = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Wrap model and tokenizer in a HuggingFace pipeline for simple inference.
# The pipeline handles tokenization, forward pass, and softmax in one call.
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Sample sentences representing bullish, bearish, and ambiguous earnings language.
sentences = [
    "We delivered exceptional revenue growth and strong margins this quarter.",
    "We are facing significant headwinds in our core markets.",
    "We remain cautiously optimistic about the next quarter.",
]

# Run inference and print results for each sentence.
for sentence in sentences:
    result = nlp(sentence)[0]
    print(f"Sentence:   {sentence}")
    print(f"Sentiment:  {result['label']}")
    print(f"Confidence: {result['score']:.4f}")
    print()
