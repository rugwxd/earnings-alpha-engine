"""Tests for the sentiment_analyzer module."""

import pandas as pd
import pytest

from src.sentiment_analyzer import load_model, score_batch


@pytest.fixture(scope="module")
def finbert():
    """Load FinBERT once for the whole test module."""
    tokenizer, model, device = load_model()
    return tokenizer, model, device


class TestFinBERTLoading:
    def test_model_loads(self, finbert):
        tokenizer, model, device = finbert
        assert tokenizer is not None
        assert model is not None
        assert device in ("cpu", "mps", "cuda")


class TestSentimentScoring:
    def test_positive_sentence(self, finbert):
        tokenizer, model, device = finbert
        scores = score_batch(
            ["Revenue grew 40% year-over-year with record margins."],
            tokenizer, model, device,
        )
        pos, neg, neu = scores[0]
        assert pos > neg, "Bullish sentence should score higher positive than negative"

    def test_negative_sentence(self, finbert):
        tokenizer, model, device = finbert
        scores = score_batch(
            ["We experienced severe losses and declining revenue this quarter."],
            tokenizer, model, device,
        )
        pos, neg, neu = scores[0]
        assert neg > pos, "Bearish sentence should score higher negative than positive"

    def test_batch_returns_correct_count(self, finbert):
        tokenizer, model, device = finbert
        texts = ["Good quarter.", "Bad quarter.", "Flat quarter."]
        scores = score_batch(texts, tokenizer, model, device)
        assert len(scores) == 3

    def test_scores_sum_to_one(self, finbert):
        tokenizer, model, device = finbert
        scores = score_batch(["Test sentence."], tokenizer, model, device)
        pos, neg, neu = scores[0]
        assert abs(pos + neg + neu - 1.0) < 1e-4, "Probabilities should sum to 1"

    def test_scores_are_probabilities(self, finbert):
        tokenizer, model, device = finbert
        scores = score_batch(["Some financial text."], tokenizer, model, device)
        for pos, neg, neu in scores:
            assert 0 <= pos <= 1
            assert 0 <= neg <= 1
            assert 0 <= neu <= 1
