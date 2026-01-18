"""
VADER-based sentiment analyzer as a lightweight fallback.

Uses NLTK's VADER with financial domain lexicon additions.
Suitable for quick analysis or when GPU is unavailable.
"""

from typing import Optional

import pandas as pd
import numpy as np

from .finbert import SentimentResult


class VaderAnalyzer:
    """
    Financial sentiment analyzer using VADER with domain-specific lexicon.

    Features:
    - Lightweight, no GPU required
    - Enhanced with 30+ financial domain terms
    - Fast processing for large batches
    - Consistent output format with FinBertAnalyzer

    Example:
        >>> analyzer = VaderAnalyzer()
        >>> result = analyzer.analyze("Stock surges on bullish earnings report")
        >>> print(f"Score: {result.score:.3f}, Signal: {result.signal}")
        Score: 0.847, Signal: bullish
    """

    # Financial domain lexicon additions
    # Values are sentiment intensity scores (typically -4 to +4)
    FINANCIAL_LEXICON = {
        # Strong positive
        "bullish": 3.0,
        "surge": 2.5,
        "soar": 2.5,
        "skyrocket": 3.0,
        "outperform": 2.0,
        "upgrade": 2.5,
        "beat": 2.0,
        "exceeds": 2.0,
        "exceeded": 2.0,
        "record": 1.5,
        "breakthrough": 2.5,
        "rally": 2.0,
        "boom": 2.0,
        "gain": 1.5,
        "gains": 1.5,
        "profit": 1.5,
        "profitable": 1.5,
        "growth": 1.5,
        "growing": 1.5,
        "strong": 1.0,
        "positive": 1.5,
        "optimistic": 2.0,
        "momentum": 1.0,
        "upside": 1.5,
        "buy": 1.5,
        "accumulate": 1.0,

        # Strong negative
        "bearish": -3.0,
        "plunge": -2.5,
        "plummet": -3.0,
        "crash": -3.5,
        "collapse": -3.0,
        "underperform": -2.0,
        "downgrade": -2.5,
        "miss": -2.0,
        "missed": -2.0,
        "misses": -2.0,
        "loss": -2.0,
        "losses": -2.0,
        "decline": -2.0,
        "declining": -2.0,
        "drop": -1.5,
        "drops": -1.5,
        "fall": -1.5,
        "falls": -1.5,
        "weak": -1.5,
        "weakness": -1.5,
        "negative": -1.5,
        "pessimistic": -2.0,
        "sell": -1.5,
        "selloff": -2.5,
        "downturn": -2.0,
        "recession": -2.5,
        "bankruptcy": -3.5,
        "default": -3.0,
        "fraud": -3.5,
        "scandal": -3.0,
        "investigation": -2.0,
        "lawsuit": -2.0,
        "layoffs": -2.0,
        "layoff": -2.0,
        "cuts": -1.5,
        "warning": -2.0,
        "concern": -1.5,
        "concerns": -1.5,
        "risk": -1.0,
        "risky": -1.5,
        "volatile": -1.0,
        "volatility": -1.0,
        "uncertainty": -1.5,
        "headwind": -1.5,
        "headwinds": -1.5,

        # Moderate/Neutral with context
        "stable": 0.5,
        "steady": 0.5,
        "flat": 0.0,
        "unchanged": 0.0,
        "hold": 0.0,
        "neutral": 0.0,
        "mixed": 0.0,
    }

    MODEL_NAME = "vader_financial"
    MIN_TEXT_LENGTH = 5

    def __init__(self, use_financial_lexicon: bool = True):
        """
        Initialize the VADER analyzer.

        Args:
            use_financial_lexicon: Whether to add financial domain terms to VADER.
        """
        self.use_financial_lexicon = use_financial_lexicon
        self._analyzer = None

    def _ensure_loaded(self) -> None:
        """Ensure VADER is loaded and configured."""
        if self._analyzer is not None:
            return

        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            import nltk

            # Download vader_lexicon if needed
            try:
                nltk.data.find("sentiment/vader_lexicon.zip")
            except LookupError:
                nltk.download("vader_lexicon", quiet=True)

            self._analyzer = SentimentIntensityAnalyzer()

            # Add financial lexicon
            if self.use_financial_lexicon:
                self._analyzer.lexicon.update(self.FINANCIAL_LEXICON)

        except ImportError as e:
            raise ImportError(
                "nltk is required for VADER. Install with: pip install nltk"
            ) from e

    def analyze(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of a single text.

        Args:
            text: Text to analyze.

        Returns:
            SentimentResult with score, confidence, and probabilities.
        """
        # Handle empty or very short text
        if not text or len(text.strip()) < self.MIN_TEXT_LENGTH:
            return SentimentResult(
                text=text or "",
                score=0.0,
                confidence=0.0,
                positive_prob=0.0,
                negative_prob=0.0,
                neutral_prob=0.0,
                model=self.MODEL_NAME,
                error=f"Text too short (min {self.MIN_TEXT_LENGTH} characters)"
            )

        try:
            self._ensure_loaded()

            # Get VADER scores
            scores = self._analyzer.polarity_scores(text.strip())

            # VADER returns: neg, neu, pos, compound
            # compound is already in [-1, +1] range
            compound = scores["compound"]
            pos = scores["pos"]
            neg = scores["neg"]
            neu = scores["neu"]

            # Use compound as score, confidence based on extremity
            confidence = abs(compound)

            return SentimentResult(
                text=text,
                score=compound,
                confidence=confidence,
                positive_prob=pos,
                negative_prob=neg,
                neutral_prob=neu,
                model=self.MODEL_NAME,
                error=None
            )

        except Exception as e:
            return SentimentResult(
                text=text,
                score=0.0,
                confidence=0.0,
                positive_prob=0.0,
                negative_prob=0.0,
                neutral_prob=0.0,
                model=self.MODEL_NAME,
                error=str(e)
            )

    def analyze_batch(
        self,
        df: pd.DataFrame,
        text_column: str = "headline",
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Analyze sentiment for a batch of articles.

        Args:
            df: DataFrame with articles.
            text_column: Column containing text to analyze.
            show_progress: Whether to show tqdm progress bar.

        Returns:
            DataFrame with added sentiment columns:
            - sentiment_score: [-1, +1]
            - sentiment_confidence: [0, 1]
            - prob_positive, prob_negative, prob_neutral
            - sentiment_signal: 'bullish'/'bearish'/'neutral'
        """
        if df.empty:
            return df.copy()

        # Make a copy
        result_df = df.copy()

        # Ensure VADER is loaded
        self._ensure_loaded()

        # Initialize columns
        result_df["sentiment_score"] = 0.0
        result_df["sentiment_confidence"] = 0.0
        result_df["prob_positive"] = 0.0
        result_df["prob_negative"] = 0.0
        result_df["prob_neutral"] = 0.0
        result_df["sentiment_signal"] = "neutral"

        # Process with optional progress bar
        try:
            from tqdm import tqdm
            iterator = df.iterrows()
            if show_progress:
                iterator = tqdm(iterator, total=len(df), desc="VADER sentiment", unit="article")
        except ImportError:
            iterator = df.iterrows()

        for idx, row in iterator:
            text = str(row.get(text_column, "")).strip()

            if len(text) >= self.MIN_TEXT_LENGTH:
                result = self.analyze(text)

                result_df.at[idx, "sentiment_score"] = result.score
                result_df.at[idx, "sentiment_confidence"] = result.confidence
                result_df.at[idx, "prob_positive"] = result.positive_prob
                result_df.at[idx, "prob_negative"] = result.negative_prob
                result_df.at[idx, "prob_neutral"] = result.neutral_prob
                result_df.at[idx, "sentiment_signal"] = result.signal

        return result_df

    def get_lexicon_info(self) -> dict:
        """Get information about the current lexicon configuration."""
        self._ensure_loaded()

        return {
            "model_name": self.MODEL_NAME,
            "financial_lexicon_enabled": self.use_financial_lexicon,
            "financial_terms_count": len(self.FINANCIAL_LEXICON),
            "total_lexicon_size": len(self._analyzer.lexicon) if self._analyzer else 0,
            "sample_financial_terms": {
                k: v for k, v in list(self.FINANCIAL_LEXICON.items())[:10]
            }
        }


def get_analyzer(use_finbert: bool = True) -> "FinBertAnalyzer | VaderAnalyzer":
    """
    Factory function to get the appropriate sentiment analyzer.

    Args:
        use_finbert: If True, tries to use FinBERT. Falls back to VADER on error.

    Returns:
        Sentiment analyzer instance.
    """
    if use_finbert:
        try:
            from .finbert import FinBertAnalyzer
            return FinBertAnalyzer()
        except ImportError:
            pass

    return VaderAnalyzer()
