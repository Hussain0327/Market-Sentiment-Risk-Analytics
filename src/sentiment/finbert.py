"""
FinBERT-based financial sentiment analyzer.

Uses ProsusAI/finbert model for financial text sentiment analysis.
Provides both single-text and batch processing capabilities.
"""

from dataclasses import dataclass, field
from typing import Optional, ClassVar
import warnings

import pandas as pd
import numpy as np

# Suppress tokenizer parallelism warning
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class SentimentResult:
    """
    Unified sentiment analysis result.

    Attributes:
        text: The analyzed text.
        score: Sentiment score in [-1, +1] range (bearish to bullish).
        confidence: Model confidence in [0, 1] range.
        positive_prob: Probability of positive sentiment.
        negative_prob: Probability of negative sentiment.
        neutral_prob: Probability of neutral sentiment.
        model: Name of the model used for analysis.
        error: Error message if analysis failed.
    """
    text: str
    score: float
    confidence: float
    positive_prob: float
    negative_prob: float
    neutral_prob: float
    model: str
    error: Optional[str] = None

    @property
    def signal(self) -> str:
        """
        Get trading signal based on sentiment score.

        Returns:
            'bullish' if score > 0.1
            'bearish' if score < -0.1
            'neutral' otherwise
        """
        if self.score > 0.1:
            return "bullish"
        elif self.score < -0.1:
            return "bearish"
        else:
            return "neutral"

    @property
    def is_valid(self) -> bool:
        """Check if the result is valid (no errors)."""
        return self.error is None


class FinBertAnalyzer:
    """
    Financial sentiment analyzer using FinBERT model.

    Features:
    - Class-level model caching to avoid repeated loading
    - Automatic GPU/CPU detection
    - Batch processing with progress tracking
    - Handles empty/short text gracefully

    Example:
        >>> analyzer = FinBertAnalyzer()
        >>> result = analyzer.analyze("Apple reports record quarterly revenue")
        >>> print(f"Score: {result.score:.3f}, Signal: {result.signal}")
        Score: 0.782, Signal: bullish
    """

    # Class-level model cache (shared across all instances)
    _model_cache: ClassVar[dict] = {}

    DEFAULT_MODEL = "ProsusAI/finbert"
    DEFAULT_BATCH_SIZE = 32
    MIN_TEXT_LENGTH = 10

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE
    ):
        """
        Initialize the FinBERT analyzer.

        Args:
            model_name: HuggingFace model name. Default: ProsusAI/finbert
            device: Device to run model on ('cuda', 'mps', 'cpu', or None for auto-detect)
            batch_size: Batch size for batch processing. Default: 32
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self._device = device

        # Lazy load model on first use
        self._pipeline = None

    @property
    def device(self) -> str:
        """Get the device to use for inference."""
        if self._device is not None:
            return self._device

        # Auto-detect device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass

        return "cpu"

    def _get_pipeline(self):
        """Get or create the sentiment analysis pipeline."""
        cache_key = f"{self.model_name}:{self.device}"

        if cache_key not in FinBertAnalyzer._model_cache:
            try:
                from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

                # Load model and tokenizer
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

                # Determine device index for pipeline
                device_arg = None
                if self.device == "cuda":
                    device_arg = 0
                elif self.device == "mps":
                    device_arg = "mps"
                # For CPU, leave as None

                # Create pipeline
                pipe = pipeline(
                    "sentiment-analysis",
                    model=model,
                    tokenizer=tokenizer,
                    device=device_arg,
                    truncation=True,
                    max_length=512
                )

                FinBertAnalyzer._model_cache[cache_key] = pipe

            except ImportError as e:
                raise ImportError(
                    "transformers and torch are required for FinBERT. "
                    "Install with: pip install transformers torch"
                ) from e

        return FinBertAnalyzer._model_cache[cache_key]

    def _ensure_loaded(self) -> None:
        """Ensure the model is loaded."""
        if self._pipeline is None:
            self._pipeline = self._get_pipeline()

    def _parse_finbert_output(self, output: dict, text: str) -> SentimentResult:
        """
        Parse FinBERT pipeline output to SentimentResult.

        FinBERT returns: {'label': 'positive/negative/neutral', 'score': probability}
        We need to convert this to our unified format.
        """
        label = output["label"].lower()
        prob = output["score"]

        # Initialize probabilities
        positive_prob = 0.0
        negative_prob = 0.0
        neutral_prob = 0.0

        # Set the probability for the predicted label
        if label == "positive":
            positive_prob = prob
            # Distribute remaining probability
            remaining = 1.0 - prob
            negative_prob = remaining * 0.3
            neutral_prob = remaining * 0.7
        elif label == "negative":
            negative_prob = prob
            remaining = 1.0 - prob
            positive_prob = remaining * 0.3
            neutral_prob = remaining * 0.7
        else:  # neutral
            neutral_prob = prob
            remaining = 1.0 - prob
            positive_prob = remaining * 0.5
            negative_prob = remaining * 0.5

        # Score formula: P(positive) - P(negative) gives [-1, +1] range
        score = positive_prob - negative_prob

        # Confidence is the max probability
        confidence = max(positive_prob, negative_prob, neutral_prob)

        return SentimentResult(
            text=text,
            score=score,
            confidence=confidence,
            positive_prob=positive_prob,
            negative_prob=negative_prob,
            neutral_prob=neutral_prob,
            model=self.model_name,
            error=None
        )

    def _create_error_result(self, text: str, error_msg: str) -> SentimentResult:
        """Create a SentimentResult for failed analysis."""
        return SentimentResult(
            text=text,
            score=0.0,
            confidence=0.0,
            positive_prob=0.0,
            negative_prob=0.0,
            neutral_prob=0.0,
            model=self.model_name,
            error=error_msg
        )

    def analyze(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of a single text.

        Args:
            text: Text to analyze (headline, summary, or combined).

        Returns:
            SentimentResult with score, confidence, and probabilities.
        """
        # Handle empty or very short text
        if not text or len(text.strip()) < self.MIN_TEXT_LENGTH:
            return self._create_error_result(
                text or "",
                f"Text too short (min {self.MIN_TEXT_LENGTH} characters)"
            )

        try:
            self._ensure_loaded()

            # Run inference
            output = self._pipeline(text.strip())[0]

            return self._parse_finbert_output(output, text)

        except Exception as e:
            return self._create_error_result(text, str(e))

    def analyze_batch(
        self,
        df: pd.DataFrame,
        text_column: str = "headline",
        summary_column: Optional[str] = "summary",
        include_summary: bool = True,
        summary_weight: float = 0.3,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Analyze sentiment for a batch of articles.

        Args:
            df: DataFrame with articles.
            text_column: Column containing main text (headlines).
            summary_column: Column containing summaries (optional).
            include_summary: Whether to include summary in analysis.
            summary_weight: Weight for summary sentiment (0-1).
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

        # Make a copy to avoid modifying original
        result_df = df.copy()

        # Ensure model is loaded
        self._ensure_loaded()

        # Prepare texts for batch processing
        texts = []
        valid_indices = []

        for idx, row in df.iterrows():
            main_text = str(row.get(text_column, "")).strip()

            if len(main_text) >= self.MIN_TEXT_LENGTH:
                texts.append(main_text)
                valid_indices.append(idx)

        if not texts:
            # No valid texts to analyze
            result_df["sentiment_score"] = 0.0
            result_df["sentiment_confidence"] = 0.0
            result_df["prob_positive"] = 0.0
            result_df["prob_negative"] = 0.0
            result_df["prob_neutral"] = 0.0
            result_df["sentiment_signal"] = "neutral"
            return result_df

        # Initialize result columns
        result_df["sentiment_score"] = 0.0
        result_df["sentiment_confidence"] = 0.0
        result_df["prob_positive"] = 0.0
        result_df["prob_negative"] = 0.0
        result_df["prob_neutral"] = 0.0
        result_df["sentiment_signal"] = "neutral"

        # Process in batches with progress
        try:
            from tqdm import tqdm
            iterator = range(0, len(texts), self.batch_size)
            if show_progress:
                iterator = tqdm(iterator, desc="Analyzing sentiment", unit="batch")
        except ImportError:
            iterator = range(0, len(texts), self.batch_size)

        for batch_start in iterator:
            batch_end = min(batch_start + self.batch_size, len(texts))
            batch_texts = texts[batch_start:batch_end]
            batch_indices = valid_indices[batch_start:batch_end]

            try:
                # Run batch inference
                outputs = self._pipeline(batch_texts)

                for i, output in enumerate(outputs):
                    idx = batch_indices[i]
                    result = self._parse_finbert_output(output, batch_texts[i])

                    result_df.at[idx, "sentiment_score"] = result.score
                    result_df.at[idx, "sentiment_confidence"] = result.confidence
                    result_df.at[idx, "prob_positive"] = result.positive_prob
                    result_df.at[idx, "prob_negative"] = result.negative_prob
                    result_df.at[idx, "prob_neutral"] = result.neutral_prob
                    result_df.at[idx, "sentiment_signal"] = result.signal

            except Exception as e:
                # Mark batch as failed
                warnings.warn(f"Batch processing failed: {e}")
                for idx in batch_indices:
                    result_df.at[idx, "sentiment_signal"] = "neutral"

        # Process summaries if requested and column exists
        if include_summary and summary_column and summary_column in df.columns:
            self._add_summary_sentiment(
                result_df, df, summary_column, summary_weight, show_progress
            )

        return result_df

    def _add_summary_sentiment(
        self,
        result_df: pd.DataFrame,
        original_df: pd.DataFrame,
        summary_column: str,
        summary_weight: float,
        show_progress: bool
    ) -> None:
        """Add summary sentiment and combine with headline sentiment."""
        # Get summaries
        summary_texts = []
        summary_indices = []

        for idx, row in original_df.iterrows():
            summary = str(row.get(summary_column, "")).strip()
            if len(summary) >= self.MIN_TEXT_LENGTH:
                summary_texts.append(summary)
                summary_indices.append(idx)

        if not summary_texts:
            return

        # Process summaries
        try:
            from tqdm import tqdm
            iterator = range(0, len(summary_texts), self.batch_size)
            if show_progress:
                iterator = tqdm(iterator, desc="Analyzing summaries", unit="batch")
        except ImportError:
            iterator = range(0, len(summary_texts), self.batch_size)

        headline_weight = 1.0 - summary_weight

        for batch_start in iterator:
            batch_end = min(batch_start + self.batch_size, len(summary_texts))
            batch_texts = summary_texts[batch_start:batch_end]
            batch_indices = summary_indices[batch_start:batch_end]

            try:
                outputs = self._pipeline(batch_texts)

                for i, output in enumerate(outputs):
                    idx = batch_indices[i]
                    result = self._parse_finbert_output(output, batch_texts[i])

                    # Combine headline and summary scores
                    headline_score = result_df.at[idx, "sentiment_score"]
                    combined_score = (
                        headline_weight * headline_score +
                        summary_weight * result.score
                    )

                    # Update with combined score
                    result_df.at[idx, "sentiment_score"] = combined_score

                    # Update signal based on combined score
                    if combined_score > 0.1:
                        result_df.at[idx, "sentiment_signal"] = "bullish"
                    elif combined_score < -0.1:
                        result_df.at[idx, "sentiment_signal"] = "bearish"
                    else:
                        result_df.at[idx, "sentiment_signal"] = "neutral"

            except Exception as e:
                warnings.warn(f"Summary batch processing failed: {e}")

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the model cache to free memory."""
        cls._model_cache.clear()

    def get_device_info(self) -> dict:
        """Get information about the current device configuration."""
        info = {
            "model_name": self.model_name,
            "device": self.device,
            "batch_size": self.batch_size,
            "model_loaded": self._pipeline is not None,
            "cached_models": list(FinBertAnalyzer._model_cache.keys())
        }

        try:
            import torch
            info["torch_version"] = torch.__version__
            info["cuda_available"] = torch.cuda.is_available()
            info["mps_available"] = torch.backends.mps.is_available()
            if torch.cuda.is_available():
                info["cuda_device_name"] = torch.cuda.get_device_name(0)
        except ImportError:
            info["torch_version"] = "not installed"

        return info
