"""
Comprehensive test suite for the sentiment analysis module.

Tests:
- SentimentResult dataclass
- FinBertAnalyzer (with mocking for CI)
- VaderAnalyzer
- SentimentAggregator
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.sentiment import (
    FinBertAnalyzer,
    SentimentResult,
    VaderAnalyzer,
    SentimentAggregator,
    get_analyzer,
)


class TestSentimentResult:
    """Tests for SentimentResult dataclass."""

    def test_bullish_signal(self):
        """Test bullish signal classification."""
        result = SentimentResult(
            text="Great earnings",
            score=0.5,
            confidence=0.8,
            positive_prob=0.7,
            negative_prob=0.1,
            neutral_prob=0.2,
            model="test"
        )
        assert result.signal == "bullish"
        assert result.is_valid is True

    def test_bearish_signal(self):
        """Test bearish signal classification."""
        result = SentimentResult(
            text="Terrible losses",
            score=-0.5,
            confidence=0.8,
            positive_prob=0.1,
            negative_prob=0.7,
            neutral_prob=0.2,
            model="test"
        )
        assert result.signal == "bearish"

    def test_neutral_signal(self):
        """Test neutral signal classification."""
        result = SentimentResult(
            text="Company reports results",
            score=0.05,
            confidence=0.5,
            positive_prob=0.35,
            negative_prob=0.3,
            neutral_prob=0.35,
            model="test"
        )
        assert result.signal == "neutral"

    def test_error_result(self):
        """Test result with error."""
        result = SentimentResult(
            text="",
            score=0.0,
            confidence=0.0,
            positive_prob=0.0,
            negative_prob=0.0,
            neutral_prob=0.0,
            model="test",
            error="Text too short"
        )
        assert result.is_valid is False


class TestVaderAnalyzer:
    """Tests for VaderAnalyzer (lightweight, runs without GPU)."""

    @pytest.fixture
    def analyzer(self):
        """Create a VADER analyzer instance."""
        return VaderAnalyzer(use_financial_lexicon=True)

    def test_positive_sentiment(self, analyzer):
        """Test positive financial text."""
        result = analyzer.analyze("Stock surges on bullish earnings report")
        assert result.score > 0.3
        assert result.signal == "bullish"
        assert result.model == "vader_financial"

    def test_negative_sentiment(self, analyzer):
        """Test negative financial text."""
        result = analyzer.analyze("Company stock plunges after downgrade")
        assert result.score < -0.3
        assert result.signal == "bearish"

    def test_neutral_sentiment(self, analyzer):
        """Test neutral text."""
        result = analyzer.analyze("Company announces quarterly results")
        assert -0.2 < result.score < 0.2

    def test_empty_text(self, analyzer):
        """Test handling of empty text."""
        result = analyzer.analyze("")
        assert result.error is not None
        assert result.score == 0.0

    def test_short_text(self, analyzer):
        """Test handling of very short text."""
        result = analyzer.analyze("Hi")
        assert result.error is not None

    def test_financial_lexicon(self, analyzer):
        """Test that financial lexicon enhances sentiment."""
        # "bullish" is a strongly positive financial term
        result = analyzer.analyze("Analysts are bullish on the stock")
        assert result.score > 0.4

        # "bearish" is a strongly negative financial term
        result = analyzer.analyze("Market sentiment is bearish")
        assert result.score < -0.4

    def test_batch_analyze(self, analyzer):
        """Test batch analysis."""
        df = pd.DataFrame({
            "headline": [
                "Stock surges on great earnings",
                "Company faces major losses",
                "Quarterly report released today"
            ]
        })

        result_df = analyzer.analyze_batch(df, show_progress=False)

        assert "sentiment_score" in result_df.columns
        assert "sentiment_signal" in result_df.columns
        assert len(result_df) == 3

        # First should be bullish
        assert result_df.iloc[0]["sentiment_score"] > 0.2
        # Second should be bearish
        assert result_df.iloc[1]["sentiment_score"] < -0.2

    def test_lexicon_info(self, analyzer):
        """Test lexicon info method."""
        info = analyzer.get_lexicon_info()
        assert info["financial_lexicon_enabled"] is True
        assert info["financial_terms_count"] > 0

    def test_without_financial_lexicon(self):
        """Test analyzer without financial lexicon."""
        analyzer = VaderAnalyzer(use_financial_lexicon=False)
        info = analyzer.get_lexicon_info()
        assert info["financial_lexicon_enabled"] is False


class TestFinBertAnalyzer:
    """Tests for FinBertAnalyzer.

    Uses mocking to avoid requiring GPU/model downloads in CI.
    """

    @pytest.fixture
    def mock_pipeline(self):
        """Create a mock transformers pipeline."""
        mock_pipe = Mock()
        # Default return value for single text
        mock_pipe.return_value = [{"label": "positive", "score": 0.85}]
        return mock_pipe

    @pytest.fixture
    def analyzer_with_mock(self, mock_pipeline):
        """Create analyzer with mocked pipeline."""
        with patch.object(FinBertAnalyzer, '_get_pipeline', return_value=mock_pipeline):
            analyzer = FinBertAnalyzer(device="cpu")
            analyzer._pipeline = mock_pipeline
            return analyzer

    def test_analyze_positive(self, analyzer_with_mock, mock_pipeline):
        """Test positive sentiment analysis."""
        mock_pipeline.return_value = [{"label": "positive", "score": 0.9}]

        result = analyzer_with_mock.analyze("Apple reports record revenue")

        assert result.score > 0.5
        assert result.signal == "bullish"
        assert result.positive_prob > 0.5

    def test_analyze_negative(self, analyzer_with_mock, mock_pipeline):
        """Test negative sentiment analysis."""
        mock_pipeline.return_value = [{"label": "negative", "score": 0.85}]

        result = analyzer_with_mock.analyze("Company announces massive layoffs")

        assert result.score < -0.3
        assert result.signal == "bearish"
        assert result.negative_prob > 0.5

    def test_analyze_neutral(self, analyzer_with_mock, mock_pipeline):
        """Test neutral sentiment analysis."""
        mock_pipeline.return_value = [{"label": "neutral", "score": 0.7}]

        result = analyzer_with_mock.analyze("Company releases quarterly report")

        assert -0.3 < result.score < 0.3

    def test_empty_text(self, analyzer_with_mock):
        """Test handling of empty text."""
        result = analyzer_with_mock.analyze("")
        assert result.error is not None
        assert result.score == 0.0

    def test_short_text(self, analyzer_with_mock):
        """Test handling of short text."""
        result = analyzer_with_mock.analyze("Hi")
        assert result.error is not None

    def test_batch_analyze(self, analyzer_with_mock, mock_pipeline):
        """Test batch analysis."""
        # Configure mock to return results for batch
        mock_pipeline.return_value = [
            {"label": "positive", "score": 0.8},
            {"label": "negative", "score": 0.75},
            {"label": "neutral", "score": 0.6}
        ]

        df = pd.DataFrame({
            "headline": [
                "Record profits announced",
                "Major losses reported",
                "Quarterly results released"
            ],
            "symbol": ["AAPL", "AAPL", "AAPL"]
        })

        result_df = analyzer_with_mock.analyze_batch(
            df, text_column="headline", include_summary=False, show_progress=False
        )

        assert "sentiment_score" in result_df.columns
        assert "sentiment_signal" in result_df.columns
        assert len(result_df) == 3

    def test_device_detection(self):
        """Test device auto-detection."""
        analyzer = FinBertAnalyzer(device=None)
        # Should return one of the valid devices
        assert analyzer.device in ["cuda", "mps", "cpu"]

    def test_explicit_device(self):
        """Test explicit device setting."""
        analyzer = FinBertAnalyzer(device="cpu")
        assert analyzer.device == "cpu"

    def test_clear_cache(self):
        """Test model cache clearing."""
        FinBertAnalyzer._model_cache = {"test_key": Mock()}
        FinBertAnalyzer.clear_cache()
        assert len(FinBertAnalyzer._model_cache) == 0


class TestSentimentAggregator:
    """Tests for SentimentAggregator."""

    @pytest.fixture
    def aggregator(self):
        """Create aggregator instance."""
        return SentimentAggregator(
            decay_halflife_hours=24.0,
            min_articles_for_signal=2,
            confidence_threshold=0.3
        )

    @pytest.fixture
    def sample_sentiment_df(self):
        """Create sample sentiment DataFrame."""
        base_date = datetime(2024, 1, 15)
        data = []

        # AAPL articles
        for i in range(5):
            data.append({
                "symbol": "AAPL",
                "datetime": base_date + timedelta(hours=i * 3),
                "headline": f"AAPL headline {i}",
                "sentiment_score": 0.3 + 0.1 * i,  # 0.3 to 0.7
                "sentiment_confidence": 0.7,
                "sentiment_signal": "bullish" if 0.3 + 0.1 * i > 0.1 else "neutral"
            })

        # MSFT articles - more negative
        for i in range(3):
            data.append({
                "symbol": "MSFT",
                "datetime": base_date + timedelta(hours=i * 4),
                "headline": f"MSFT headline {i}",
                "sentiment_score": -0.2 - 0.1 * i,  # -0.2 to -0.4
                "sentiment_confidence": 0.6,
                "sentiment_signal": "bearish"
            })

        return pd.DataFrame(data)

    def test_aggregate_daily(self, aggregator, sample_sentiment_df):
        """Test daily aggregation."""
        result = aggregator.aggregate_daily(sample_sentiment_df)

        assert "symbol" in result.columns
        assert "date" in result.columns
        assert "sentiment_score" in result.columns
        assert "article_count" in result.columns
        assert "signal_valid" in result.columns

        # Should have one row per symbol per day
        assert len(result) == 2  # AAPL and MSFT, same day

        # AAPL should be positive
        aapl_row = result[result["symbol"] == "AAPL"].iloc[0]
        assert aapl_row["sentiment_score"] > 0
        assert aapl_row["article_count"] == 5
        assert aapl_row["signal_valid"] == True  # >= 2 articles, >= 0.3 confidence

        # MSFT should be negative
        msft_row = result[result["symbol"] == "MSFT"].iloc[0]
        assert msft_row["sentiment_score"] < 0
        assert msft_row["article_count"] == 3

    def test_aggregate_weekly(self, aggregator, sample_sentiment_df):
        """Test weekly aggregation."""
        result = aggregator.aggregate_weekly(sample_sentiment_df)

        assert "week_start" in result.columns
        assert "sentiment_momentum" in result.columns

        # Should have data for both symbols
        assert len(result["symbol"].unique()) == 2

    def test_time_decay_weights(self, aggregator):
        """Test time decay weight calculation."""
        now = datetime.now()
        timestamps = pd.Series([
            now - timedelta(hours=0),   # Most recent
            now - timedelta(hours=24),  # 1 day old (half weight)
            now - timedelta(hours=48),  # 2 days old (quarter weight)
        ])

        weights = aggregator._calculate_time_weights(timestamps, now)

        # Most recent should have highest weight
        assert weights[0] > weights[1] > weights[2]

        # At halflife (24h), weight should be ~0.5
        assert 0.45 < weights[1] < 0.55

    def test_generate_absolute_signals(self, aggregator, sample_sentiment_df):
        """Test absolute signal generation."""
        daily = aggregator.aggregate_daily(sample_sentiment_df)
        result = aggregator.generate_signals(daily, method="absolute")

        assert "signal" in result.columns
        assert "signal_strength" in result.columns
        assert "normalized_score" in result.columns

    def test_generate_zscore_signals(self, aggregator, sample_sentiment_df):
        """Test z-score signal generation."""
        daily = aggregator.aggregate_daily(sample_sentiment_df)
        result = aggregator.generate_signals(daily, method="zscore")

        assert "signal" in result.columns
        assert "normalized_score" in result.columns

        # Z-scores should be normalized
        # With only 2 data points, one should be positive and one negative
        scores = result["normalized_score"].values
        if len(scores) > 1:
            assert not np.isnan(scores).all()

    def test_generate_rank_signals(self, aggregator, sample_sentiment_df):
        """Test rank-based signal generation."""
        daily = aggregator.aggregate_daily(sample_sentiment_df)
        result = aggregator.generate_signals(daily, method="rank")

        assert "signal" in result.columns
        assert "normalized_score" in result.columns

        # Ranks should be between 0 and 1
        assert result["normalized_score"].between(0, 1).all()

    def test_invalid_method(self, aggregator, sample_sentiment_df):
        """Test invalid signal method raises error."""
        daily = aggregator.aggregate_daily(sample_sentiment_df)

        with pytest.raises(ValueError):
            aggregator.generate_signals(daily, method="invalid")

    def test_calculate_sentiment_momentum(self, aggregator):
        """Test sentiment momentum calculation."""
        # Create multi-day data
        dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
        df = pd.DataFrame({
            "symbol": ["AAPL"] * 10,
            "date": dates,
            "sentiment_score": [0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
        })

        result = aggregator.calculate_sentiment_momentum(df, lookback_days=3)

        assert "sentiment_momentum" in result.columns
        assert "sentiment_acceleration" in result.columns

        # First 3 rows should have NaN momentum (not enough lookback)
        assert pd.isna(result["sentiment_momentum"].iloc[0])
        assert pd.isna(result["sentiment_momentum"].iloc[2])

        # 4th row should have momentum
        assert not pd.isna(result["sentiment_momentum"].iloc[3])

    def test_empty_dataframe(self, aggregator):
        """Test handling of empty DataFrames."""
        empty_df = pd.DataFrame()

        daily = aggregator.aggregate_daily(empty_df)
        assert len(daily) == 0

        weekly = aggregator.aggregate_weekly(empty_df)
        assert len(weekly) == 0

    def test_signal_validity_thresholds(self):
        """Test signal validity with different thresholds."""
        # Strict aggregator
        strict_agg = SentimentAggregator(
            min_articles_for_signal=5,
            confidence_threshold=0.8
        )

        df = pd.DataFrame({
            "symbol": ["AAPL"] * 3,
            "datetime": pd.date_range(start="2024-01-01", periods=3, freq="h"),
            "sentiment_score": [0.5, 0.6, 0.7],
            "sentiment_confidence": [0.5, 0.5, 0.5],
            "sentiment_signal": ["bullish"] * 3
        })

        result = strict_agg.aggregate_daily(df)

        # Should be invalid: only 3 articles (need 5), confidence 0.5 (need 0.8)
        assert result.iloc[0]["signal_valid"] == False


class TestGetAnalyzer:
    """Tests for get_analyzer factory function."""

    def test_get_vader_analyzer(self):
        """Test getting VADER analyzer."""
        analyzer = get_analyzer(use_finbert=False)
        assert isinstance(analyzer, VaderAnalyzer)

    def test_fallback_to_vader(self):
        """Test fallback to VADER when use_finbert=False."""
        # When use_finbert=False, should always return VaderAnalyzer
        analyzer = get_analyzer(use_finbert=False)
        assert isinstance(analyzer, VaderAnalyzer)

    def test_finbert_request(self):
        """Test that get_analyzer with use_finbert=True returns FinBertAnalyzer when available."""
        # Since transformers is installed, this should return FinBertAnalyzer
        analyzer = get_analyzer(use_finbert=True)
        assert isinstance(analyzer, FinBertAnalyzer)


class TestIntegration:
    """Integration tests using real (but fast) operations."""

    def test_vader_full_pipeline(self):
        """Test full pipeline with VADER (no GPU needed)."""
        # Create sample news data
        df = pd.DataFrame({
            "symbol": ["AAPL"] * 5 + ["MSFT"] * 5,
            "datetime": pd.date_range(start="2024-01-15 09:00", periods=10, freq="1h"),
            "headline": [
                "Apple stock surges on strong iPhone sales",
                "Bullish outlook for Apple in 2024",
                "Apple reports record quarterly profits",
                "Apple announces new product launch",
                "Analysts upgrade Apple stock",
                "Microsoft cloud revenue disappoints",
                "Microsoft faces antitrust concerns",
                "Microsoft stock drops on weak guidance",
                "Microsoft announces layoffs",
                "Microsoft reports losses",
            ]
        })

        # Analyze with VADER
        analyzer = VaderAnalyzer()
        sentiment_df = analyzer.analyze_batch(df, show_progress=False)

        assert "sentiment_score" in sentiment_df.columns
        assert len(sentiment_df) == 10

        # Aggregate
        aggregator = SentimentAggregator(min_articles_for_signal=3)
        daily = aggregator.aggregate_daily(sentiment_df)

        # Should have 2 symbols
        assert len(daily["symbol"].unique()) == 2

        # Generate signals
        signals = aggregator.generate_signals(daily, method="absolute")

        assert "signal" in signals.columns
        assert all(s in ["bullish", "bearish", "neutral"] for s in signals["signal"])

    def test_aggregator_with_real_time_decay(self):
        """Test that time decay properly weights recent articles."""
        aggregator = SentimentAggregator(decay_halflife_hours=6)

        # Create same-day data with clear time separation
        base_date = datetime(2024, 1, 15, 8, 0, 0)  # 8am
        df = pd.DataFrame({
            "symbol": ["AAPL"] * 4,
            "datetime": [
                base_date,                        # Old (8am), low weight
                base_date + timedelta(hours=4),   # 12pm
                base_date + timedelta(hours=8),   # 4pm
                base_date + timedelta(hours=12),  # 8pm, most recent, highest weight
            ],
            "sentiment_score": [-0.8, -0.5, 0.5, 0.8],  # Old=negative, new=positive
            "sentiment_confidence": [0.7] * 4,
            "sentiment_signal": ["bearish", "bearish", "bullish", "bullish"]
        })

        daily = aggregator.aggregate_daily(df)

        # All data is on same day, so we should have 1 row
        assert len(daily) == 1

        # Result should be weighted toward recent positive sentiment
        # With 6h halflife, 8am has low weight, 8pm has high weight
        assert daily.iloc[0]["sentiment_score"] > 0


# Markers for slow tests that require model downloads
@pytest.mark.slow
class TestFinBertReal:
    """Real FinBERT tests (requires model download, GPU recommended).

    Run with: pytest -m slow
    """

    @pytest.fixture
    def real_analyzer(self):
        """Create real FinBERT analyzer."""
        try:
            return FinBertAnalyzer(device="cpu")
        except ImportError:
            pytest.skip("transformers not installed")

    def test_real_finbert_analysis(self, real_analyzer):
        """Test real FinBERT analysis."""
        result = real_analyzer.analyze("Apple reports record quarterly revenue beating expectations")

        assert result.is_valid
        assert result.score > 0  # Should be positive
        assert result.confidence > 0.3

    def test_real_finbert_batch(self, real_analyzer):
        """Test real batch analysis."""
        df = pd.DataFrame({
            "headline": [
                "Company announces record profits",
                "Stock plunges on earnings miss",
                "Quarterly results meet expectations"
            ]
        })

        result = real_analyzer.analyze_batch(df, include_summary=False, show_progress=False)

        assert len(result) == 3
        assert result.iloc[0]["sentiment_score"] > result.iloc[1]["sentiment_score"]
