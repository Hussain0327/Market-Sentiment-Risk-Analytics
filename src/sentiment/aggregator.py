"""
Sentiment aggregation and signal generation.

Aggregates article-level sentiment to daily/weekly signals
with time-weighted averaging and cross-sectional ranking.
"""

from typing import Literal, Optional
from datetime import datetime, timedelta

import pandas as pd
import numpy as np


class SentimentAggregator:
    """
    Aggregates sentiment scores and generates trading signals.

    Features:
    - Time-weighted averaging with exponential decay
    - Daily and weekly aggregation
    - Multiple signal generation methods (absolute, zscore, rank)
    - Quality thresholds for signal validity

    Example:
        >>> agg = SentimentAggregator()
        >>> daily = agg.aggregate_daily(sentiment_df)
        >>> signals = agg.generate_signals(daily, method="zscore")
    """

    DEFAULT_DECAY_HALFLIFE_HOURS = 24.0
    DEFAULT_MIN_ARTICLES = 3
    DEFAULT_CONFIDENCE_THRESHOLD = 0.3

    def __init__(
        self,
        decay_halflife_hours: float = DEFAULT_DECAY_HALFLIFE_HOURS,
        min_articles_for_signal: int = DEFAULT_MIN_ARTICLES,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    ):
        """
        Initialize the aggregator.

        Args:
            decay_halflife_hours: Half-life for time decay in hours. Default: 24h
            min_articles_for_signal: Minimum articles for a valid signal.
            confidence_threshold: Minimum average confidence for valid signal.
        """
        self.decay_halflife_hours = decay_halflife_hours
        self.min_articles_for_signal = min_articles_for_signal
        self.confidence_threshold = confidence_threshold

        # Decay constant: lambda = ln(2) / halflife
        self._decay_lambda = np.log(2) / decay_halflife_hours

    def _calculate_time_weights(
        self,
        timestamps: pd.Series,
        reference_time: Optional[datetime] = None
    ) -> np.ndarray:
        """
        Calculate time-decay weights for a series of timestamps.

        Args:
            timestamps: Series of datetime values.
            reference_time: Reference time for decay calculation. Defaults to max timestamp.

        Returns:
            Array of weights in [0, 1] range.
        """
        if timestamps.empty:
            return np.array([])

        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(timestamps):
            timestamps = pd.to_datetime(timestamps)

        # Use max timestamp as reference if not provided
        if reference_time is None:
            reference_time = timestamps.max()

        # Calculate hours since reference
        time_diff = (reference_time - timestamps).dt.total_seconds() / 3600

        # Exponential decay: w = exp(-lambda * t)
        weights = np.exp(-self._decay_lambda * time_diff.values)

        return weights

    def aggregate_daily(
        self,
        df: pd.DataFrame,
        symbol_column: str = "symbol",
        datetime_column: str = "datetime",
        score_column: str = "sentiment_score",
        confidence_column: str = "sentiment_confidence"
    ) -> pd.DataFrame:
        """
        Aggregate sentiment to daily level with time-weighted averaging.

        Args:
            df: DataFrame with article-level sentiment.
            symbol_column: Column with stock symbols.
            datetime_column: Column with article timestamps.
            score_column: Column with sentiment scores.
            confidence_column: Column with confidence scores.

        Returns:
            DataFrame with columns:
            - symbol: Stock symbol
            - date: Date
            - sentiment_score: Time-weighted average sentiment
            - sentiment_confidence: Average confidence
            - article_count: Number of articles (buzz metric)
            - bullish_ratio: Fraction of bullish articles
            - bearish_ratio: Fraction of bearish articles
            - sentiment_std: Standard deviation (disagreement metric)
            - signal_valid: Whether signal meets quality thresholds
        """
        if df.empty:
            return pd.DataFrame(columns=[
                "symbol", "date", "sentiment_score", "sentiment_confidence",
                "article_count", "bullish_ratio", "bearish_ratio",
                "sentiment_std", "signal_valid"
            ])

        # Make a copy and ensure datetime is parsed
        work_df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(work_df[datetime_column]):
            work_df[datetime_column] = pd.to_datetime(work_df[datetime_column])

        # Extract date
        work_df["_date"] = work_df[datetime_column].dt.date

        results = []

        # Group by symbol and date
        for (symbol, date), group in work_df.groupby([symbol_column, "_date"]):
            # Calculate time weights (within the day)
            day_end = datetime.combine(date, datetime.max.time())
            weights = self._calculate_time_weights(group[datetime_column], day_end)

            # Weighted average sentiment
            scores = group[score_column].values
            if len(weights) > 0 and weights.sum() > 0:
                weighted_score = np.average(scores, weights=weights)
            else:
                weighted_score = scores.mean() if len(scores) > 0 else 0.0

            # Average confidence
            confidence = group[confidence_column].mean() if confidence_column in group.columns else 0.5

            # Article count (buzz)
            article_count = len(group)

            # Bullish/bearish ratios
            if "sentiment_signal" in group.columns:
                bullish_ratio = (group["sentiment_signal"] == "bullish").mean()
                bearish_ratio = (group["sentiment_signal"] == "bearish").mean()
            else:
                bullish_ratio = (scores > 0.1).mean()
                bearish_ratio = (scores < -0.1).mean()

            # Sentiment standard deviation (disagreement)
            sentiment_std = scores.std() if len(scores) > 1 else 0.0

            # Signal validity check
            signal_valid = (
                article_count >= self.min_articles_for_signal and
                confidence >= self.confidence_threshold
            )

            results.append({
                "symbol": symbol,
                "date": pd.Timestamp(date),
                "sentiment_score": weighted_score,
                "sentiment_confidence": confidence,
                "article_count": article_count,
                "bullish_ratio": bullish_ratio,
                "bearish_ratio": bearish_ratio,
                "sentiment_std": sentiment_std,
                "signal_valid": signal_valid
            })

        result_df = pd.DataFrame(results)

        # Sort by symbol and date
        if not result_df.empty:
            result_df = result_df.sort_values(["symbol", "date"]).reset_index(drop=True)

        return result_df

    def aggregate_weekly(
        self,
        df: pd.DataFrame,
        symbol_column: str = "symbol",
        datetime_column: str = "datetime",
        score_column: str = "sentiment_score",
        confidence_column: str = "sentiment_confidence"
    ) -> pd.DataFrame:
        """
        Aggregate sentiment to weekly level with momentum calculation.

        Args:
            df: DataFrame with article-level sentiment.
            symbol_column: Column with stock symbols.
            datetime_column: Column with article timestamps.
            score_column: Column with sentiment scores.
            confidence_column: Column with confidence scores.

        Returns:
            DataFrame with daily columns plus:
            - week_start: Start of the week (Monday)
            - sentiment_momentum: Change from previous week
        """
        if df.empty:
            return pd.DataFrame(columns=[
                "symbol", "week_start", "sentiment_score", "sentiment_confidence",
                "article_count", "bullish_ratio", "bearish_ratio",
                "sentiment_std", "signal_valid", "sentiment_momentum"
            ])

        # Make a copy and ensure datetime is parsed
        work_df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(work_df[datetime_column]):
            work_df[datetime_column] = pd.to_datetime(work_df[datetime_column])

        # Get week start (Monday)
        work_df["_week_start"] = work_df[datetime_column].dt.to_period("W-SUN").dt.start_time

        results = []

        # Group by symbol and week
        for (symbol, week_start), group in work_df.groupby([symbol_column, "_week_start"]):
            # Calculate time weights (within the week)
            week_end = week_start + timedelta(days=7)
            weights = self._calculate_time_weights(group[datetime_column], week_end)

            # Weighted average sentiment
            scores = group[score_column].values
            if len(weights) > 0 and weights.sum() > 0:
                weighted_score = np.average(scores, weights=weights)
            else:
                weighted_score = scores.mean() if len(scores) > 0 else 0.0

            # Average confidence
            confidence = group[confidence_column].mean() if confidence_column in group.columns else 0.5

            # Article count
            article_count = len(group)

            # Bullish/bearish ratios
            if "sentiment_signal" in group.columns:
                bullish_ratio = (group["sentiment_signal"] == "bullish").mean()
                bearish_ratio = (group["sentiment_signal"] == "bearish").mean()
            else:
                bullish_ratio = (scores > 0.1).mean()
                bearish_ratio = (scores < -0.1).mean()

            # Sentiment standard deviation
            sentiment_std = scores.std() if len(scores) > 1 else 0.0

            # Signal validity
            signal_valid = (
                article_count >= self.min_articles_for_signal and
                confidence >= self.confidence_threshold
            )

            results.append({
                "symbol": symbol,
                "week_start": week_start,
                "sentiment_score": weighted_score,
                "sentiment_confidence": confidence,
                "article_count": article_count,
                "bullish_ratio": bullish_ratio,
                "bearish_ratio": bearish_ratio,
                "sentiment_std": sentiment_std,
                "signal_valid": signal_valid
            })

        result_df = pd.DataFrame(results)

        if result_df.empty:
            result_df["sentiment_momentum"] = []
            return result_df

        # Sort and calculate momentum
        result_df = result_df.sort_values(["symbol", "week_start"]).reset_index(drop=True)

        # Momentum = current week - previous week
        result_df["sentiment_momentum"] = result_df.groupby("symbol")["sentiment_score"].diff()

        return result_df

    def generate_signals(
        self,
        df: pd.DataFrame,
        method: Literal["absolute", "zscore", "rank"] = "zscore",
        score_column: str = "sentiment_score",
        symbol_column: str = "symbol",
        date_column: str = "date",
        bullish_threshold: float = 0.5,
        bearish_threshold: float = -0.5
    ) -> pd.DataFrame:
        """
        Generate trading signals from aggregated sentiment.

        Args:
            df: DataFrame with aggregated sentiment (from aggregate_daily/weekly).
            method: Signal generation method:
                - "absolute": Use raw sentiment score thresholds
                - "zscore": Cross-sectional z-score normalization
                - "rank": Cross-sectional percentile ranking
            score_column: Column with sentiment scores.
            symbol_column: Column with stock symbols.
            date_column: Column with dates.
            bullish_threshold: Threshold for bullish signal (for absolute method).
            bearish_threshold: Threshold for bearish signal (for absolute method).

        Returns:
            DataFrame with added columns:
            - signal: 'bullish', 'bearish', or 'neutral'
            - signal_strength: Magnitude of signal [0, 1]
            - normalized_score: Score after method-specific normalization
        """
        if df.empty:
            result = df.copy()
            result["signal"] = []
            result["signal_strength"] = []
            result["normalized_score"] = []
            return result

        result_df = df.copy()

        if method == "absolute":
            result_df = self._generate_absolute_signals(
                result_df, score_column, bullish_threshold, bearish_threshold
            )
        elif method == "zscore":
            result_df = self._generate_zscore_signals(
                result_df, score_column, date_column
            )
        elif method == "rank":
            result_df = self._generate_rank_signals(
                result_df, score_column, date_column
            )
        else:
            raise ValueError(f"Unknown method: {method}. Use 'absolute', 'zscore', or 'rank'")

        return result_df

    def _generate_absolute_signals(
        self,
        df: pd.DataFrame,
        score_column: str,
        bullish_threshold: float,
        bearish_threshold: float
    ) -> pd.DataFrame:
        """Generate signals using absolute thresholds."""
        df = df.copy()
        scores = df[score_column]

        # Normalized score is just the raw score clipped to [-1, 1]
        df["normalized_score"] = scores.clip(-1, 1)

        # Generate signals
        df["signal"] = "neutral"
        df.loc[scores > bullish_threshold, "signal"] = "bullish"
        df.loc[scores < bearish_threshold, "signal"] = "bearish"

        # Signal strength is absolute score normalized to [0, 1]
        df["signal_strength"] = np.abs(scores).clip(0, 1)

        return df

    def _generate_zscore_signals(
        self,
        df: pd.DataFrame,
        score_column: str,
        date_column: str
    ) -> pd.DataFrame:
        """Generate signals using cross-sectional z-scores."""
        df = df.copy()

        # Calculate z-score within each date
        df["normalized_score"] = df.groupby(date_column)[score_column].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )

        # Signals based on z-score
        df["signal"] = "neutral"
        df.loc[df["normalized_score"] > 1.0, "signal"] = "bullish"
        df.loc[df["normalized_score"] < -1.0, "signal"] = "bearish"

        # Signal strength from z-score magnitude (capped at 3 std)
        df["signal_strength"] = (np.abs(df["normalized_score"]) / 3).clip(0, 1)

        return df

    def _generate_rank_signals(
        self,
        df: pd.DataFrame,
        score_column: str,
        date_column: str
    ) -> pd.DataFrame:
        """Generate signals using cross-sectional percentile ranks."""
        df = df.copy()

        # Calculate percentile rank within each date (0 to 1)
        df["normalized_score"] = df.groupby(date_column)[score_column].transform(
            lambda x: x.rank(pct=True) if len(x) > 1 else 0.5
        )

        # Signals based on percentile
        df["signal"] = "neutral"
        df.loc[df["normalized_score"] > 0.75, "signal"] = "bullish"
        df.loc[df["normalized_score"] < 0.25, "signal"] = "bearish"

        # Signal strength from distance to median
        df["signal_strength"] = np.abs(df["normalized_score"] - 0.5) * 2

        return df

    def calculate_sentiment_momentum(
        self,
        df: pd.DataFrame,
        lookback_days: int = 5,
        score_column: str = "sentiment_score",
        symbol_column: str = "symbol",
        date_column: str = "date"
    ) -> pd.DataFrame:
        """
        Calculate sentiment momentum (rate of change).

        Args:
            df: DataFrame with daily aggregated sentiment.
            lookback_days: Number of days for momentum calculation.
            score_column: Column with sentiment scores.
            symbol_column: Column with stock symbols.
            date_column: Column with dates.

        Returns:
            DataFrame with added columns:
            - sentiment_momentum: Change over lookback period
            - sentiment_acceleration: Change in momentum (2nd derivative)
        """
        if df.empty:
            result = df.copy()
            result["sentiment_momentum"] = []
            result["sentiment_acceleration"] = []
            return result

        result_df = df.copy()

        # Ensure sorted
        result_df = result_df.sort_values([symbol_column, date_column])

        # Calculate momentum per symbol
        result_df["sentiment_momentum"] = result_df.groupby(symbol_column)[score_column].transform(
            lambda x: x.diff(lookback_days)
        )

        # Calculate acceleration (change in momentum)
        result_df["sentiment_acceleration"] = result_df.groupby(symbol_column)["sentiment_momentum"].transform(
            lambda x: x.diff()
        )

        return result_df

    def merge_with_prices(
        self,
        sentiment_df: pd.DataFrame,
        price_df: pd.DataFrame,
        symbol_column: str = "symbol",
        date_column: str = "date",
        price_date_column: str = "date"
    ) -> pd.DataFrame:
        """
        Merge sentiment data with price data for backtesting.

        Args:
            sentiment_df: DataFrame with aggregated sentiment.
            price_df: DataFrame with price data.
            symbol_column: Column with stock symbols.
            date_column: Date column in sentiment data.
            price_date_column: Date column in price data.

        Returns:
            Merged DataFrame with sentiment and price data.
        """
        if sentiment_df.empty or price_df.empty:
            return sentiment_df.copy()

        # Prepare merge
        sent_df = sentiment_df.copy()
        px_df = price_df.copy()

        # Ensure date columns are the same type
        sent_df[date_column] = pd.to_datetime(sent_df[date_column]).dt.date
        px_df[price_date_column] = pd.to_datetime(px_df[price_date_column]).dt.date

        # Merge
        merged = sent_df.merge(
            px_df,
            left_on=[symbol_column, date_column],
            right_on=[symbol_column, price_date_column],
            how="left"
        )

        return merged
