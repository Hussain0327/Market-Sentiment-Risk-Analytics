"""
Sentiment Feature Builder.

Provides sentiment-based features for ML models:
- Lagged sentiment scores
- Sentiment momentum and changes
- Sentiment disagreement (uncertainty proxy)
- News volume/buzz indicators
- Rolling z-scores for relative sentiment
"""

from typing import Optional, List

import numpy as np
import pandas as pd


class SentimentFeatureBuilder:
    """
    Builder for sentiment-derived features.

    Features are designed to capture different aspects of market sentiment:
    - Level: Current sentiment score
    - Momentum: Changes in sentiment over time
    - Uncertainty: Disagreement among articles
    - Attention: News volume as a buzz indicator

    All methods avoid lookahead bias - features at time t
    only use data from time t and earlier.

    Example:
        >>> builder = SentimentFeatureBuilder()
        >>> sentiment = pd.read_csv('daily_sentiment.csv')
        >>> features = builder.build_all(sentiment, symbol='AAPL')
    """

    DEFAULT_LAGS = [1, 2, 3, 5, 7]
    DEFAULT_MOMENTUM_WINDOWS = [3, 5, 7]
    DEFAULT_ZSCORE_WINDOW = 21

    def __init__(
        self,
        lags: Optional[List[int]] = None,
        momentum_windows: Optional[List[int]] = None,
        zscore_window: int = DEFAULT_ZSCORE_WINDOW
    ):
        """
        Initialize the sentiment feature builder.

        Args:
            lags: Lag periods for sentiment scores. Default: [1, 2, 3, 5, 7]
            momentum_windows: Windows for sentiment momentum. Default: [3, 5, 7]
            zscore_window: Window for rolling z-score. Default: 21
        """
        self.lags = lags or self.DEFAULT_LAGS
        self.momentum_windows = momentum_windows or self.DEFAULT_MOMENTUM_WINDOWS
        self.zscore_window = zscore_window

    def _validate_sentiment(
        self,
        sentiment: pd.DataFrame,
        symbol: Optional[str] = None
    ) -> pd.DataFrame:
        """Validate and filter sentiment data."""
        if 'sentiment_score' not in sentiment.columns:
            raise ValueError("Sentiment data must have 'sentiment_score' column")

        df = sentiment.copy()

        # Filter by symbol if provided
        if symbol and 'symbol' in df.columns:
            df = df[df['symbol'] == symbol].copy()

        # Ensure date is datetime and sorted
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            df = df.set_index('date')

        return df

    def sentiment_lags(
        self,
        sentiment: pd.DataFrame,
        symbol: Optional[str] = None,
        lags: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Create lagged sentiment score features.

        Lagged features capture the persistence of sentiment and
        allow models to learn from past sentiment patterns.

        Args:
            sentiment: DataFrame with 'sentiment_score' column.
            symbol: Symbol to filter for (if multi-symbol data).
            lags: Lag periods. Default: [1, 2, 3, 5, 7]

        Returns:
            DataFrame with lagged sentiment columns.

        Example:
            >>> lags_df = builder.sentiment_lags(sentiment, symbol='AAPL')
            >>> print(lags_df.columns.tolist())
            ['sentiment_lag_1', 'sentiment_lag_2', ...]
        """
        df = self._validate_sentiment(sentiment, symbol)
        lags = lags or self.lags

        result = pd.DataFrame(index=df.index)

        for lag in lags:
            col_name = f'sentiment_lag_{lag}'
            result[col_name] = df['sentiment_score'].shift(lag)

        return result

    def sentiment_momentum(
        self,
        sentiment: pd.DataFrame,
        symbol: Optional[str] = None,
        windows: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Calculate sentiment momentum features.

        Momentum captures the change in sentiment over different
        time horizons, indicating improving or deteriorating sentiment.

        Args:
            sentiment: DataFrame with 'sentiment_score' column.
            symbol: Symbol to filter for.
            windows: Window sizes for momentum. Default: [3, 5, 7]

        Returns:
            DataFrame with momentum features:
            - sentiment_change_N: N-day change in sentiment
            - sentiment_pct_change_N: N-day percentage change

        Example:
            >>> mom_df = builder.sentiment_momentum(sentiment)
            >>> if mom_df['sentiment_change_3'].iloc[-1] > 0:
            ...     print("Sentiment improving")
        """
        df = self._validate_sentiment(sentiment, symbol)
        windows = windows or self.momentum_windows

        result = pd.DataFrame(index=df.index)
        score = df['sentiment_score']

        for window in windows:
            # Absolute change
            result[f'sentiment_change_{window}'] = score.diff(periods=window)

            # Relative change (handle division by zero)
            shifted = score.shift(window)
            pct_change = (score - shifted) / (shifted.abs() + 0.001)
            result[f'sentiment_pct_change_{window}'] = pct_change

        return result

    def sentiment_ma(
        self,
        sentiment: pd.DataFrame,
        symbol: Optional[str] = None,
        windows: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Calculate sentiment moving averages.

        Moving averages smooth noisy daily sentiment and reveal trends.

        Args:
            sentiment: DataFrame with 'sentiment_score' column.
            symbol: Symbol to filter for.
            windows: Window sizes for moving averages.

        Returns:
            DataFrame with MA features:
            - sentiment_sma_N: N-day simple moving average
            - sentiment_ema_N: N-day exponential moving average
        """
        df = self._validate_sentiment(sentiment, symbol)
        windows = windows or self.momentum_windows

        result = pd.DataFrame(index=df.index)
        score = df['sentiment_score']

        for window in windows:
            result[f'sentiment_sma_{window}'] = score.rolling(window=window).mean()
            result[f'sentiment_ema_{window}'] = score.ewm(span=window, adjust=False).mean()

        return result

    def sentiment_disagreement(
        self,
        sentiment: pd.DataFrame,
        symbol: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Calculate sentiment disagreement features.

        Disagreement (measured by sentiment standard deviation)
        indicates uncertainty - when analysts disagree, the market
        is less certain about the direction.

        Args:
            sentiment: DataFrame with sentiment columns.
            symbol: Symbol to filter for.

        Returns:
            DataFrame with disagreement features:
            - sentiment_std: Standard deviation of article scores
            - sentiment_range: Range from bullish to bearish ratio
            - disagreement_high: 1 if disagreement above median

        Note:
            Requires 'sentiment_std', 'bullish_ratio', 'bearish_ratio' columns
            from daily_sentiment.csv
        """
        df = self._validate_sentiment(sentiment, symbol)

        result = pd.DataFrame(index=df.index)

        # Use pre-computed standard deviation if available
        if 'sentiment_std' in df.columns:
            result['sentiment_std'] = df['sentiment_std']

            # High disagreement indicator
            median_std = df['sentiment_std'].expanding().median()
            result['disagreement_high'] = (df['sentiment_std'] > median_std).astype(int)

        # Calculate sentiment range from bullish/bearish ratios
        if 'bullish_ratio' in df.columns and 'bearish_ratio' in df.columns:
            result['sentiment_range'] = df['bullish_ratio'] - df['bearish_ratio']
            result['sentiment_spread'] = df['bullish_ratio'] + df['bearish_ratio']

        return result

    def sentiment_zscore(
        self,
        sentiment: pd.DataFrame,
        symbol: Optional[str] = None,
        window: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Calculate rolling z-score for relative sentiment.

        Z-score normalizes sentiment relative to recent history,
        making it easier to identify unusually positive or negative sentiment.

        Args:
            sentiment: DataFrame with 'sentiment_score' column.
            symbol: Symbol to filter for.
            window: Window for rolling statistics. Default: 21

        Returns:
            DataFrame with z-score features:
            - sentiment_zscore: Rolling z-score
            - sentiment_percentile: Rolling percentile rank

        Example:
            >>> zscore_df = builder.sentiment_zscore(sentiment)
            >>> if abs(zscore_df['sentiment_zscore'].iloc[-1]) > 2:
            ...     print("Extreme sentiment")
        """
        df = self._validate_sentiment(sentiment, symbol)
        window = window or self.zscore_window

        result = pd.DataFrame(index=df.index)
        score = df['sentiment_score']

        # Rolling z-score
        rolling_mean = score.rolling(window=window).mean()
        rolling_std = score.rolling(window=window).std()
        result['sentiment_zscore'] = (score - rolling_mean) / (rolling_std + 1e-6)

        # Rolling percentile rank
        def percentile_rank(x):
            if len(x) < 2:
                return 0.5
            current = x.iloc[-1]
            return (x < current).sum() / len(x)

        result['sentiment_percentile'] = score.rolling(window=window).apply(
            percentile_rank, raw=False
        )

        return result

    def article_count_features(
        self,
        sentiment: pd.DataFrame,
        symbol: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Calculate article count / news volume features.

        News volume can indicate market attention and potential
        price movement - high attention often precedes volatility.

        Args:
            sentiment: DataFrame with 'article_count' column.
            symbol: Symbol to filter for.

        Returns:
            DataFrame with volume features:
            - article_count: Daily article count
            - article_count_sma_5: 5-day average article count
            - article_count_zscore: Z-score of article count
            - high_attention: 1 if above average attention

        Example:
            >>> count_df = builder.article_count_features(sentiment)
            >>> if count_df['high_attention'].iloc[-1]:
            ...     print("High news attention")
        """
        df = self._validate_sentiment(sentiment, symbol)

        result = pd.DataFrame(index=df.index)

        if 'article_count' not in df.columns:
            return result

        count = df['article_count']

        result['article_count'] = count
        result['article_count_sma_5'] = count.rolling(window=5).mean()
        result['article_count_sma_21'] = count.rolling(window=21).mean()

        # Z-score of article count
        rolling_mean = count.rolling(window=21).mean()
        rolling_std = count.rolling(window=21).std()
        result['article_count_zscore'] = (count - rolling_mean) / (rolling_std + 1e-6)

        # High attention indicator
        result['high_attention'] = (count > result['article_count_sma_21']).astype(int)

        # Article count momentum
        result['article_count_change'] = count.diff()
        result['article_count_pct_change'] = count.pct_change()

        return result

    def confidence_features(
        self,
        sentiment: pd.DataFrame,
        symbol: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Calculate confidence-based features.

        Model confidence can indicate signal quality - high confidence
        predictions may be more reliable.

        Args:
            sentiment: DataFrame with 'sentiment_confidence' column.
            symbol: Symbol to filter for.

        Returns:
            DataFrame with confidence features.
        """
        df = self._validate_sentiment(sentiment, symbol)

        result = pd.DataFrame(index=df.index)

        if 'sentiment_confidence' not in df.columns:
            return result

        conf = df['sentiment_confidence']

        result['sentiment_confidence'] = conf
        result['confidence_sma_5'] = conf.rolling(window=5).mean()

        # Weighted sentiment (score * confidence)
        if 'sentiment_score' in df.columns:
            result['weighted_sentiment'] = df['sentiment_score'] * conf

        # High confidence indicator
        median_conf = conf.expanding().median()
        result['high_confidence'] = (conf > median_conf).astype(int)

        return result

    def signal_features(
        self,
        sentiment: pd.DataFrame,
        symbol: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Calculate trading signal features.

        Args:
            sentiment: DataFrame with sentiment columns.
            symbol: Symbol to filter for.

        Returns:
            DataFrame with signal features:
            - signal_valid: Whether signal passes quality threshold
            - bullish_signal: 1 if bullish sentiment
            - bearish_signal: 1 if bearish sentiment
        """
        df = self._validate_sentiment(sentiment, symbol)

        result = pd.DataFrame(index=df.index)

        if 'signal_valid' in df.columns:
            result['signal_valid'] = df['signal_valid'].astype(int)

        if 'sentiment_score' in df.columns:
            score = df['sentiment_score']
            # Bullish: sentiment > 0.1
            result['bullish_signal'] = (score > 0.1).astype(int)
            # Bearish: sentiment < -0.1
            result['bearish_signal'] = (score < -0.1).astype(int)
            # Neutral: between -0.1 and 0.1
            result['neutral_signal'] = ((score >= -0.1) & (score <= 0.1)).astype(int)

        return result

    def interaction_features(
        self,
        sentiment: pd.DataFrame,
        symbol: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Calculate interaction features combining multiple signals.

        Args:
            sentiment: DataFrame with sentiment columns.
            symbol: Symbol to filter for.

        Returns:
            DataFrame with interaction features.
        """
        df = self._validate_sentiment(sentiment, symbol)

        result = pd.DataFrame(index=df.index)

        if 'sentiment_score' not in df.columns:
            return result

        score = df['sentiment_score']

        # Sentiment x confidence interaction
        if 'sentiment_confidence' in df.columns:
            result['sent_x_conf'] = score * df['sentiment_confidence']

        # Sentiment x volume interaction
        if 'article_count' in df.columns:
            count_norm = df['article_count'] / (df['article_count'].rolling(21).mean() + 1)
            result['sent_x_attention'] = score * count_norm

        # Sentiment acceleration (2nd derivative)
        result['sentiment_acceleration'] = score.diff().diff()

        return result

    def build_all(
        self,
        sentiment: pd.DataFrame,
        symbol: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Build all sentiment features.

        Args:
            sentiment: DataFrame with sentiment columns.
            symbol: Symbol to filter for.

        Returns:
            DataFrame with all sentiment features.

        Example:
            >>> sentiment = pd.read_csv('daily_sentiment.csv')
            >>> features = builder.build_all(sentiment, symbol='AAPL')
            >>> print(f"Built {len(features.columns)} features")
        """
        df = self._validate_sentiment(sentiment, symbol)
        features = pd.DataFrame(index=df.index)

        # Current sentiment score
        if 'sentiment_score' in df.columns:
            features['sentiment_score'] = df['sentiment_score']

        # Lagged features
        lags_df = self.sentiment_lags(sentiment, symbol)
        features = pd.concat([features, lags_df], axis=1)

        # Momentum features
        momentum_df = self.sentiment_momentum(sentiment, symbol)
        features = pd.concat([features, momentum_df], axis=1)

        # Moving averages
        ma_df = self.sentiment_ma(sentiment, symbol)
        features = pd.concat([features, ma_df], axis=1)

        # Disagreement features
        disagreement_df = self.sentiment_disagreement(sentiment, symbol)
        features = pd.concat([features, disagreement_df], axis=1)

        # Z-score features
        zscore_df = self.sentiment_zscore(sentiment, symbol)
        features = pd.concat([features, zscore_df], axis=1)

        # Article count features
        count_df = self.article_count_features(sentiment, symbol)
        features = pd.concat([features, count_df], axis=1)

        # Confidence features
        conf_df = self.confidence_features(sentiment, symbol)
        features = pd.concat([features, conf_df], axis=1)

        # Signal features
        signal_df = self.signal_features(sentiment, symbol)
        features = pd.concat([features, signal_df], axis=1)

        # Interaction features
        interaction_df = self.interaction_features(sentiment, symbol)
        features = pd.concat([features, interaction_df], axis=1)

        return features
