"""
Feature Pipeline Builder.

Orchestrates the feature building process, combining:
- Price/technical features
- Sentiment features
- Risk features

Handles date alignment, target creation, and saving.
"""

from pathlib import Path
from typing import Optional, List, Union
import logging

import numpy as np
import pandas as pd

from .price_features import PriceFeatureBuilder
from .sentiment_features import SentimentFeatureBuilder
from .risk_features import RiskFeatureBuilder


logger = logging.getLogger(__name__)


class FeatureBuilder:
    """
    Main feature building pipeline.

    Combines price, sentiment, and risk features into a single
    ML-ready dataset with proper date alignment and target creation.

    Key principles:
    - No lookahead bias: Features at time t only use data from t and earlier
    - Proper alignment: All feature sources aligned by date
    - Target creation: Forward-looking targets for prediction

    Example:
        >>> builder = FeatureBuilder()
        >>> prices = pd.read_csv('data/raw/AAPL_prices.csv')
        >>> sentiment = pd.read_csv('data/processed/daily_sentiment.csv')
        >>> features = builder.build_features(prices, sentiment, symbol='AAPL')
        >>> X, y = builder.create_ml_dataset(features, target_horizon=1)
    """

    def __init__(
        self,
        price_builder: Optional[PriceFeatureBuilder] = None,
        sentiment_builder: Optional[SentimentFeatureBuilder] = None,
        risk_builder: Optional[RiskFeatureBuilder] = None
    ):
        """
        Initialize the feature builder.

        Args:
            price_builder: PriceFeatureBuilder instance.
            sentiment_builder: SentimentFeatureBuilder instance.
            risk_builder: RiskFeatureBuilder instance.
        """
        self.price_builder = price_builder or PriceFeatureBuilder()
        self.sentiment_builder = sentiment_builder or SentimentFeatureBuilder()
        self.risk_builder = risk_builder or RiskFeatureBuilder()

    def _prepare_prices(
        self,
        prices: pd.DataFrame
    ) -> pd.DataFrame:
        """Prepare price data with proper index."""
        df = prices.copy()

        # Ensure date column is datetime
        if 'Date' in df.columns:
            # Use utc=True to handle mixed timezones, then convert to naive
            df['Date'] = pd.to_datetime(df['Date'], utc=True)
            df = df.set_index('Date')
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], utc=True)
            df = df.set_index('date')

        # If index is not datetime, try to convert
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index, utc=True)
            except Exception:
                pass

        # Strip timezone for consistent alignment (convert to date-only)
        if isinstance(df.index, pd.DatetimeIndex):
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            # Normalize to date (remove time component)
            df.index = df.index.normalize()

        # Sort by date
        df = df.sort_index()

        return df

    def _prepare_sentiment(
        self,
        sentiment: pd.DataFrame,
        symbol: str
    ) -> pd.DataFrame:
        """Prepare sentiment data for a specific symbol."""
        df = sentiment.copy()

        # Filter by symbol
        if 'symbol' in df.columns:
            df = df[df['symbol'] == symbol].copy()

        # Ensure date is datetime index
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')

        # Strip timezone if present and normalize
        if isinstance(df.index, pd.DatetimeIndex):
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            # Normalize to date (remove time component)
            df.index = df.index.normalize()

        df = df.sort_index()

        return df

    def build_price_features(
        self,
        prices: pd.DataFrame,
        include_volume: bool = True
    ) -> pd.DataFrame:
        """
        Build price/technical features only.

        Args:
            prices: DataFrame with OHLCV columns.
            include_volume: Whether to include volume features.

        Returns:
            DataFrame with price features.
        """
        prices = self._prepare_prices(prices)
        return self.price_builder.build_all(prices, include_volume=include_volume)

    def build_sentiment_features(
        self,
        sentiment: pd.DataFrame,
        symbol: str
    ) -> pd.DataFrame:
        """
        Build sentiment features for a symbol.

        Args:
            sentiment: DataFrame with sentiment data.
            symbol: Symbol to filter for.

        Returns:
            DataFrame with sentiment features.
        """
        sentiment = self._prepare_sentiment(sentiment, symbol)
        # Since data is already filtered and indexed, pass directly
        return self.sentiment_builder.build_all(sentiment)

    def build_risk_features(
        self,
        prices: pd.DataFrame,
        include_garch: bool = False
    ) -> pd.DataFrame:
        """
        Build risk features.

        Args:
            prices: DataFrame with price data.
            include_garch: Whether to include GARCH features.

        Returns:
            DataFrame with risk features.
        """
        prices = self._prepare_prices(prices)
        return self.risk_builder.build_all(prices, include_garch=include_garch)

    def align_features(
        self,
        *feature_dfs: pd.DataFrame,
        how: str = 'inner'
    ) -> pd.DataFrame:
        """
        Align multiple feature DataFrames by date.

        Args:
            *feature_dfs: Variable number of feature DataFrames.
            how: Join method ('inner', 'outer', 'left', 'right').

        Returns:
            Combined DataFrame with aligned dates.

        Example:
            >>> aligned = builder.align_features(price_features, sentiment_features)
        """
        valid_dfs = [df for df in feature_dfs if df is not None and len(df) > 0]

        if len(valid_dfs) == 0:
            return pd.DataFrame()

        if len(valid_dfs) == 1:
            return valid_dfs[0]

        # Start with first DataFrame
        result = valid_dfs[0]

        # Iteratively join others
        for df in valid_dfs[1:]:
            # Ensure indices are comparable (both date-only)
            if isinstance(result.index, pd.DatetimeIndex):
                result.index = result.index.normalize()
            if isinstance(df.index, pd.DatetimeIndex):
                df.index = df.index.normalize()

            result = result.join(df, how=how, rsuffix='_dup')

            # Remove duplicate columns
            dup_cols = [c for c in result.columns if c.endswith('_dup')]
            result = result.drop(columns=dup_cols)

        return result

    def build_features(
        self,
        prices: pd.DataFrame,
        sentiment: Optional[pd.DataFrame] = None,
        symbol: Optional[str] = None,
        include_volume: bool = True,
        include_garch: bool = False,
        include_sentiment: bool = True,
        include_risk: bool = True
    ) -> pd.DataFrame:
        """
        Build complete feature set combining all sources.

        This is the main method for creating ML-ready features.

        Args:
            prices: DataFrame with OHLCV columns.
            sentiment: DataFrame with sentiment data (optional).
            symbol: Symbol for filtering sentiment data.
            include_volume: Whether to include volume features.
            include_garch: Whether to include GARCH features.
            include_sentiment: Whether to include sentiment features.
            include_risk: Whether to include risk features.

        Returns:
            DataFrame with all features aligned by date.

        Example:
            >>> features = builder.build_features(
            ...     prices=prices,
            ...     sentiment=sentiment,
            ...     symbol='AAPL',
            ...     include_garch=False
            ... )
            >>> print(f"Feature matrix: {features.shape}")
        """
        prices = self._prepare_prices(prices)
        feature_dfs = []

        # Build price features
        logger.info("Building price features...")
        price_features = self.price_builder.build_all(
            prices, include_volume=include_volume
        )
        feature_dfs.append(price_features)

        # Build sentiment features
        if include_sentiment and sentiment is not None and symbol is not None:
            logger.info(f"Building sentiment features for {symbol}...")
            sentiment_prep = self._prepare_sentiment(sentiment, symbol)
            if len(sentiment_prep) > 0:
                # Build features - need to pass original sentiment for builder
                sent_features = self.sentiment_builder.build_all(sentiment_prep)
                feature_dfs.append(sent_features)

        # Build risk features
        if include_risk:
            logger.info("Building risk features...")
            risk_features = self.risk_builder.build_all(
                prices, include_garch=include_garch
            )
            feature_dfs.append(risk_features)

        # Align all features
        logger.info("Aligning features...")
        features = self.align_features(*feature_dfs)

        # Add symbol column if provided
        if symbol:
            features['symbol'] = symbol

        return features

    def create_target(
        self,
        prices: pd.DataFrame,
        horizon: int = 1,
        target_type: str = 'return'
    ) -> pd.Series:
        """
        Create prediction target.

        The target is forward-looking (future return/direction).

        Args:
            prices: DataFrame with 'Close' column.
            horizon: Prediction horizon in days. Default: 1
            target_type: Type of target:
                - 'return': Raw return over horizon
                - 'log_return': Log return over horizon
                - 'direction': Binary direction (1 if up, 0 if down)
                - 'direction_signed': Signed direction (-1, 0, +1)

        Returns:
            Series with target values.

        Note:
            Target is shifted so that target[t] represents the future
            return from t to t+horizon. The last `horizon` rows will be NaN.
        """
        prices = self._prepare_prices(prices)

        if 'Close' in prices.columns:
            close = prices['Close']
        else:
            close = prices.iloc[:, 0]

        if target_type == 'return':
            # Future return: (close[t+h] - close[t]) / close[t]
            future_price = close.shift(-horizon)
            target = (future_price - close) / close
            target.name = f'target_return_{horizon}d'

        elif target_type == 'log_return':
            future_price = close.shift(-horizon)
            target = np.log(future_price / close)
            target.name = f'target_log_return_{horizon}d'

        elif target_type == 'direction':
            future_price = close.shift(-horizon)
            target = (future_price > close).astype(int)
            target.name = f'target_direction_{horizon}d'

        elif target_type == 'direction_signed':
            future_price = close.shift(-horizon)
            diff = future_price - close
            target = np.sign(diff)
            target.name = f'target_direction_signed_{horizon}d'

        else:
            raise ValueError(f"Unknown target_type: {target_type}")

        return target

    def create_ml_dataset(
        self,
        features: pd.DataFrame,
        prices: pd.DataFrame,
        target_horizon: int = 1,
        target_type: str = 'return',
        dropna: bool = True
    ) -> tuple:
        """
        Create ML-ready dataset with features and target.

        Args:
            features: DataFrame with features.
            prices: DataFrame with prices for target creation.
            target_horizon: Prediction horizon in days.
            target_type: Type of target ('return', 'direction', etc.).
            dropna: Whether to drop rows with any NaN values.

        Returns:
            Tuple of (X, y) where X is features DataFrame and y is target Series.

        Example:
            >>> features = builder.build_features(prices, sentiment, symbol='AAPL')
            >>> X, y = builder.create_ml_dataset(features, prices, target_horizon=1)
            >>> print(f"Training samples: {len(X)}")
        """
        # Create target
        target = self.create_target(prices, horizon=target_horizon, target_type=target_type)

        # Align features and target
        prices_prep = self._prepare_prices(prices)
        target.index = prices_prep.index

        # Ensure feature index matches
        if isinstance(features.index, pd.DatetimeIndex):
            features.index = features.index.normalize()
        if isinstance(target.index, pd.DatetimeIndex):
            target.index = target.index.normalize()

        # Join features and target
        combined = features.join(target.to_frame(), how='inner')

        if dropna:
            combined = combined.dropna()

        # Separate X and y
        target_col = target.name
        X = combined.drop(columns=[target_col])
        y = combined[target_col]

        # Remove non-numeric and symbol columns from X
        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            X = X.drop(columns=non_numeric)

        logger.info(f"ML dataset: {X.shape[0]} samples, {X.shape[1]} features")

        return X, y

    def save_features(
        self,
        features: pd.DataFrame,
        symbol: str,
        output_dir: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Save features to CSV.

        Args:
            features: DataFrame with features.
            symbol: Symbol name for filename.
            output_dir: Output directory. Default: data/processed/

        Returns:
            Path to saved file.
        """
        if output_dir is None:
            from config import Config
            output_dir = Config.PROCESSED_DATA_DIR

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{symbol}_features.csv"
        filepath = output_dir / filename

        features.to_csv(filepath)
        logger.info(f"Saved features to {filepath}")

        return filepath

    def load_features(
        self,
        symbol: str,
        input_dir: Optional[Union[str, Path]] = None
    ) -> pd.DataFrame:
        """
        Load previously saved features.

        Args:
            symbol: Symbol name for filename.
            input_dir: Input directory. Default: data/processed/

        Returns:
            DataFrame with features.
        """
        if input_dir is None:
            from config import Config
            input_dir = Config.PROCESSED_DATA_DIR

        input_dir = Path(input_dir)
        filename = f"{symbol}_features.csv"
        filepath = input_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Feature file not found: {filepath}")

        features = pd.read_csv(filepath, index_col=0, parse_dates=True)
        logger.info(f"Loaded features from {filepath}: {features.shape}")

        return features

    def get_feature_names(
        self,
        features: pd.DataFrame,
        exclude_symbol: bool = True
    ) -> List[str]:
        """
        Get list of feature column names.

        Args:
            features: DataFrame with features.
            exclude_symbol: Whether to exclude 'symbol' column.

        Returns:
            List of feature column names.
        """
        cols = features.columns.tolist()

        if exclude_symbol and 'symbol' in cols:
            cols.remove('symbol')

        # Remove any target columns that might be present
        cols = [c for c in cols if not c.startswith('target_')]

        return cols

    def feature_summary(
        self,
        features: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate summary statistics for features.

        Args:
            features: DataFrame with features.

        Returns:
            DataFrame with feature statistics.
        """
        # Get only numeric columns
        numeric_features = features.select_dtypes(include=[np.number])

        summary = pd.DataFrame({
            'count': numeric_features.count(),
            'missing': numeric_features.isna().sum(),
            'missing_pct': (numeric_features.isna().sum() / len(features) * 100).round(2),
            'mean': numeric_features.mean(),
            'std': numeric_features.std(),
            'min': numeric_features.min(),
            'max': numeric_features.max()
        })

        return summary.sort_values('missing_pct', ascending=False)
