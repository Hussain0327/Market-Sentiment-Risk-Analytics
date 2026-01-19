"""
Price and Technical Feature Builder.

Provides technical indicators and price-based features:
- Multi-period returns
- Momentum indicators (RSI, MACD, Stochastic)
- Volatility features
- Bollinger Bands
- Volume-based features
"""

from typing import Optional, Union, List

import numpy as np
import pandas as pd


class PriceFeatureBuilder:
    """
    Builder for price-based and technical features.

    All methods are designed to avoid lookahead bias - features
    at time t only use data from time t and earlier.

    Example:
        >>> builder = PriceFeatureBuilder()
        >>> prices = pd.read_csv('prices.csv')
        >>> features = builder.build_all(prices)
        >>> print(f"Features shape: {features.shape}")
    """

    DEFAULT_WINDOWS = [5, 10, 21, 63]
    DEFAULT_RETURN_PERIODS = [1, 5, 21]

    def __init__(
        self,
        windows: Optional[List[int]] = None,
        return_periods: Optional[List[int]] = None
    ):
        """
        Initialize the price feature builder.

        Args:
            windows: Rolling window sizes for volatility and indicators.
                     Default: [5, 10, 21, 63]
            return_periods: Periods for calculating returns.
                           Default: [1, 5, 21]
        """
        self.windows = windows or self.DEFAULT_WINDOWS
        self.return_periods = return_periods or self.DEFAULT_RETURN_PERIODS

    def _validate_prices(
        self,
        prices: Union[pd.DataFrame, pd.Series]
    ) -> pd.DataFrame:
        """Validate and convert prices to DataFrame format."""
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name='Close')

        required_cols = ['Close']
        missing = [c for c in required_cols if c not in prices.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Ensure numeric and handle NaN
        prices = prices.copy()
        for col in prices.columns:
            if prices[col].dtype == 'object':
                prices[col] = pd.to_numeric(prices[col], errors='coerce')

        return prices

    def returns(
        self,
        prices: Union[pd.DataFrame, pd.Series],
        periods: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Calculate multi-period returns.

        Args:
            prices: DataFrame with 'Close' column or Series of prices.
            periods: List of periods for returns. Default: [1, 5, 21]

        Returns:
            DataFrame with return columns: return_1d, return_5d, etc.

        Example:
            >>> returns_df = builder.returns(prices, periods=[1, 5, 21])
            >>> daily_return = returns_df['return_1d'].iloc[-1]
        """
        prices = self._validate_prices(prices)
        close = prices['Close']
        periods = periods or self.return_periods

        result = pd.DataFrame(index=prices.index)

        for period in periods:
            col_name = f'return_{period}d'
            result[col_name] = close.pct_change(periods=period)

        return result

    def log_returns(
        self,
        prices: Union[pd.DataFrame, pd.Series],
        periods: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Calculate multi-period log returns.

        Log returns are additive across time and more appropriate
        for longer horizons.

        Args:
            prices: DataFrame with 'Close' column or Series of prices.
            periods: List of periods for returns.

        Returns:
            DataFrame with log return columns.
        """
        prices = self._validate_prices(prices)
        close = prices['Close']
        periods = periods or self.return_periods

        result = pd.DataFrame(index=prices.index)

        for period in periods:
            col_name = f'log_return_{period}d'
            result[col_name] = np.log(close / close.shift(period))

        return result

    def rsi(
        self,
        prices: Union[pd.DataFrame, pd.Series],
        window: int = 14
    ) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).

        RSI measures momentum by comparing average gains to average losses.
        Values above 70 indicate overbought, below 30 indicate oversold.

        Args:
            prices: DataFrame with 'Close' column or Series of prices.
            window: Lookback window for RSI calculation. Default: 14

        Returns:
            Series with RSI values (0-100).

        Example:
            >>> rsi = builder.rsi(prices, window=14)
            >>> if rsi.iloc[-1] > 70:
            ...     print("Overbought")
        """
        prices = self._validate_prices(prices)
        close = prices['Close']

        # Calculate price changes
        delta = close.diff()

        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = (-delta).where(delta < 0, 0)

        # Calculate average gains/losses using EWMA (Wilder's smoothing)
        avg_gain = gains.ewm(span=window, adjust=False).mean()
        avg_loss = losses.ewm(span=window, adjust=False).mean()

        # Calculate RS and RSI
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        # Handle edge cases
        rsi = rsi.fillna(50)  # Neutral when no movement

        rsi.name = f'rsi_{window}'
        return rsi

    def macd(
        self,
        prices: Union[pd.DataFrame, pd.Series],
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> pd.DataFrame:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        MACD shows trend direction and momentum through the relationship
        between two moving averages.

        Args:
            prices: DataFrame with 'Close' column or Series of prices.
            fast: Fast EMA period. Default: 12
            slow: Slow EMA period. Default: 26
            signal: Signal line EMA period. Default: 9

        Returns:
            DataFrame with columns: macd, macd_signal, macd_histogram

        Example:
            >>> macd_df = builder.macd(prices)
            >>> if macd_df['macd_histogram'].iloc[-1] > 0:
            ...     print("Bullish momentum")
        """
        prices = self._validate_prices(prices)
        close = prices['Close']

        # Calculate EMAs
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()

        # MACD line
        macd_line = ema_fast - ema_slow

        # Signal line
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()

        # Histogram
        histogram = macd_line - signal_line

        result = pd.DataFrame({
            'macd': macd_line,
            'macd_signal': signal_line,
            'macd_histogram': histogram
        }, index=prices.index)

        return result

    def stochastic(
        self,
        prices: pd.DataFrame,
        k_window: int = 14,
        d_window: int = 3
    ) -> pd.DataFrame:
        """
        Calculate Stochastic Oscillator (%K and %D).

        Compares closing price to the high-low range over a period.
        Values above 80 indicate overbought, below 20 indicate oversold.

        Args:
            prices: DataFrame with 'High', 'Low', 'Close' columns.
            k_window: Window for %K calculation. Default: 14
            d_window: Window for %D (signal line). Default: 3

        Returns:
            DataFrame with columns: stoch_k, stoch_d
        """
        required = ['High', 'Low', 'Close']
        missing = [c for c in required if c not in prices.columns]
        if missing:
            raise ValueError(f"Stochastic requires columns: {required}. Missing: {missing}")

        high = prices['High']
        low = prices['Low']
        close = prices['Close']

        # Highest high and lowest low over window
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()

        # %K - fast stochastic
        stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)

        # %D - slow stochastic (SMA of %K)
        stoch_d = stoch_k.rolling(window=d_window).mean()

        return pd.DataFrame({
            'stoch_k': stoch_k,
            'stoch_d': stoch_d
        }, index=prices.index)

    def volatility_features(
        self,
        prices: Union[pd.DataFrame, pd.Series],
        windows: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Calculate volatility features at multiple windows.

        Args:
            prices: DataFrame with 'Close' column or Series of prices.
            windows: List of window sizes. Default: [5, 10, 21, 63]

        Returns:
            DataFrame with volatility columns for each window.

        Example:
            >>> vol_df = builder.volatility_features(prices)
            >>> current_vol = vol_df['volatility_21'].iloc[-1]
        """
        prices = self._validate_prices(prices)
        close = prices['Close']
        windows = windows or self.windows

        # Calculate returns for volatility
        returns = close.pct_change()

        result = pd.DataFrame(index=prices.index)

        for window in windows:
            col_name = f'volatility_{window}'
            # Annualized volatility
            result[col_name] = returns.rolling(window=window).std() * np.sqrt(252)

        return result

    def bollinger_bands(
        self,
        prices: Union[pd.DataFrame, pd.Series],
        window: int = 20,
        num_std: float = 2.0
    ) -> pd.DataFrame:
        """
        Calculate Bollinger Bands and derived features.

        Bollinger Bands show price relative to recent volatility.
        Useful for identifying overbought/oversold conditions.

        Args:
            prices: DataFrame with 'Close' column or Series of prices.
            window: Window for moving average. Default: 20
            num_std: Number of standard deviations. Default: 2.0

        Returns:
            DataFrame with columns:
            - bb_middle: Middle band (SMA)
            - bb_upper: Upper band
            - bb_lower: Lower band
            - bb_pct_b: %B (price position relative to bands, 0-1)
            - bb_bandwidth: Band width (volatility proxy)

        Example:
            >>> bb_df = builder.bollinger_bands(prices)
            >>> if bb_df['bb_pct_b'].iloc[-1] > 1:
            ...     print("Price above upper band")
        """
        prices = self._validate_prices(prices)
        close = prices['Close']

        # Middle band (SMA)
        middle = close.rolling(window=window).mean()

        # Standard deviation
        std = close.rolling(window=window).std()

        # Upper and lower bands
        upper = middle + (num_std * std)
        lower = middle - (num_std * std)

        # %B - where price is relative to bands
        pct_b = (close - lower) / (upper - lower + 1e-10)

        # Bandwidth - volatility measure
        bandwidth = (upper - lower) / middle

        return pd.DataFrame({
            'bb_middle': middle,
            'bb_upper': upper,
            'bb_lower': lower,
            'bb_pct_b': pct_b,
            'bb_bandwidth': bandwidth
        }, index=prices.index)

    def moving_averages(
        self,
        prices: Union[pd.DataFrame, pd.Series],
        windows: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Calculate simple and exponential moving averages.

        Args:
            prices: DataFrame with 'Close' column or Series of prices.
            windows: List of window sizes.

        Returns:
            DataFrame with SMA and EMA for each window.
        """
        prices = self._validate_prices(prices)
        close = prices['Close']
        windows = windows or self.windows

        result = pd.DataFrame(index=prices.index)

        for window in windows:
            result[f'sma_{window}'] = close.rolling(window=window).mean()
            result[f'ema_{window}'] = close.ewm(span=window, adjust=False).mean()

        return result

    def ma_crossover_features(
        self,
        prices: Union[pd.DataFrame, pd.Series],
        fast_window: int = 10,
        slow_window: int = 21
    ) -> pd.DataFrame:
        """
        Calculate moving average crossover features.

        Args:
            prices: DataFrame with 'Close' column or Series of prices.
            fast_window: Fast MA window. Default: 10
            slow_window: Slow MA window. Default: 21

        Returns:
            DataFrame with crossover features:
            - ma_diff: Difference between fast and slow MA
            - ma_ratio: Ratio of fast to slow MA
            - ma_cross_up: 1 if fast crosses above slow, else 0
            - ma_cross_down: 1 if fast crosses below slow, else 0
        """
        prices = self._validate_prices(prices)
        close = prices['Close']

        fast_ma = close.ewm(span=fast_window, adjust=False).mean()
        slow_ma = close.ewm(span=slow_window, adjust=False).mean()

        ma_diff = fast_ma - slow_ma
        ma_ratio = fast_ma / slow_ma

        # Crossover signals
        fast_above = fast_ma > slow_ma
        fast_above_shifted = fast_above.shift(1)
        # Use where instead of fillna to avoid deprecation warning
        fast_above_prev = fast_above_shifted.where(fast_above_shifted.notna(), False).astype(bool)
        cross_up = (fast_above & ~fast_above_prev).astype(int)
        cross_down = (~fast_above & fast_above_prev).astype(int)

        return pd.DataFrame({
            'ma_diff': ma_diff,
            'ma_ratio': ma_ratio,
            'ma_cross_up': cross_up,
            'ma_cross_down': cross_down
        }, index=prices.index)

    def volume_features(
        self,
        prices: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate volume-based features.

        Args:
            prices: DataFrame with 'Close' and 'Volume' columns.

        Returns:
            DataFrame with volume features:
            - volume_sma: 20-day volume SMA
            - volume_ratio: Current volume / SMA
            - obv: On-Balance Volume
            - obv_sma: 20-day OBV SMA
            - volume_momentum: 5-day volume change

        Note:
            Returns empty DataFrame if 'Volume' column is missing.
        """
        if 'Volume' not in prices.columns:
            return pd.DataFrame(index=prices.index)

        close = prices['Close']
        volume = prices['Volume']

        result = pd.DataFrame(index=prices.index)

        # Volume SMA and ratio
        result['volume_sma'] = volume.rolling(window=20).mean()
        result['volume_ratio'] = volume / result['volume_sma'].replace(0, np.nan)

        # On-Balance Volume (OBV)
        price_change = close.diff()
        obv_sign = np.sign(price_change)
        obv_sign.iloc[0] = 0  # First value has no direction
        result['obv'] = (obv_sign * volume).cumsum()
        result['obv_sma'] = result['obv'].rolling(window=20).mean()

        # Volume momentum
        result['volume_momentum'] = volume.pct_change(periods=5)

        return result

    def price_momentum(
        self,
        prices: Union[pd.DataFrame, pd.Series],
        windows: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Calculate price momentum features.

        Args:
            prices: DataFrame with 'Close' column or Series of prices.
            windows: List of window sizes.

        Returns:
            DataFrame with momentum features:
            - momentum_N: N-day price momentum (current / N days ago - 1)
            - roc_N: Rate of change (same as momentum, percentage)
        """
        prices = self._validate_prices(prices)
        close = prices['Close']
        windows = windows or self.windows

        result = pd.DataFrame(index=prices.index)

        for window in windows:
            result[f'momentum_{window}'] = close / close.shift(window) - 1
            result[f'roc_{window}'] = close.pct_change(periods=window) * 100

        return result

    def atr(
        self,
        prices: pd.DataFrame,
        window: int = 14
    ) -> pd.Series:
        """
        Calculate Average True Range (ATR).

        ATR measures volatility using the full trading range.

        Args:
            prices: DataFrame with 'High', 'Low', 'Close' columns.
            window: Window for ATR calculation. Default: 14

        Returns:
            Series with ATR values.
        """
        required = ['High', 'Low', 'Close']
        missing = [c for c in required if c not in prices.columns]
        if missing:
            return pd.Series(index=prices.index, name='atr', dtype=float)

        high = prices['High']
        low = prices['Low']
        close = prices['Close']

        # True Range components
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        # True Range is the max of the three
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR is the smoothed average of TR
        atr = true_range.ewm(span=window, adjust=False).mean()
        atr.name = 'atr'

        return atr

    def build_all(
        self,
        prices: pd.DataFrame,
        include_volume: bool = True
    ) -> pd.DataFrame:
        """
        Build all price-based features.

        Args:
            prices: DataFrame with OHLCV columns.
            include_volume: Whether to include volume features.

        Returns:
            DataFrame with all calculated features.

        Example:
            >>> prices = pd.read_csv('AAPL_prices.csv')
            >>> features = builder.build_all(prices)
            >>> print(f"Built {len(features.columns)} features")
        """
        prices = self._validate_prices(prices)
        features = pd.DataFrame(index=prices.index)

        # Returns
        returns_df = self.returns(prices)
        log_returns_df = self.log_returns(prices)
        features = pd.concat([features, returns_df, log_returns_df], axis=1)

        # Momentum indicators
        features['rsi_14'] = self.rsi(prices, window=14)

        macd_df = self.macd(prices)
        features = pd.concat([features, macd_df], axis=1)

        # Stochastic (if we have OHLC)
        if all(c in prices.columns for c in ['High', 'Low', 'Close']):
            stoch_df = self.stochastic(prices)
            features = pd.concat([features, stoch_df], axis=1)

        # Volatility
        vol_df = self.volatility_features(prices)
        features = pd.concat([features, vol_df], axis=1)

        # Bollinger Bands
        bb_df = self.bollinger_bands(prices)
        features = pd.concat([features, bb_df], axis=1)

        # Moving averages and crossovers
        ma_df = self.moving_averages(prices)
        ma_cross_df = self.ma_crossover_features(prices)
        features = pd.concat([features, ma_df, ma_cross_df], axis=1)

        # Momentum
        momentum_df = self.price_momentum(prices)
        features = pd.concat([features, momentum_df], axis=1)

        # ATR
        features['atr'] = self.atr(prices)

        # Volume features (optional)
        if include_volume and 'Volume' in prices.columns:
            vol_features = self.volume_features(prices)
            features = pd.concat([features, vol_features], axis=1)

        return features
