"""
Risk Feature Builder.

Provides risk-based features for ML models:
- Rolling VaR at multiple confidence levels
- Volatility regime features
- Drawdown features
- GARCH-based conditional volatility
"""

from typing import Optional, List, Union

import numpy as np
import pandas as pd

from ..risk import VaRCalculator, VolatilityForecaster, DrawdownAnalyzer


class RiskFeatureBuilder:
    """
    Builder for risk-derived features.

    Integrates with the existing risk module components to generate
    ML-ready features from VaR, volatility, and drawdown calculations.

    All methods avoid lookahead bias - features at time t
    only use data from time t and earlier.

    Example:
        >>> builder = RiskFeatureBuilder()
        >>> prices = pd.read_csv('prices.csv')
        >>> features = builder.build_all(prices)
    """

    DEFAULT_VAR_WINDOWS = [21, 63]
    DEFAULT_VAR_CONFIDENCES = [0.95, 0.99]
    DEFAULT_VOL_WINDOWS = [5, 10, 21, 63]

    def __init__(
        self,
        var_calculator: Optional[VaRCalculator] = None,
        vol_forecaster: Optional[VolatilityForecaster] = None,
        dd_analyzer: Optional[DrawdownAnalyzer] = None,
        var_windows: Optional[List[int]] = None,
        var_confidences: Optional[List[float]] = None,
        vol_windows: Optional[List[int]] = None
    ):
        """
        Initialize the risk feature builder.

        Args:
            var_calculator: VaRCalculator instance (creates new if None).
            vol_forecaster: VolatilityForecaster instance (creates new if None).
            dd_analyzer: DrawdownAnalyzer instance (creates new if None).
            var_windows: Windows for rolling VaR. Default: [21, 63]
            var_confidences: Confidence levels for VaR. Default: [0.95, 0.99]
            vol_windows: Windows for volatility. Default: [5, 10, 21, 63]
        """
        self.var_calculator = var_calculator or VaRCalculator()
        self.vol_forecaster = vol_forecaster or VolatilityForecaster()
        self.dd_analyzer = dd_analyzer or DrawdownAnalyzer()

        self.var_windows = var_windows or self.DEFAULT_VAR_WINDOWS
        self.var_confidences = var_confidences or self.DEFAULT_VAR_CONFIDENCES
        self.vol_windows = vol_windows or self.DEFAULT_VOL_WINDOWS

    def _get_returns(
        self,
        prices: Union[pd.DataFrame, pd.Series]
    ) -> pd.Series:
        """Extract and validate returns from prices."""
        if isinstance(prices, pd.DataFrame):
            if 'Close' in prices.columns:
                close = prices['Close']
            else:
                close = prices.iloc[:, 0]
        else:
            close = prices

        returns = close.pct_change().dropna()
        return returns

    def _get_close(
        self,
        prices: Union[pd.DataFrame, pd.Series]
    ) -> pd.Series:
        """Extract close prices."""
        if isinstance(prices, pd.DataFrame):
            if 'Close' in prices.columns:
                return prices['Close']
            return prices.iloc[:, 0]
        return prices

    def var_features(
        self,
        prices: Union[pd.DataFrame, pd.Series],
        windows: Optional[List[int]] = None,
        confidences: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """
        Calculate rolling VaR features.

        Args:
            prices: DataFrame with 'Close' column or Series of prices.
            windows: Rolling windows for VaR. Default: [21, 63]
            confidences: Confidence levels. Default: [0.95, 0.99]

        Returns:
            DataFrame with rolling VaR columns:
            - var_{conf}_{window}: Rolling VaR at given confidence and window
            - es_{conf}_{window}: Rolling Expected Shortfall

        Example:
            >>> var_df = builder.var_features(prices)
            >>> current_var_95 = var_df['var_95_21'].iloc[-1]
        """
        returns = self._get_returns(prices)
        windows = windows or self.var_windows
        confidences = confidences or self.var_confidences

        result = pd.DataFrame(index=returns.index)

        for window in windows:
            for conf in confidences:
                conf_str = f"{int(conf * 100)}"

                # Rolling VaR
                var_col = f'var_{conf_str}_{window}'
                result[var_col] = self.var_calculator.rolling_var(
                    returns, window=window, confidence=conf
                )

                # Rolling Expected Shortfall
                es_col = f'es_{conf_str}_{window}'
                result[es_col] = self._rolling_es(returns, window, conf)

        return result

    def _rolling_es(
        self,
        returns: pd.Series,
        window: int,
        confidence: float
    ) -> pd.Series:
        """Calculate rolling Expected Shortfall."""
        def es_func(x):
            if len(x) < 10:
                return np.nan
            try:
                return self.var_calculator.expected_shortfall(x, confidence)
            except Exception:
                return np.nan

        return returns.rolling(window=window).apply(es_func, raw=False)

    def volatility_features(
        self,
        prices: Union[pd.DataFrame, pd.Series],
        windows: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Calculate volatility features.

        Args:
            prices: DataFrame with 'Close' column or Series of prices.
            windows: Rolling windows. Default: [5, 10, 21, 63]

        Returns:
            DataFrame with volatility columns:
            - realized_vol_N: Realized volatility (annualized)
            - ewma_vol_N: EWMA volatility (annualized)
            - vol_ratio_N_M: Ratio of short to long window volatility
        """
        returns = self._get_returns(prices)
        windows = windows or self.vol_windows

        result = pd.DataFrame(index=returns.index)

        for window in windows:
            # Realized volatility
            result[f'realized_vol_{window}'] = self.vol_forecaster.realized_volatility(
                returns, window=window, annualize=True
            )

            # EWMA volatility
            result[f'ewma_vol_{window}'] = self.vol_forecaster.ewma_volatility(
                returns, span=window, annualize=True
            )

        # Volatility ratios (term structure indicators)
        if len(windows) >= 2:
            sorted_windows = sorted(windows)
            for i in range(len(sorted_windows) - 1):
                short_w = sorted_windows[i]
                long_w = sorted_windows[i + 1]
                ratio_col = f'vol_ratio_{short_w}_{long_w}'
                result[ratio_col] = (
                    result[f'realized_vol_{short_w}'] /
                    (result[f'realized_vol_{long_w}'] + 1e-6)
                )

        return result

    def volatility_regime_features(
        self,
        prices: Union[pd.DataFrame, pd.Series],
        window: int = 63
    ) -> pd.DataFrame:
        """
        Calculate volatility regime features.

        Args:
            prices: DataFrame with 'Close' column or Series of prices.
            window: Window for regime calculation.

        Returns:
            DataFrame with regime features:
            - vol_regime: 'low', 'medium', or 'high'
            - vol_regime_low: 1 if in low vol regime
            - vol_regime_high: 1 if in high vol regime
            - vol_regime_duration: Days in current regime
        """
        returns = self._get_returns(prices)

        result = pd.DataFrame(index=returns.index)

        # Get volatility regime
        regime_series = self.vol_forecaster.volatility_regime(
            returns, window=window
        )
        result['vol_regime'] = regime_series

        # One-hot encode regime
        result['vol_regime_low'] = (regime_series == 'low').astype(int)
        result['vol_regime_medium'] = (regime_series == 'medium').astype(int)
        result['vol_regime_high'] = (regime_series == 'high').astype(int)

        # Regime duration
        result['vol_regime_duration'] = self._calculate_regime_duration(regime_series)

        return result

    def _calculate_regime_duration(self, regime_series: pd.Series) -> pd.Series:
        """Calculate how many days in current regime."""
        duration = pd.Series(index=regime_series.index, dtype=float)

        if len(regime_series) == 0:
            return duration

        current_regime = None
        count = 0

        for i, (idx, regime) in enumerate(regime_series.items()):
            if pd.isna(regime):
                duration.iloc[i] = np.nan
                continue

            if regime == current_regime:
                count += 1
            else:
                current_regime = regime
                count = 1

            duration.iloc[i] = count

        return duration

    def drawdown_features(
        self,
        prices: Union[pd.DataFrame, pd.Series]
    ) -> pd.DataFrame:
        """
        Calculate drawdown features.

        Args:
            prices: DataFrame with 'Close' column or Series of prices.

        Returns:
            DataFrame with drawdown features:
            - current_drawdown: Current drawdown from peak
            - days_since_peak: Days since last all-time high
            - underwater: 1 if in drawdown, 0 if at peak
            - drawdown_zscore: Z-score of current drawdown
        """
        close = self._get_close(prices)

        result = pd.DataFrame(index=close.index)

        # Calculate drawdown series
        dd_df = self.dd_analyzer.calculate_drawdown(close)

        result['current_drawdown'] = dd_df['drawdown']
        result['underwater'] = (dd_df['drawdown'] > 0).astype(int)

        # Days since peak
        peak = dd_df['peak']
        at_peak = close >= peak
        days_since = pd.Series(index=close.index, dtype=float)

        count = 0
        for i, (idx, is_peak) in enumerate(at_peak.items()):
            if is_peak:
                count = 0
            else:
                count += 1
            days_since.iloc[i] = count

        result['days_since_peak'] = days_since

        # Drawdown z-score (relative to historical drawdowns)
        dd = result['current_drawdown']
        rolling_mean = dd.rolling(window=63).mean()
        rolling_std = dd.rolling(window=63).std()
        result['drawdown_zscore'] = (dd - rolling_mean) / (rolling_std + 1e-6)

        return result

    def garch_features(
        self,
        prices: Union[pd.DataFrame, pd.Series],
        refit_every: int = 21
    ) -> pd.DataFrame:
        """
        Calculate GARCH-based features.

        Due to computational cost, GARCH is refitted periodically
        rather than on every observation.

        Args:
            prices: DataFrame with 'Close' column or Series of prices.
            refit_every: Days between GARCH refitting. Default: 21

        Returns:
            DataFrame with GARCH features:
            - garch_vol: GARCH conditional volatility
            - garch_forecast_1: 1-day ahead forecast
            - garch_persistence: Model persistence (alpha + beta)
            - vol_surprise: Realized vs predicted volatility ratio

        Note:
            Returns empty columns if GARCH fitting fails.
        """
        returns = self._get_returns(prices)

        result = pd.DataFrame(index=returns.index)

        # Need sufficient data for GARCH
        if len(returns) < 100:
            result['garch_vol'] = np.nan
            result['garch_forecast_1'] = np.nan
            result['garch_persistence'] = np.nan
            result['vol_surprise'] = np.nan
            return result

        try:
            # Fit GARCH once on full sample for conditional vol
            garch_result = self.vol_forecaster.garch_forecast(
                returns, horizon=1
            )

            # Conditional volatility (annualized)
            cond_vol = garch_result.conditional_volatility * np.sqrt(252)

            # Align index
            result['garch_vol'] = cond_vol

            # Persistence
            result['garch_persistence'] = garch_result.persistence

            # 1-day forecast (last value repeated - approximation)
            result['garch_forecast_1'] = garch_result.forecast[0] * np.sqrt(252)

            # Volatility surprise: realized vs GARCH predicted
            realized = returns.rolling(window=5).std() * np.sqrt(252)
            result['vol_surprise'] = realized / (result['garch_vol'].shift(1) + 1e-6)

        except Exception:
            result['garch_vol'] = np.nan
            result['garch_forecast_1'] = np.nan
            result['garch_persistence'] = np.nan
            result['vol_surprise'] = np.nan

        return result

    def tail_risk_features(
        self,
        prices: Union[pd.DataFrame, pd.Series],
        window: int = 63
    ) -> pd.DataFrame:
        """
        Calculate tail risk features.

        Args:
            prices: DataFrame with 'Close' column or Series of prices.
            window: Rolling window for calculations.

        Returns:
            DataFrame with tail risk features:
            - skewness: Rolling return skewness
            - kurtosis: Rolling return kurtosis (excess)
            - tail_ratio: Ratio of 5th to 95th percentile returns
        """
        returns = self._get_returns(prices)

        result = pd.DataFrame(index=returns.index)

        # Rolling skewness
        result['skewness'] = returns.rolling(window=window).skew()

        # Rolling kurtosis (excess)
        result['kurtosis'] = returns.rolling(window=window).kurt()

        # Tail ratio (left tail / right tail)
        def tail_ratio(x):
            if len(x) < 10:
                return np.nan
            p5 = np.percentile(x, 5)
            p95 = np.percentile(x, 95)
            if p95 == 0:
                return np.nan
            return abs(p5) / (abs(p95) + 1e-6)

        result['tail_ratio'] = returns.rolling(window=window).apply(tail_ratio, raw=True)

        return result

    def risk_adjusted_features(
        self,
        prices: Union[pd.DataFrame, pd.Series],
        window: int = 21
    ) -> pd.DataFrame:
        """
        Calculate risk-adjusted return features.

        Args:
            prices: DataFrame with 'Close' column or Series of prices.
            window: Rolling window for calculations.

        Returns:
            DataFrame with risk-adjusted features:
            - sharpe_ratio: Rolling Sharpe ratio (annualized, 0 rf)
            - sortino_ratio: Rolling Sortino ratio (downside deviation)
            - calmar_ratio: Rolling Calmar ratio (return / max dd)
        """
        returns = self._get_returns(prices)
        close = self._get_close(prices)

        result = pd.DataFrame(index=returns.index)

        # Rolling Sharpe ratio (assuming 0 risk-free rate)
        rolling_mean = returns.rolling(window=window).mean() * 252
        rolling_std = returns.rolling(window=window).std() * np.sqrt(252)
        result['sharpe_ratio'] = rolling_mean / (rolling_std + 1e-6)

        # Rolling Sortino ratio
        def sortino_func(x):
            if len(x) < 10:
                return np.nan
            mean_ret = x.mean() * 252
            downside = x[x < 0]
            if len(downside) < 2:
                return np.nan
            downside_std = downside.std() * np.sqrt(252)
            return mean_ret / (downside_std + 1e-6)

        result['sortino_ratio'] = returns.rolling(window=window).apply(
            sortino_func, raw=False
        )

        # Rolling Calmar ratio (approximation using rolling max dd)
        dd = self.dd_analyzer.calculate_drawdown(close)['drawdown']
        rolling_max_dd = dd.rolling(window=window).max()
        rolling_return = returns.rolling(window=window).mean() * 252
        result['calmar_ratio'] = rolling_return / (rolling_max_dd + 1e-6)

        return result

    def build_all(
        self,
        prices: Union[pd.DataFrame, pd.Series],
        include_garch: bool = True
    ) -> pd.DataFrame:
        """
        Build all risk features.

        Args:
            prices: DataFrame with price columns.
            include_garch: Whether to include GARCH features (slower).

        Returns:
            DataFrame with all calculated risk features.

        Example:
            >>> prices = pd.read_csv('AAPL_prices.csv')
            >>> features = builder.build_all(prices)
            >>> print(f"Built {len(features.columns)} risk features")
        """
        returns = self._get_returns(prices)
        features = pd.DataFrame(index=returns.index)

        # VaR features
        var_df = self.var_features(prices)
        features = pd.concat([features, var_df], axis=1)

        # Volatility features
        vol_df = self.volatility_features(prices)
        features = pd.concat([features, vol_df], axis=1)

        # Volatility regime features
        regime_df = self.volatility_regime_features(prices)
        features = pd.concat([features, regime_df], axis=1)

        # Drawdown features
        dd_df = self.drawdown_features(prices)
        features = pd.concat([features, dd_df], axis=1)

        # GARCH features (optional, computationally expensive)
        if include_garch:
            garch_df = self.garch_features(prices)
            features = pd.concat([features, garch_df], axis=1)

        # Tail risk features
        tail_df = self.tail_risk_features(prices)
        features = pd.concat([features, tail_df], axis=1)

        # Risk-adjusted features
        risk_adj_df = self.risk_adjusted_features(prices)
        features = pd.concat([features, risk_adj_df], axis=1)

        return features
