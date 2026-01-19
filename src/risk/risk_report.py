"""
Risk Report Generator.

Consolidates all risk metrics into comprehensive reports:
- VaR analysis
- Volatility analysis
- Drawdown analysis
- Cross-symbol comparisons
"""

from dataclasses import dataclass
from typing import Optional, Union, Dict, Any, List
from datetime import datetime

import numpy as np
import pandas as pd

from .var import VaRCalculator
from .volatility import VolatilityForecaster
from .drawdown import DrawdownAnalyzer


@dataclass
class RiskMetrics:
    """
    Consolidated risk metrics for a single asset.

    Attributes:
        symbol: Asset symbol/identifier
        var_95: 95% Value at Risk
        var_99: 99% Value at Risk
        cvar_95: 95% Conditional VaR (Expected Shortfall)
        cvar_99: 99% Conditional VaR
        volatility_21d: 21-day realized volatility (annualized)
        volatility_63d: 63-day realized volatility (annualized)
        garch_forecast: 5-day GARCH volatility forecast
        max_drawdown: Maximum drawdown
        current_drawdown: Current drawdown
        calmar_ratio: Calmar ratio
        volatility_regime: Current volatility regime
        timestamp: When metrics were calculated
    """
    symbol: str
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    volatility_21d: float
    volatility_63d: float
    garch_forecast: Optional[float]
    max_drawdown: float
    current_drawdown: float
    calmar_ratio: float
    volatility_regime: str
    timestamp: datetime


class RiskReport:
    """
    Comprehensive risk report generator.

    Combines VaR, volatility, and drawdown analysis into
    consolidated risk reports for single assets or portfolios.

    Example:
        >>> report = RiskReport()
        >>> metrics = report.generate_report(prices, returns, symbol='AAPL')
        >>> print(f"95% VaR: {metrics.var_95:.4f}")
        >>> print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
    """

    def __init__(
        self,
        var_calculator: Optional[VaRCalculator] = None,
        vol_forecaster: Optional[VolatilityForecaster] = None,
        dd_analyzer: Optional[DrawdownAnalyzer] = None
    ):
        """
        Initialize the risk report generator.

        Args:
            var_calculator: VaRCalculator instance (created if None).
            vol_forecaster: VolatilityForecaster instance (created if None).
            dd_analyzer: DrawdownAnalyzer instance (created if None).
        """
        self.var_calc = var_calculator or VaRCalculator()
        self.vol_forecaster = vol_forecaster or VolatilityForecaster()
        self.dd_analyzer = dd_analyzer or DrawdownAnalyzer()

    def generate_report(
        self,
        prices: Union[pd.Series, np.ndarray],
        returns: Optional[Union[pd.Series, np.ndarray]] = None,
        symbol: str = "UNKNOWN"
    ) -> RiskMetrics:
        """
        Generate comprehensive risk report for a single asset.

        Args:
            prices: Price series for the asset.
            returns: Return series (computed from prices if None).
            symbol: Asset symbol/identifier.

        Returns:
            RiskMetrics dataclass with all risk measures.

        Example:
            >>> prices = pd.read_csv('AAPL_prices.csv')['Close']
            >>> metrics = report.generate_report(prices, symbol='AAPL')
        """
        # Ensure pandas Series
        if isinstance(prices, np.ndarray):
            prices = pd.Series(prices)

        prices = prices.dropna()

        # Calculate returns if not provided
        if returns is None:
            returns = prices.pct_change().dropna()
        elif isinstance(returns, np.ndarray):
            returns = pd.Series(returns)
            returns = returns.dropna()

        # VaR calculations
        var_95 = self.var_calc.historical_var(returns, 0.95)
        var_99 = self.var_calc.historical_var(returns, 0.99)
        cvar_95 = self.var_calc.expected_shortfall(returns, 0.95)
        cvar_99 = self.var_calc.expected_shortfall(returns, 0.99)

        # Volatility calculations
        vol_21d = self.vol_forecaster.realized_volatility(returns, window=21)
        vol_63d = self.vol_forecaster.realized_volatility(returns, window=63)

        # GARCH forecast (with fallback)
        try:
            garch_result = self.vol_forecaster.garch_forecast(returns, horizon=5)
            # Average of 5-day forecast, annualized
            garch_forecast = float(np.mean(garch_result.forecast)) * np.sqrt(252)
        except Exception:
            garch_forecast = None

        # Volatility regime
        try:
            regimes = self.vol_forecaster.volatility_regime(returns)
            vol_regime = regimes.iloc[-1] if not pd.isna(regimes.iloc[-1]) else 'unknown'
        except Exception:
            vol_regime = 'unknown'

        # Drawdown calculations
        dd_summary = self.dd_analyzer.drawdown_summary(prices)

        return RiskMetrics(
            symbol=symbol,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            volatility_21d=float(vol_21d.iloc[-1]) if len(vol_21d) > 0 else 0,
            volatility_63d=float(vol_63d.iloc[-1]) if len(vol_63d) > 0 else 0,
            garch_forecast=garch_forecast,
            max_drawdown=dd_summary['max_drawdown'],
            current_drawdown=dd_summary['current_drawdown'],
            calmar_ratio=dd_summary['calmar_ratio'],
            volatility_regime=str(vol_regime),
            timestamp=datetime.now()
        )

    def to_dataframe(
        self,
        metrics: Union[RiskMetrics, List[RiskMetrics]]
    ) -> pd.DataFrame:
        """
        Convert risk metrics to DataFrame format.

        Args:
            metrics: Single RiskMetrics or list of RiskMetrics.

        Returns:
            DataFrame with risk metrics as columns.
        """
        if isinstance(metrics, RiskMetrics):
            metrics = [metrics]

        data = []
        for m in metrics:
            data.append({
                'symbol': m.symbol,
                'var_95': m.var_95,
                'var_99': m.var_99,
                'cvar_95': m.cvar_95,
                'cvar_99': m.cvar_99,
                'volatility_21d': m.volatility_21d,
                'volatility_63d': m.volatility_63d,
                'garch_forecast': m.garch_forecast,
                'max_drawdown': m.max_drawdown,
                'current_drawdown': m.current_drawdown,
                'calmar_ratio': m.calmar_ratio,
                'volatility_regime': m.volatility_regime,
                'timestamp': m.timestamp
            })

        return pd.DataFrame(data)

    def compare_symbols(
        self,
        symbol_data: Dict[str, Dict[str, pd.Series]]
    ) -> pd.DataFrame:
        """
        Generate comparative risk report for multiple symbols.

        Args:
            symbol_data: Dictionary mapping symbol to {'prices': Series, 'returns': Series}.
                        'returns' is optional and will be computed from prices if missing.

        Returns:
            DataFrame with risk metrics for each symbol, sorted by risk.

        Example:
            >>> data = {
            ...     'AAPL': {'prices': aapl_prices},
            ...     'MSFT': {'prices': msft_prices},
            ...     'TSLA': {'prices': tsla_prices}
            ... }
            >>> comparison = report.compare_symbols(data)
            >>> print(comparison.to_string())
        """
        metrics_list = []

        for symbol, data in symbol_data.items():
            prices = data.get('prices')
            returns = data.get('returns')

            if prices is None:
                continue

            try:
                metrics = self.generate_report(prices, returns, symbol)
                metrics_list.append(metrics)
            except Exception as e:
                # Log error but continue with other symbols
                print(f"Warning: Failed to generate report for {symbol}: {e}")

        if not metrics_list:
            return pd.DataFrame()

        df = self.to_dataframe(metrics_list)

        # Sort by risk (max drawdown descending)
        df = df.sort_values('max_drawdown', ascending=False)

        return df

    def risk_ranking(
        self,
        symbol_data: Dict[str, Dict[str, pd.Series]],
        metric: str = 'var_95'
    ) -> pd.DataFrame:
        """
        Rank symbols by a specific risk metric.

        Args:
            symbol_data: Dictionary mapping symbol to price/return data.
            metric: Metric to rank by. Options: 'var_95', 'var_99', 'max_drawdown',
                   'volatility_21d', 'volatility_63d'.

        Returns:
            DataFrame with symbols ranked by the specified metric.
        """
        df = self.compare_symbols(symbol_data)

        if df.empty:
            return df

        # Higher values = more risk for these metrics
        ascending = metric in ['calmar_ratio']  # Lower Calmar = more risk

        df = df.sort_values(metric, ascending=ascending)
        df['risk_rank'] = range(1, len(df) + 1)

        return df[['symbol', metric, 'risk_rank']]

    def generate_full_report(
        self,
        prices: pd.Series,
        returns: Optional[pd.Series] = None,
        symbol: str = "UNKNOWN"
    ) -> Dict[str, Any]:
        """
        Generate complete risk report with all details.

        Returns a dictionary with comprehensive risk analysis
        suitable for reporting or dashboard display.

        Args:
            prices: Price series.
            returns: Return series (optional).
            symbol: Asset symbol.

        Returns:
            Dictionary with categorized risk metrics.
        """
        if returns is None:
            returns = prices.pct_change().dropna()

        # Get basic metrics
        metrics = self.generate_report(prices, returns, symbol)

        # Get detailed VaR analysis
        var_df = self.var_calc.calculate_all(returns)

        # Get volatility term structure
        vol_term = self.vol_forecaster.volatility_term_structure(returns)

        # Get drawdown periods
        dd_periods = self.dd_analyzer.underwater_periods(prices)
        top_drawdowns = sorted(dd_periods, key=lambda x: x.drawdown, reverse=True)[:5]

        return {
            'summary': {
                'symbol': symbol,
                'timestamp': metrics.timestamp.isoformat(),
                'current_price': float(prices.iloc[-1]),
                'volatility_regime': metrics.volatility_regime
            },
            'var_analysis': {
                'var_95_daily': metrics.var_95,
                'var_99_daily': metrics.var_99,
                'cvar_95_daily': metrics.cvar_95,
                'cvar_99_daily': metrics.cvar_99,
                'var_95_annualized': metrics.var_95 * np.sqrt(252),
                'var_99_annualized': metrics.var_99 * np.sqrt(252),
                'detailed_var': var_df.to_dict('records')
            },
            'volatility_analysis': {
                'volatility_21d': metrics.volatility_21d,
                'volatility_63d': metrics.volatility_63d,
                'garch_forecast_5d': metrics.garch_forecast,
                'term_structure': vol_term.to_dict('records')
            },
            'drawdown_analysis': {
                'max_drawdown': metrics.max_drawdown,
                'current_drawdown': metrics.current_drawdown,
                'calmar_ratio': metrics.calmar_ratio,
                'top_drawdowns': [
                    {
                        'start_date': dd.start_date.isoformat() if dd.start_date else None,
                        'trough_date': dd.trough_date.isoformat() if dd.trough_date else None,
                        'drawdown': dd.drawdown,
                        'duration_days': dd.duration_days,
                        'recovered': dd.is_recovered
                    }
                    for dd in top_drawdowns
                ]
            },
            'risk_scores': {
                'overall_risk_score': self._calculate_risk_score(metrics),
                'var_score': min(metrics.var_99 * 100, 10),  # Scale to 0-10
                'volatility_score': min(metrics.volatility_21d * 10, 10),
                'drawdown_score': min(metrics.max_drawdown * 10, 10)
            }
        }

    def _calculate_risk_score(self, metrics: RiskMetrics) -> float:
        """
        Calculate composite risk score (0-10 scale).

        Higher score = higher risk.
        """
        # Weighted combination of risk factors
        var_component = min(metrics.var_99 * 50, 3.33)  # Max 3.33
        vol_component = min(metrics.volatility_21d * 5, 3.33)  # Max 3.33
        dd_component = min(metrics.max_drawdown * 10, 3.34)  # Max 3.34

        return var_component + vol_component + dd_component

    def export_report(
        self,
        metrics: Union[RiskMetrics, List[RiskMetrics]],
        filepath: str,
        format: str = 'csv'
    ) -> None:
        """
        Export risk report to file.

        Args:
            metrics: RiskMetrics or list of RiskMetrics.
            filepath: Output file path.
            format: Export format ('csv' or 'json').
        """
        df = self.to_dataframe(metrics)

        if format == 'csv':
            df.to_csv(filepath, index=False)
        elif format == 'json':
            df.to_json(filepath, orient='records', date_format='iso')
        else:
            raise ValueError(f"Unsupported format: {format}")
