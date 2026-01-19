"""
Drawdown Analyzer.

Provides comprehensive drawdown analysis:
- Drawdown calculation and tracking
- Maximum drawdown identification
- Drawdown duration analysis
- Recovery time estimation
- Underwater period detection
"""

from dataclasses import dataclass
from typing import Optional, Union, List
from datetime import datetime

import numpy as np
import pandas as pd


@dataclass
class DrawdownPeriod:
    """
    Represents a single drawdown period.

    Attributes:
        start_date: Date when drawdown started (peak)
        trough_date: Date of maximum drawdown
        end_date: Date when recovered (or None if ongoing)
        peak_value: Price/value at peak
        trough_value: Price/value at trough
        drawdown: Maximum drawdown during period (as positive decimal)
        duration_days: Total days in drawdown
        recovery_days: Days from trough to recovery (or None if ongoing)
        is_recovered: Whether the drawdown has recovered
    """
    start_date: datetime
    trough_date: datetime
    end_date: Optional[datetime]
    peak_value: float
    trough_value: float
    drawdown: float
    duration_days: int
    recovery_days: Optional[int]
    is_recovered: bool


class DrawdownAnalyzer:
    """
    Comprehensive drawdown analysis toolkit.

    Features:
    - Real-time drawdown tracking
    - Historical max drawdown calculation
    - Recovery analysis
    - Underwater period identification

    Example:
        >>> analyzer = DrawdownAnalyzer()
        >>> prices = pd.read_csv('prices.csv')['Close']
        >>> dd_df = analyzer.calculate_drawdown(prices)
        >>> max_dd = analyzer.max_drawdown(prices)
        >>> print(f"Maximum drawdown: {max_dd:.2%}")
    """

    def __init__(self):
        """Initialize the drawdown analyzer."""
        pass

    def _validate_prices(
        self,
        prices: Union[pd.Series, np.ndarray]
    ) -> pd.Series:
        """Validate and convert prices to pandas Series."""
        if isinstance(prices, np.ndarray):
            prices = pd.Series(prices)

        prices = prices.dropna()

        if len(prices) < 2:
            raise ValueError(
                f"Insufficient data: {len(prices)} observations. "
                "Need at least 2 for drawdown calculation."
            )

        if (prices <= 0).any():
            raise ValueError("Prices must be positive for drawdown calculation.")

        return prices

    def calculate_drawdown(
        self,
        prices: Union[pd.Series, np.ndarray]
    ) -> pd.DataFrame:
        """
        Calculate drawdown series from prices.

        Returns a DataFrame with:
        - prices: Original price series
        - peak: Running maximum (high water mark)
        - drawdown: Current drawdown from peak (as positive decimal)
        - drawdown_pct: Drawdown as percentage string

        Args:
            prices: Series or array of prices.

        Returns:
            DataFrame with drawdown analysis columns.

        Example:
            >>> dd_df = analyzer.calculate_drawdown(prices)
            >>> current_dd = dd_df['drawdown'].iloc[-1]
            >>> print(f"Current drawdown: {current_dd:.2%}")
        """
        prices = self._validate_prices(prices)

        # Calculate running maximum (peak/high water mark)
        peak = prices.cummax()

        # Drawdown is the decline from peak (as positive number)
        drawdown = (peak - prices) / peak

        df = pd.DataFrame({
            'prices': prices,
            'peak': peak,
            'drawdown': drawdown
        })

        df.index = prices.index

        return df

    def max_drawdown(
        self,
        prices: Union[pd.Series, np.ndarray]
    ) -> float:
        """
        Calculate maximum drawdown.

        Maximum drawdown is the largest peak-to-trough decline
        in the entire series.

        Args:
            prices: Series or array of prices.

        Returns:
            Maximum drawdown as positive decimal (e.g., 0.25 for 25%).

        Example:
            >>> mdd = analyzer.max_drawdown(prices)
            >>> print(f"Max drawdown: {mdd:.2%}")
        """
        dd_df = self.calculate_drawdown(prices)
        return float(dd_df['drawdown'].max())

    def drawdown_duration(
        self,
        prices: Union[pd.Series, np.ndarray]
    ) -> dict:
        """
        Calculate drawdown duration statistics.

        Returns the duration of the current drawdown and
        the maximum drawdown duration in the series.

        Args:
            prices: Series or array of prices.

        Returns:
            Dictionary with:
            - current_duration: Days in current drawdown (0 if at peak)
            - max_duration: Longest drawdown period in days
            - is_in_drawdown: Whether currently in a drawdown
        """
        dd_df = self.calculate_drawdown(prices)
        drawdown = dd_df['drawdown']

        # Find periods where we're in a drawdown
        in_drawdown = drawdown > 0

        # Current duration
        if not in_drawdown.iloc[-1]:
            current_duration = 0
        else:
            # Count backwards from end to find last peak
            current_duration = 0
            for i in range(len(in_drawdown) - 1, -1, -1):
                if in_drawdown.iloc[i]:
                    current_duration += 1
                else:
                    break

        # Maximum duration - find longest consecutive drawdown period
        max_duration = 0
        current_streak = 0

        for val in in_drawdown:
            if val:
                current_streak += 1
                max_duration = max(max_duration, current_streak)
            else:
                current_streak = 0

        return {
            'current_duration': current_duration,
            'max_duration': max_duration,
            'is_in_drawdown': bool(in_drawdown.iloc[-1])
        }

    def recovery_time(
        self,
        prices: Union[pd.Series, np.ndarray]
    ) -> dict:
        """
        Analyze recovery times from drawdowns.

        Args:
            prices: Series or array of prices.

        Returns:
            Dictionary with:
            - avg_recovery_days: Average days to recover from drawdowns
            - max_recovery_days: Longest recovery time
            - total_recoveries: Number of completed recoveries
            - current_recovery_days: Days since max drawdown trough (if recovering)
        """
        prices = self._validate_prices(prices)
        dd_df = self.calculate_drawdown(prices)

        periods = self.underwater_periods(prices)
        recovered_periods = [p for p in periods if p.is_recovered]

        if not recovered_periods:
            return {
                'avg_recovery_days': None,
                'max_recovery_days': None,
                'total_recoveries': 0,
                'current_recovery_days': None
            }

        recovery_times = [p.recovery_days for p in recovered_periods if p.recovery_days]

        # Check if currently in recovery
        current_recovery = None
        ongoing = [p for p in periods if not p.is_recovered]
        if ongoing:
            latest = ongoing[-1]
            if latest.trough_date:
                # Days since trough
                if hasattr(dd_df.index, 'to_pydatetime'):
                    last_date = dd_df.index[-1]
                    if hasattr(last_date, 'to_pydatetime'):
                        last_date = last_date.to_pydatetime()
                    current_recovery = (last_date - latest.trough_date).days

        return {
            'avg_recovery_days': np.mean(recovery_times) if recovery_times else None,
            'max_recovery_days': max(recovery_times) if recovery_times else None,
            'total_recoveries': len(recovered_periods),
            'current_recovery_days': current_recovery
        }

    def underwater_periods(
        self,
        prices: Union[pd.Series, np.ndarray],
        min_drawdown: float = 0.01
    ) -> List[DrawdownPeriod]:
        """
        Identify all underwater (drawdown) periods.

        An underwater period starts when price drops below a previous
        peak and ends when a new peak is reached.

        Args:
            prices: Series or array of prices.
            min_drawdown: Minimum drawdown to report (default 1%).

        Returns:
            List of DrawdownPeriod objects sorted by start date.

        Example:
            >>> periods = analyzer.underwater_periods(prices)
            >>> for p in periods[:3]:
            ...     print(f"{p.start_date}: {p.drawdown:.2%} over {p.duration_days} days")
        """
        prices = self._validate_prices(prices)
        dd_df = self.calculate_drawdown(prices)

        drawdown = dd_df['drawdown']
        price_values = dd_df['prices']
        peak_values = dd_df['peak']

        periods = []
        in_period = False
        period_start = None
        period_start_val = None
        period_trough = None
        period_trough_val = None
        period_trough_dd = 0

        for i, (idx, dd) in enumerate(drawdown.items()):
            if dd > 0 and not in_period:
                # Starting a new drawdown period
                in_period = True
                # Find the previous peak (could be same index or previous)
                if i > 0:
                    period_start = drawdown.index[i - 1]
                    period_start_val = float(peak_values.iloc[i])
                else:
                    period_start = idx
                    period_start_val = float(price_values.iloc[i])
                period_trough = idx
                period_trough_val = float(price_values.loc[idx])
                period_trough_dd = dd

            elif dd > 0 and in_period:
                # Continue in drawdown - track trough
                if dd > period_trough_dd:
                    period_trough = idx
                    period_trough_val = float(price_values.loc[idx])
                    period_trough_dd = dd

            elif dd == 0 and in_period:
                # Recovered - close the period
                in_period = False

                if period_trough_dd >= min_drawdown:
                    # Convert timestamps to datetime
                    start_dt = self._to_datetime(period_start)
                    trough_dt = self._to_datetime(period_trough)
                    end_dt = self._to_datetime(idx)

                    duration = (end_dt - start_dt).days if start_dt and end_dt else 0
                    recovery = (end_dt - trough_dt).days if trough_dt and end_dt else 0

                    periods.append(DrawdownPeriod(
                        start_date=start_dt,
                        trough_date=trough_dt,
                        end_date=end_dt,
                        peak_value=period_start_val,
                        trough_value=period_trough_val,
                        drawdown=period_trough_dd,
                        duration_days=duration,
                        recovery_days=recovery,
                        is_recovered=True
                    ))

                period_start = None
                period_trough = None
                period_trough_dd = 0

        # Handle ongoing drawdown
        if in_period and period_trough_dd >= min_drawdown:
            start_dt = self._to_datetime(period_start)
            trough_dt = self._to_datetime(period_trough)
            end_dt = self._to_datetime(drawdown.index[-1])

            duration = (end_dt - start_dt).days if start_dt and end_dt else 0

            periods.append(DrawdownPeriod(
                start_date=start_dt,
                trough_date=trough_dt,
                end_date=None,
                peak_value=period_start_val,
                trough_value=period_trough_val,
                drawdown=period_trough_dd,
                duration_days=duration,
                recovery_days=None,
                is_recovered=False
            ))

        return periods

    def _to_datetime(self, ts) -> Optional[datetime]:
        """Convert various timestamp types to datetime."""
        if ts is None:
            return None
        if isinstance(ts, datetime):
            return ts
        if hasattr(ts, 'to_pydatetime'):
            return ts.to_pydatetime()
        if isinstance(ts, str):
            return pd.to_datetime(ts).to_pydatetime()
        if isinstance(ts, (int, float)):
            return datetime.fromordinal(int(ts))
        return None

    def calmar_ratio(
        self,
        prices: Union[pd.Series, np.ndarray],
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Calmar ratio (annualized return / max drawdown).

        The Calmar ratio measures risk-adjusted performance using
        maximum drawdown as the risk measure.

        Args:
            prices: Series or array of prices.
            periods_per_year: Number of periods per year (252 for daily).

        Returns:
            Calmar ratio (higher is better).

        Example:
            >>> calmar = analyzer.calmar_ratio(prices)
            >>> print(f"Calmar ratio: {calmar:.2f}")
        """
        prices = self._validate_prices(prices)

        # Calculate annualized return
        total_return = prices.iloc[-1] / prices.iloc[0] - 1
        n_periods = len(prices)
        annualized_return = (1 + total_return) ** (periods_per_year / n_periods) - 1

        # Get max drawdown
        mdd = self.max_drawdown(prices)

        if mdd == 0:
            return np.inf if annualized_return > 0 else 0

        return annualized_return / mdd

    def ulcer_index(
        self,
        prices: Union[pd.Series, np.ndarray]
    ) -> float:
        """
        Calculate Ulcer Index.

        The Ulcer Index measures downside volatility based on
        drawdown depth and duration. Higher values indicate more pain.

        Args:
            prices: Series or array of prices.

        Returns:
            Ulcer Index value.
        """
        dd_df = self.calculate_drawdown(prices)
        drawdown_pct = dd_df['drawdown'] * 100  # Convert to percentage

        # Ulcer Index = sqrt(mean(drawdown^2))
        ulcer = np.sqrt(np.mean(drawdown_pct ** 2))

        return float(ulcer)

    def pain_index(
        self,
        prices: Union[pd.Series, np.ndarray]
    ) -> float:
        """
        Calculate Pain Index.

        The Pain Index is the average drawdown over the period.
        Similar to Ulcer Index but uses mean instead of RMS.

        Args:
            prices: Series or array of prices.

        Returns:
            Pain Index value.
        """
        dd_df = self.calculate_drawdown(prices)
        return float(dd_df['drawdown'].mean())

    def drawdown_summary(
        self,
        prices: Union[pd.Series, np.ndarray]
    ) -> dict:
        """
        Generate comprehensive drawdown summary.

        Args:
            prices: Series or array of prices.

        Returns:
            Dictionary with all drawdown metrics.
        """
        prices = self._validate_prices(prices)
        dd_df = self.calculate_drawdown(prices)
        duration_info = self.drawdown_duration(prices)
        recovery_info = self.recovery_time(prices)
        periods = self.underwater_periods(prices)

        return {
            'current_drawdown': float(dd_df['drawdown'].iloc[-1]),
            'max_drawdown': self.max_drawdown(prices),
            'avg_drawdown': float(dd_df['drawdown'].mean()),
            'current_duration_days': duration_info['current_duration'],
            'max_duration_days': duration_info['max_duration'],
            'is_in_drawdown': duration_info['is_in_drawdown'],
            'total_underwater_periods': len(periods),
            'recovered_periods': len([p for p in periods if p.is_recovered]),
            'avg_recovery_days': recovery_info['avg_recovery_days'],
            'max_recovery_days': recovery_info['max_recovery_days'],
            'calmar_ratio': self.calmar_ratio(prices),
            'ulcer_index': self.ulcer_index(prices),
            'pain_index': self.pain_index(prices)
        }
