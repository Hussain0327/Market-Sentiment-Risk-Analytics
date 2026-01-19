"""
Value at Risk (VaR) Calculator.

Provides multiple VaR calculation methods:
- Historical VaR: Percentile-based from historical returns
- Parametric VaR: Assumes normal distribution
- Monte Carlo VaR: Simulation-based estimation
- Expected Shortfall (CVaR): Average loss beyond VaR threshold
"""

from dataclasses import dataclass
from typing import Optional, Union, List
import warnings

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class VaRResult:
    """
    Result of a VaR calculation.

    Attributes:
        var: Value at Risk (as positive number representing loss)
        confidence: Confidence level used (e.g., 0.95, 0.99)
        method: Calculation method used
        expected_shortfall: Average loss beyond VaR (CVaR)
        n_observations: Number of observations used
        annualized_var: VaR annualized (multiplied by sqrt(252))
    """
    var: float
    confidence: float
    method: str
    expected_shortfall: Optional[float] = None
    n_observations: int = 0
    annualized_var: Optional[float] = None


class VaRCalculator:
    """
    Value at Risk calculator supporting multiple methodologies.

    Features:
    - Historical, parametric, and Monte Carlo VaR
    - Expected Shortfall (CVaR) calculation
    - Configurable confidence levels
    - Annualization support

    Example:
        >>> calculator = VaRCalculator()
        >>> import pandas as pd
        >>> returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        >>> var_95 = calculator.historical_var(returns, 0.95)
        >>> print(f"95% VaR: {var_95:.4f}")
    """

    DEFAULT_CONFIDENCE_LEVELS = [0.95, 0.99]
    DEFAULT_MC_SIMULATIONS = 10000
    TRADING_DAYS_PER_YEAR = 252

    def __init__(
        self,
        confidence_levels: Optional[List[float]] = None,
        annualize: bool = True
    ):
        """
        Initialize the VaR calculator.

        Args:
            confidence_levels: List of confidence levels to calculate.
                               Default: [0.95, 0.99]
            annualize: Whether to include annualized VaR in results.
        """
        self.confidence_levels = confidence_levels or self.DEFAULT_CONFIDENCE_LEVELS
        self.annualize = annualize

    def _validate_returns(
        self,
        returns: Union[pd.Series, np.ndarray]
    ) -> np.ndarray:
        """Validate and convert returns to numpy array."""
        if isinstance(returns, pd.Series):
            returns = returns.dropna().values
        else:
            returns = np.asarray(returns)
            returns = returns[~np.isnan(returns)]

        if len(returns) < 10:
            raise ValueError(
                f"Insufficient data: {len(returns)} observations. "
                "Need at least 10 for VaR calculation."
            )

        return returns

    def historical_var(
        self,
        returns: Union[pd.Series, np.ndarray],
        confidence: float = 0.95
    ) -> float:
        """
        Calculate Historical VaR using percentile method.

        Historical VaR is the empirical quantile of the return distribution.
        It makes no assumptions about the distribution shape.

        Args:
            returns: Series or array of returns (daily returns).
            confidence: Confidence level (e.g., 0.95 for 95% VaR).

        Returns:
            VaR as a positive number representing potential loss.

        Example:
            >>> var = calculator.historical_var(returns, 0.95)
            >>> print(f"95% chance daily loss won't exceed {var:.2%}")
        """
        returns = self._validate_returns(returns)

        # VaR is the negative of the (1-confidence) quantile
        # e.g., 95% VaR = -5th percentile
        var = -np.percentile(returns, (1 - confidence) * 100)

        return float(var)

    def parametric_var(
        self,
        returns: Union[pd.Series, np.ndarray],
        confidence: float = 0.95
    ) -> float:
        """
        Calculate Parametric (Gaussian) VaR.

        Assumes returns follow a normal distribution. Uses mean and
        standard deviation to estimate the VaR threshold.

        Args:
            returns: Series or array of returns.
            confidence: Confidence level.

        Returns:
            VaR as a positive number representing potential loss.

        Note:
            May underestimate risk if returns have fat tails.
        """
        returns = self._validate_returns(returns)

        mean = np.mean(returns)
        std = np.std(returns, ddof=1)

        # z-score for the confidence level
        z = stats.norm.ppf(1 - confidence)

        # VaR = -(mean + z * std)
        var = -(mean + z * std)

        return float(var)

    def monte_carlo_var(
        self,
        returns: Union[pd.Series, np.ndarray],
        confidence: float = 0.95,
        simulations: int = DEFAULT_MC_SIMULATIONS,
        seed: Optional[int] = None
    ) -> float:
        """
        Calculate Monte Carlo VaR via simulation.

        Generates simulated returns based on historical distribution
        parameters and calculates VaR from the simulated distribution.

        Args:
            returns: Series or array of historical returns.
            confidence: Confidence level.
            simulations: Number of Monte Carlo simulations.
            seed: Random seed for reproducibility.

        Returns:
            VaR as a positive number representing potential loss.
        """
        returns = self._validate_returns(returns)

        if seed is not None:
            np.random.seed(seed)

        mean = np.mean(returns)
        std = np.std(returns, ddof=1)

        # Generate simulated returns
        simulated_returns = np.random.normal(mean, std, simulations)

        # Calculate VaR from simulated distribution
        var = -np.percentile(simulated_returns, (1 - confidence) * 100)

        return float(var)

    def expected_shortfall(
        self,
        returns: Union[pd.Series, np.ndarray],
        confidence: float = 0.95
    ) -> float:
        """
        Calculate Expected Shortfall (Conditional VaR).

        Expected Shortfall is the average loss in the worst
        (1-confidence)% of cases. Also known as CVaR or Average VaR.

        Args:
            returns: Series or array of returns.
            confidence: Confidence level.

        Returns:
            Expected Shortfall as a positive number.

        Example:
            >>> es = calculator.expected_shortfall(returns, 0.95)
            >>> print(f"Average loss in worst 5% of days: {es:.2%}")
        """
        returns = self._validate_returns(returns)

        # Find the VaR threshold
        var_threshold = np.percentile(returns, (1 - confidence) * 100)

        # Average of returns below the threshold
        tail_returns = returns[returns <= var_threshold]

        if len(tail_returns) == 0:
            # Fallback if no returns below threshold
            return self.historical_var(returns, confidence)

        es = -np.mean(tail_returns)

        return float(es)

    def calculate_var_result(
        self,
        returns: Union[pd.Series, np.ndarray],
        confidence: float,
        method: str = "historical"
    ) -> VaRResult:
        """
        Calculate VaR and return detailed result object.

        Args:
            returns: Series or array of returns.
            confidence: Confidence level.
            method: 'historical', 'parametric', or 'monte_carlo'.

        Returns:
            VaRResult with VaR, ES, and metadata.
        """
        returns_arr = self._validate_returns(returns)

        # Calculate VaR based on method
        if method == "historical":
            var = self.historical_var(returns_arr, confidence)
        elif method == "parametric":
            var = self.parametric_var(returns_arr, confidence)
        elif method == "monte_carlo":
            var = self.monte_carlo_var(returns_arr, confidence)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Calculate Expected Shortfall
        es = self.expected_shortfall(returns_arr, confidence)

        # Annualize if requested
        annualized = None
        if self.annualize:
            annualized = var * np.sqrt(self.TRADING_DAYS_PER_YEAR)

        return VaRResult(
            var=var,
            confidence=confidence,
            method=method,
            expected_shortfall=es,
            n_observations=len(returns_arr),
            annualized_var=annualized
        )

    def calculate_all(
        self,
        returns: Union[pd.Series, np.ndarray],
        methods: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Calculate VaR using all methods and confidence levels.

        Args:
            returns: Series or array of returns.
            methods: List of methods to use.
                     Default: ['historical', 'parametric', 'monte_carlo']

        Returns:
            DataFrame with VaR metrics for each method/confidence combination.

        Example:
            >>> df = calculator.calculate_all(returns)
            >>> print(df.to_string())
        """
        if methods is None:
            methods = ["historical", "parametric", "monte_carlo"]

        results = []

        for method in methods:
            for confidence in self.confidence_levels:
                try:
                    result = self.calculate_var_result(returns, confidence, method)
                    results.append({
                        "method": method,
                        "confidence": f"{confidence:.0%}",
                        "var": result.var,
                        "expected_shortfall": result.expected_shortfall,
                        "annualized_var": result.annualized_var,
                        "n_observations": result.n_observations
                    })
                except Exception as e:
                    warnings.warn(f"Failed to calculate {method} VaR: {e}")

        return pd.DataFrame(results)

    def rolling_var(
        self,
        returns: Union[pd.Series, np.ndarray],
        window: int = 63,
        confidence: float = 0.95,
        method: str = "historical"
    ) -> pd.Series:
        """
        Calculate rolling VaR over a window.

        Args:
            returns: Series of returns (must be pandas Series for rolling).
            window: Rolling window size in days.
            confidence: Confidence level.
            method: VaR calculation method.

        Returns:
            Series of rolling VaR values.
        """
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)

        returns = returns.dropna()

        def var_func(x):
            if len(x) < 10:
                return np.nan
            if method == "historical":
                return self.historical_var(x, confidence)
            elif method == "parametric":
                return self.parametric_var(x, confidence)
            else:
                return self.monte_carlo_var(x, confidence)

        rolling_var = returns.rolling(window=window).apply(var_func, raw=True)

        return rolling_var

    def var_backtest(
        self,
        returns: Union[pd.Series, np.ndarray],
        var_series: Union[pd.Series, np.ndarray],
        confidence: float = 0.95
    ) -> dict:
        """
        Backtest VaR predictions by counting exceedances.

        Args:
            returns: Actual returns.
            var_series: Predicted VaR values (positive numbers).
            confidence: Confidence level used for VaR.

        Returns:
            Dictionary with backtest statistics:
            - exceedances: Number of times loss exceeded VaR
            - expected_exceedances: Expected number based on confidence
            - exceedance_ratio: Actual/Expected ratio
            - p_value: Kupiec test p-value
        """
        if isinstance(returns, pd.Series):
            returns = returns.values
        if isinstance(var_series, pd.Series):
            var_series = var_series.values

        # Align lengths
        min_len = min(len(returns), len(var_series))
        returns = returns[:min_len]
        var_series = var_series[:min_len]

        # Remove NaN values
        mask = ~(np.isnan(returns) | np.isnan(var_series))
        returns = returns[mask]
        var_series = var_series[mask]

        n = len(returns)
        if n == 0:
            return {"error": "No valid observations"}

        # Count exceedances (when loss > VaR)
        exceedances = np.sum(-returns > var_series)

        expected = n * (1 - confidence)
        ratio = exceedances / expected if expected > 0 else np.inf

        # Kupiec likelihood ratio test
        p = exceedances / n if n > 0 else 0
        expected_p = 1 - confidence

        if 0 < p < 1:
            lr_stat = -2 * (
                n * np.log(1 - expected_p) +
                exceedances * np.log(expected_p / (1 - expected_p)) -
                (n - exceedances) * np.log(1 - p) -
                exceedances * np.log(p)
            )
            p_value = 1 - stats.chi2.cdf(lr_stat, 1)
        else:
            p_value = 0.0 if p > expected_p else 1.0

        return {
            "n_observations": n,
            "exceedances": int(exceedances),
            "expected_exceedances": expected,
            "exceedance_ratio": ratio,
            "p_value": p_value,
            "test_passed": p_value > 0.05
        }
