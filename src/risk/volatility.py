"""
Volatility Forecaster.

Provides multiple volatility estimation and forecasting methods:
- Realized volatility (rolling standard deviation)
- EWMA volatility (exponentially weighted)
- GARCH(1,1) volatility forecasting
- Volatility regime classification
"""

from dataclasses import dataclass
from typing import Optional, Union, List
import warnings

import numpy as np
import pandas as pd


@dataclass
class GARCHResult:
    """
    Result of GARCH model fitting and forecasting.

    Attributes:
        conditional_volatility: Series of fitted conditional volatilities
        forecast: Forecasted volatility values
        forecast_horizon: Number of periods forecasted
        omega: GARCH constant term
        alpha: ARCH coefficient (lag-1 squared return)
        beta: GARCH coefficient (lag-1 variance)
        persistence: alpha + beta (should be < 1 for stationarity)
        log_likelihood: Model log-likelihood
        aic: Akaike Information Criterion
        bic: Bayesian Information Criterion
        converged: Whether optimization converged
    """
    conditional_volatility: pd.Series
    forecast: np.ndarray
    forecast_horizon: int
    omega: float
    alpha: float
    beta: float
    persistence: float
    log_likelihood: float
    aic: float
    bic: float
    converged: bool


class VolatilityForecaster:
    """
    Volatility estimation and forecasting toolkit.

    Features:
    - Multiple volatility estimation methods
    - GARCH(1,1) model fitting and forecasting
    - Volatility regime classification
    - Annualization support

    Example:
        >>> forecaster = VolatilityForecaster()
        >>> returns = prices['Close'].pct_change().dropna()
        >>> ewma_vol = forecaster.ewma_volatility(returns, span=21)
        >>> garch_result = forecaster.garch_forecast(returns, horizon=5)
    """

    TRADING_DAYS_PER_YEAR = 252

    def __init__(
        self,
        annualization_factor: int = TRADING_DAYS_PER_YEAR
    ):
        """
        Initialize the volatility forecaster.

        Args:
            annualization_factor: Number of periods per year for annualization.
                                  Default: 252 (trading days).
        """
        self.annualization_factor = annualization_factor

    def _validate_returns(
        self,
        returns: Union[pd.Series, np.ndarray]
    ) -> pd.Series:
        """Validate and convert returns to pandas Series."""
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)

        returns = returns.dropna()

        if len(returns) < 20:
            raise ValueError(
                f"Insufficient data: {len(returns)} observations. "
                "Need at least 20 for volatility estimation."
            )

        return returns

    def realized_volatility(
        self,
        returns: Union[pd.Series, np.ndarray],
        window: int = 21,
        annualize: bool = True
    ) -> pd.Series:
        """
        Calculate realized volatility using rolling standard deviation.

        This is the simplest volatility estimator using a rolling window
        of historical returns.

        Args:
            returns: Series or array of returns.
            window: Rolling window size in periods. Default: 21 (1 month).
            annualize: Whether to annualize the volatility.

        Returns:
            Series of rolling volatility values.

        Example:
            >>> vol = forecaster.realized_volatility(returns, window=21)
            >>> print(f"Current 21-day vol: {vol.iloc[-1]:.2%}")
        """
        returns = self._validate_returns(returns)

        vol = returns.rolling(window=window).std()

        if annualize:
            vol = vol * np.sqrt(self.annualization_factor)

        return vol

    def ewma_volatility(
        self,
        returns: Union[pd.Series, np.ndarray],
        span: int = 21,
        annualize: bool = True
    ) -> pd.Series:
        """
        Calculate EWMA (Exponentially Weighted Moving Average) volatility.

        EWMA gives more weight to recent observations, making it more
        responsive to recent market conditions than simple rolling volatility.

        Args:
            returns: Series or array of returns.
            span: Span for exponential weighting (like a window size).
            annualize: Whether to annualize the volatility.

        Returns:
            Series of EWMA volatility values.

        Note:
            The decay factor lambda = (span-1)/(span+1)
            Common values: span=21 -> lambda=0.905, span=63 -> lambda=0.969
        """
        returns = self._validate_returns(returns)

        # EWMA variance
        ewma_var = returns.ewm(span=span, adjust=False).var()

        # EWMA volatility (std dev)
        vol = np.sqrt(ewma_var)

        if annualize:
            vol = vol * np.sqrt(self.annualization_factor)

        return vol

    def parkinson_volatility(
        self,
        high: Union[pd.Series, np.ndarray],
        low: Union[pd.Series, np.ndarray],
        window: int = 21,
        annualize: bool = True
    ) -> pd.Series:
        """
        Calculate Parkinson volatility using high-low range.

        Parkinson volatility uses the daily high-low range, which captures
        intraday volatility that close-to-close returns miss.

        Args:
            high: Series of daily high prices.
            low: Series of daily low prices.
            window: Rolling window size.
            annualize: Whether to annualize.

        Returns:
            Series of Parkinson volatility estimates.

        Note:
            Typically 20-30% more efficient than close-to-close volatility.
        """
        if isinstance(high, np.ndarray):
            high = pd.Series(high)
        if isinstance(low, np.ndarray):
            low = pd.Series(low)

        # Parkinson estimator: sqrt(1/(4*ln(2)) * (ln(H/L))^2)
        log_hl = np.log(high / low)
        parkinson_var = log_hl ** 2 / (4 * np.log(2))

        vol = np.sqrt(parkinson_var.rolling(window=window).mean())

        if annualize:
            vol = vol * np.sqrt(self.annualization_factor)

        return vol

    def garch_forecast(
        self,
        returns: Union[pd.Series, np.ndarray],
        horizon: int = 5,
        p: int = 1,
        q: int = 1
    ) -> GARCHResult:
        """
        Fit GARCH(p,q) model and generate volatility forecasts.

        GARCH models capture volatility clustering - the tendency for
        high volatility to follow high volatility.

        Args:
            returns: Series or array of returns.
            horizon: Forecast horizon in periods.
            p: Order of GARCH terms (lag variances).
            q: Order of ARCH terms (lag squared returns).

        Returns:
            GARCHResult with fitted model and forecasts.

        Example:
            >>> result = forecaster.garch_forecast(returns, horizon=5)
            >>> print(f"5-day vol forecast: {result.forecast}")
            >>> print(f"Persistence: {result.persistence:.3f}")
        """
        returns = self._validate_returns(returns)

        try:
            from arch import arch_model

            # Scale returns to percentage for numerical stability
            returns_pct = returns * 100

            # Fit GARCH model
            model = arch_model(
                returns_pct,
                vol='Garch',
                p=p,
                q=q,
                mean='Constant',
                rescale=False
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = model.fit(disp='off', show_warning=False)

            # Extract parameters
            omega = result.params.get('omega', 0)
            alpha = result.params.get('alpha[1]', 0)
            beta = result.params.get('beta[1]', 0)
            persistence = alpha + beta

            # Generate forecasts
            forecast = result.forecast(horizon=horizon)
            forecast_variance = forecast.variance.iloc[-1].values
            forecast_vol = np.sqrt(forecast_variance) / 100  # Scale back

            # Conditional volatility (fitted values)
            cond_vol = result.conditional_volatility / 100  # Scale back
            cond_vol.index = returns.index

            return GARCHResult(
                conditional_volatility=cond_vol,
                forecast=forecast_vol,
                forecast_horizon=horizon,
                omega=omega,
                alpha=alpha,
                beta=beta,
                persistence=persistence,
                log_likelihood=result.loglikelihood,
                aic=result.aic,
                bic=result.bic,
                converged=result.convergence_flag == 0
            )

        except ImportError:
            raise ImportError(
                "arch library is required for GARCH modeling. "
                "Install with: pip install arch"
            )
        except Exception as e:
            # Return a minimal result on failure
            warnings.warn(f"GARCH fitting failed: {e}. Using EWMA fallback.")
            ewma_vol = self.ewma_volatility(returns, span=21, annualize=False)

            return GARCHResult(
                conditional_volatility=ewma_vol,
                forecast=np.full(horizon, ewma_vol.iloc[-1]),
                forecast_horizon=horizon,
                omega=0,
                alpha=0.1,
                beta=0.85,
                persistence=0.95,
                log_likelihood=0,
                aic=0,
                bic=0,
                converged=False
            )

    def volatility_regime(
        self,
        returns: Union[pd.Series, np.ndarray],
        window: int = 63,
        low_threshold: float = 0.33,
        high_threshold: float = 0.67
    ) -> pd.Series:
        """
        Classify volatility into low/medium/high regimes.

        Uses percentile ranks of rolling volatility to determine
        the current volatility regime.

        Args:
            returns: Series or array of returns.
            window: Window for volatility calculation.
            low_threshold: Percentile below which volatility is 'low'.
            high_threshold: Percentile above which volatility is 'high'.

        Returns:
            Series with regime labels: 'low', 'medium', 'high'.

        Example:
            >>> regimes = forecaster.volatility_regime(returns)
            >>> current_regime = regimes.iloc[-1]
            >>> print(f"Current volatility regime: {current_regime}")
        """
        returns = self._validate_returns(returns)

        # Calculate rolling volatility
        vol = returns.rolling(window=window).std()

        # Calculate percentile rank over expanding window
        def percentile_rank(x):
            if len(x) < window:
                return np.nan
            current = x.iloc[-1]
            return (x < current).sum() / len(x)

        vol_rank = vol.expanding(min_periods=window).apply(
            percentile_rank, raw=False
        )

        # Classify regimes
        def classify(rank):
            if pd.isna(rank):
                return np.nan
            if rank < low_threshold:
                return 'low'
            elif rank > high_threshold:
                return 'high'
            else:
                return 'medium'

        regimes = vol_rank.apply(classify)

        return regimes

    def volatility_cone(
        self,
        returns: Union[pd.Series, np.ndarray],
        windows: Optional[List[int]] = None,
        percentiles: Optional[List[int]] = None,
        annualize: bool = True
    ) -> pd.DataFrame:
        """
        Generate volatility cone showing vol ranges across time horizons.

        A volatility cone shows the range of historical volatilities
        at different lookback windows, useful for term structure analysis.

        Args:
            returns: Series of returns.
            windows: List of window sizes. Default: [5, 10, 21, 63, 126, 252]
            percentiles: Percentiles to calculate. Default: [5, 25, 50, 75, 95]
            annualize: Whether to annualize volatilities.

        Returns:
            DataFrame with volatility percentiles for each window.
        """
        returns = self._validate_returns(returns)

        if windows is None:
            windows = [5, 10, 21, 63, 126, 252]
        if percentiles is None:
            percentiles = [5, 25, 50, 75, 95]

        cone_data = {}

        for window in windows:
            if window > len(returns):
                continue

            vol = returns.rolling(window=window).std().dropna()

            if annualize:
                vol = vol * np.sqrt(self.annualization_factor)

            cone_data[window] = {
                f'p{p}': np.percentile(vol, p) for p in percentiles
            }
            cone_data[window]['current'] = vol.iloc[-1]

        df = pd.DataFrame(cone_data).T
        df.index.name = 'window'

        return df

    def volatility_term_structure(
        self,
        returns: Union[pd.Series, np.ndarray],
        windows: Optional[List[int]] = None,
        annualize: bool = True
    ) -> pd.DataFrame:
        """
        Calculate current volatility term structure.

        Shows current volatility at different time horizons,
        useful for understanding near-term vs long-term vol expectations.

        Args:
            returns: Series of returns.
            windows: List of window sizes.
            annualize: Whether to annualize.

        Returns:
            DataFrame with volatility at each horizon.
        """
        returns = self._validate_returns(returns)

        if windows is None:
            windows = [5, 10, 21, 63, 126, 252]

        term_structure = []

        for window in windows:
            if window > len(returns):
                continue

            vol = returns.iloc[-window:].std()

            if annualize:
                vol = vol * np.sqrt(self.annualization_factor)

            term_structure.append({
                'window': window,
                'volatility': vol,
                'horizon_days': window
            })

        return pd.DataFrame(term_structure)
