"""
Comprehensive test suite for the risk metrics module.

Tests:
- VaRCalculator
- VolatilityForecaster
- DrawdownAnalyzer
- RiskReport
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.risk import (
    VaRCalculator,
    VaRResult,
    VolatilityForecaster,
    GARCHResult,
    DrawdownAnalyzer,
    DrawdownPeriod,
    RiskReport,
    RiskMetrics,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_returns():
    """Generate sample return series for testing."""
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, 252)  # ~12% annual vol
    dates = pd.date_range(start='2024-01-01', periods=252, freq='B')
    return pd.Series(returns, index=dates)


@pytest.fixture
def sample_prices(sample_returns):
    """Generate sample price series from returns."""
    prices = 100 * (1 + sample_returns).cumprod()
    return prices


@pytest.fixture
def trending_down_prices():
    """Price series with clear drawdown."""
    np.random.seed(42)
    # Start at 100, decline to 70, then recover to 90
    prices = [100]
    for i in range(50):
        prices.append(prices[-1] * (1 - 0.008 + np.random.normal(0, 0.005)))
    for i in range(50):
        prices.append(prices[-1] * (1 + 0.003 + np.random.normal(0, 0.005)))
    dates = pd.date_range(start='2024-01-01', periods=101, freq='B')
    return pd.Series(prices, index=dates)


@pytest.fixture
def var_calculator():
    """Create VaR calculator instance."""
    return VaRCalculator(confidence_levels=[0.95, 0.99])


@pytest.fixture
def vol_forecaster():
    """Create volatility forecaster instance."""
    return VolatilityForecaster()


@pytest.fixture
def dd_analyzer():
    """Create drawdown analyzer instance."""
    return DrawdownAnalyzer()


# ============================================================================
# VaRCalculator Tests
# ============================================================================

class TestVaRCalculator:
    """Tests for VaRCalculator class."""

    def test_init_default_confidence(self):
        """Test default confidence levels."""
        calc = VaRCalculator()
        assert calc.confidence_levels == [0.95, 0.99]

    def test_init_custom_confidence(self):
        """Test custom confidence levels."""
        calc = VaRCalculator(confidence_levels=[0.90, 0.95, 0.99])
        assert calc.confidence_levels == [0.90, 0.95, 0.99]

    def test_historical_var(self, var_calculator, sample_returns):
        """Test historical VaR calculation."""
        var_95 = var_calculator.historical_var(sample_returns, 0.95)

        # VaR should be positive
        assert var_95 > 0

        # 95% VaR should be less than 99% VaR
        var_99 = var_calculator.historical_var(sample_returns, 0.99)
        assert var_95 < var_99

    def test_historical_var_known_values(self, var_calculator):
        """Test historical VaR with known distribution."""
        # Create returns where we know the percentiles
        returns = pd.Series(np.linspace(-0.05, 0.05, 100))

        # 95% VaR should be approximately the 5th percentile
        var_95 = var_calculator.historical_var(returns, 0.95)
        expected = 0.05 * 0.95 - 0.05 * 0.05  # Approximately 0.045
        assert abs(var_95 - 0.045) < 0.01

    def test_parametric_var(self, var_calculator, sample_returns):
        """Test parametric (Gaussian) VaR."""
        var_95 = var_calculator.parametric_var(sample_returns, 0.95)

        assert var_95 > 0

        # Parametric VaR should be in reasonable range
        assert 0.01 < var_95 < 0.10

    def test_monte_carlo_var(self, var_calculator, sample_returns):
        """Test Monte Carlo VaR."""
        var_95 = var_calculator.monte_carlo_var(
            sample_returns, 0.95, simulations=10000, seed=42
        )

        assert var_95 > 0

        # MC VaR should be close to parametric VaR for normal returns
        parametric_var = var_calculator.parametric_var(sample_returns, 0.95)
        assert abs(var_95 - parametric_var) < 0.01

    def test_expected_shortfall(self, var_calculator, sample_returns):
        """Test Expected Shortfall (CVaR) calculation."""
        es_95 = var_calculator.expected_shortfall(sample_returns, 0.95)
        var_95 = var_calculator.historical_var(sample_returns, 0.95)

        # ES should be greater than VaR
        assert es_95 >= var_95

        # ES should be positive
        assert es_95 > 0

    def test_calculate_var_result(self, var_calculator, sample_returns):
        """Test VaRResult dataclass creation."""
        result = var_calculator.calculate_var_result(sample_returns, 0.95, "historical")

        assert isinstance(result, VaRResult)
        assert result.confidence == 0.95
        assert result.method == "historical"
        assert result.var > 0
        assert result.expected_shortfall >= result.var
        assert result.n_observations == len(sample_returns)

    def test_calculate_all(self, var_calculator, sample_returns):
        """Test comprehensive VaR calculation."""
        df = var_calculator.calculate_all(sample_returns)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 6  # 3 methods x 2 confidence levels
        assert "var" in df.columns
        assert "expected_shortfall" in df.columns

    def test_rolling_var(self, var_calculator, sample_returns):
        """Test rolling VaR calculation."""
        rolling_var = var_calculator.rolling_var(sample_returns, window=63)

        assert isinstance(rolling_var, pd.Series)
        # First 62 values should be NaN
        assert rolling_var.iloc[:62].isna().all()
        # Rest should be valid
        assert rolling_var.iloc[62:].notna().all()

    def test_var_backtest(self, var_calculator, sample_returns):
        """Test VaR backtesting."""
        # Calculate rolling VaR
        rolling_var = var_calculator.rolling_var(sample_returns, window=63)

        # Backtest
        backtest = var_calculator.var_backtest(
            sample_returns[63:],
            rolling_var[63:],
            0.95
        )

        assert "exceedances" in backtest
        assert "expected_exceedances" in backtest
        assert "exceedance_ratio" in backtest

    def test_insufficient_data(self, var_calculator):
        """Test error handling for insufficient data."""
        short_returns = pd.Series([0.01, 0.02, -0.01])

        with pytest.raises(ValueError, match="Insufficient data"):
            var_calculator.historical_var(short_returns, 0.95)


# ============================================================================
# VolatilityForecaster Tests
# ============================================================================

class TestVolatilityForecaster:
    """Tests for VolatilityForecaster class."""

    def test_realized_volatility(self, vol_forecaster, sample_returns):
        """Test realized volatility calculation."""
        vol = vol_forecaster.realized_volatility(sample_returns, window=21)

        assert isinstance(vol, pd.Series)
        # Volatility should be positive
        assert (vol.dropna() > 0).all()

    def test_realized_volatility_annualized(self, vol_forecaster, sample_returns):
        """Test annualized vs non-annualized volatility."""
        vol_annual = vol_forecaster.realized_volatility(
            sample_returns, window=21, annualize=True
        )
        vol_daily = vol_forecaster.realized_volatility(
            sample_returns, window=21, annualize=False
        )

        # Annualized should be sqrt(252) times daily
        ratio = vol_annual.dropna() / vol_daily.dropna()
        expected_ratio = np.sqrt(252)
        assert np.allclose(ratio, expected_ratio, rtol=0.01)

    def test_ewma_volatility(self, vol_forecaster, sample_returns):
        """Test EWMA volatility calculation."""
        vol = vol_forecaster.ewma_volatility(sample_returns, span=21)

        assert isinstance(vol, pd.Series)
        assert (vol.dropna() > 0).all()

    def test_parkinson_volatility(self, vol_forecaster):
        """Test Parkinson volatility with high/low prices."""
        np.random.seed(42)
        n = 100
        close = 100 * np.cumprod(1 + np.random.normal(0, 0.02, n))
        high = close * (1 + np.abs(np.random.normal(0, 0.01, n)))
        low = close * (1 - np.abs(np.random.normal(0, 0.01, n)))

        vol = vol_forecaster.parkinson_volatility(
            pd.Series(high), pd.Series(low), window=21
        )

        assert isinstance(vol, pd.Series)
        assert (vol.dropna() > 0).all()

    def test_garch_forecast(self, vol_forecaster, sample_returns):
        """Test GARCH model fitting and forecasting."""
        result = vol_forecaster.garch_forecast(sample_returns, horizon=5)

        assert isinstance(result, GARCHResult)
        assert len(result.forecast) == 5
        assert result.persistence > 0
        # Persistence should typically be < 1 for stationarity
        assert result.persistence < 1.2

    def test_garch_forecast_fallback(self, vol_forecaster):
        """Test GARCH fallback on failure."""
        # Very short series that might cause GARCH to fail
        returns = pd.Series(np.random.normal(0, 0.02, 50))

        # Should not raise, should return EWMA fallback
        result = vol_forecaster.garch_forecast(returns, horizon=5)
        assert isinstance(result, GARCHResult)

    def test_volatility_regime(self, vol_forecaster, sample_returns):
        """Test volatility regime classification."""
        regimes = vol_forecaster.volatility_regime(sample_returns, window=63)

        assert isinstance(regimes, pd.Series)
        # Check valid regime values
        valid_values = {'low', 'medium', 'high', np.nan}
        unique_vals = set(regimes.dropna().unique())
        assert unique_vals.issubset({'low', 'medium', 'high'})

    def test_volatility_cone(self, vol_forecaster, sample_returns):
        """Test volatility cone generation."""
        cone = vol_forecaster.volatility_cone(sample_returns)

        assert isinstance(cone, pd.DataFrame)
        assert 'p5' in cone.columns
        assert 'p95' in cone.columns
        assert 'current' in cone.columns

    def test_volatility_term_structure(self, vol_forecaster, sample_returns):
        """Test volatility term structure."""
        term = vol_forecaster.volatility_term_structure(sample_returns)

        assert isinstance(term, pd.DataFrame)
        assert 'window' in term.columns
        assert 'volatility' in term.columns


# ============================================================================
# DrawdownAnalyzer Tests
# ============================================================================

class TestDrawdownAnalyzer:
    """Tests for DrawdownAnalyzer class."""

    def test_calculate_drawdown(self, dd_analyzer, sample_prices):
        """Test drawdown calculation."""
        dd_df = dd_analyzer.calculate_drawdown(sample_prices)

        assert isinstance(dd_df, pd.DataFrame)
        assert 'prices' in dd_df.columns
        assert 'peak' in dd_df.columns
        assert 'drawdown' in dd_df.columns

        # Drawdown should be between 0 and 1
        assert (dd_df['drawdown'] >= 0).all()
        assert (dd_df['drawdown'] <= 1).all()

        # Peak should always be >= prices
        assert (dd_df['peak'] >= dd_df['prices']).all()

    def test_max_drawdown(self, dd_analyzer, trending_down_prices):
        """Test maximum drawdown calculation."""
        mdd = dd_analyzer.max_drawdown(trending_down_prices)

        assert 0 < mdd < 1
        # Given our test data, expect significant drawdown
        assert mdd > 0.15

    def test_max_drawdown_no_decline(self, dd_analyzer):
        """Test max drawdown for always-rising prices."""
        rising_prices = pd.Series(range(100, 200))
        mdd = dd_analyzer.max_drawdown(rising_prices)
        assert mdd == 0.0

    def test_drawdown_duration(self, dd_analyzer, trending_down_prices):
        """Test drawdown duration calculation."""
        duration = dd_analyzer.drawdown_duration(trending_down_prices)

        assert 'current_duration' in duration
        assert 'max_duration' in duration
        assert 'is_in_drawdown' in duration

        assert duration['max_duration'] > 0

    def test_recovery_time(self, dd_analyzer, trending_down_prices):
        """Test recovery time analysis."""
        recovery = dd_analyzer.recovery_time(trending_down_prices)

        assert 'total_recoveries' in recovery
        assert 'avg_recovery_days' in recovery
        assert 'max_recovery_days' in recovery

    def test_underwater_periods(self, dd_analyzer, trending_down_prices):
        """Test underwater period detection."""
        periods = dd_analyzer.underwater_periods(trending_down_prices)

        assert isinstance(periods, list)
        # Should have at least one drawdown period
        assert len(periods) > 0

        for period in periods:
            assert isinstance(period, DrawdownPeriod)
            assert period.drawdown > 0
            assert period.duration_days >= 0

    def test_calmar_ratio(self, dd_analyzer, sample_prices):
        """Test Calmar ratio calculation."""
        calmar = dd_analyzer.calmar_ratio(sample_prices)

        # Calmar ratio should be a reasonable number
        assert -100 < calmar < 100

    def test_ulcer_index(self, dd_analyzer, trending_down_prices):
        """Test Ulcer Index calculation."""
        ulcer = dd_analyzer.ulcer_index(trending_down_prices)

        assert ulcer > 0  # Should have some drawdown pain

    def test_pain_index(self, dd_analyzer, trending_down_prices):
        """Test Pain Index calculation."""
        pain = dd_analyzer.pain_index(trending_down_prices)

        assert 0 <= pain < 1  # Average drawdown as decimal

    def test_drawdown_summary(self, dd_analyzer, sample_prices):
        """Test comprehensive drawdown summary."""
        summary = dd_analyzer.drawdown_summary(sample_prices)

        assert 'current_drawdown' in summary
        assert 'max_drawdown' in summary
        assert 'calmar_ratio' in summary
        assert 'ulcer_index' in summary

    def test_invalid_prices(self, dd_analyzer):
        """Test error handling for invalid prices."""
        # Negative prices
        with pytest.raises(ValueError, match="positive"):
            dd_analyzer.calculate_drawdown(pd.Series([-1, 0, 1]))

        # Too short
        with pytest.raises(ValueError, match="Insufficient"):
            dd_analyzer.calculate_drawdown(pd.Series([100]))


# ============================================================================
# RiskReport Tests
# ============================================================================

class TestRiskReport:
    """Tests for RiskReport class."""

    def test_init_default_components(self):
        """Test default component initialization."""
        report = RiskReport()

        assert isinstance(report.var_calc, VaRCalculator)
        assert isinstance(report.vol_forecaster, VolatilityForecaster)
        assert isinstance(report.dd_analyzer, DrawdownAnalyzer)

    def test_init_custom_components(self, var_calculator, vol_forecaster, dd_analyzer):
        """Test custom component injection."""
        report = RiskReport(
            var_calculator=var_calculator,
            vol_forecaster=vol_forecaster,
            dd_analyzer=dd_analyzer
        )

        assert report.var_calc is var_calculator
        assert report.vol_forecaster is vol_forecaster
        assert report.dd_analyzer is dd_analyzer

    def test_generate_report(self, sample_prices, sample_returns):
        """Test comprehensive risk report generation."""
        report = RiskReport()
        metrics = report.generate_report(sample_prices, sample_returns, symbol='TEST')

        assert isinstance(metrics, RiskMetrics)
        assert metrics.symbol == 'TEST'
        assert metrics.var_95 > 0
        assert metrics.var_99 > metrics.var_95
        assert metrics.volatility_21d > 0
        assert 0 <= metrics.max_drawdown <= 1

    def test_generate_report_auto_returns(self, sample_prices):
        """Test report generation with automatic return calculation."""
        report = RiskReport()
        metrics = report.generate_report(sample_prices, symbol='TEST')

        assert isinstance(metrics, RiskMetrics)
        assert metrics.var_95 > 0

    def test_to_dataframe_single(self, sample_prices):
        """Test DataFrame conversion for single metrics."""
        report = RiskReport()
        metrics = report.generate_report(sample_prices, symbol='TEST')
        df = report.to_dataframe(metrics)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df['symbol'].iloc[0] == 'TEST'

    def test_to_dataframe_multiple(self, sample_prices):
        """Test DataFrame conversion for multiple metrics."""
        report = RiskReport()

        # Generate multiple reports
        metrics1 = report.generate_report(sample_prices, symbol='TEST1')
        metrics2 = report.generate_report(sample_prices * 1.1, symbol='TEST2')

        df = report.to_dataframe([metrics1, metrics2])

        assert len(df) == 2
        assert 'TEST1' in df['symbol'].values
        assert 'TEST2' in df['symbol'].values

    def test_compare_symbols(self, sample_prices):
        """Test cross-symbol comparison."""
        report = RiskReport()

        symbol_data = {
            'SYM1': {'prices': sample_prices},
            'SYM2': {'prices': sample_prices * 1.1},
            'SYM3': {'prices': sample_prices * 0.9}
        }

        comparison = report.compare_symbols(symbol_data)

        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 3

    def test_risk_ranking(self, sample_prices):
        """Test risk ranking functionality."""
        report = RiskReport()

        symbol_data = {
            'LOW_VOL': {'prices': sample_prices},
            'HIGH_VOL': {'prices': sample_prices * (1 + np.random.normal(0, 0.1, len(sample_prices)))}
        }

        ranking = report.risk_ranking(symbol_data, metric='var_95')

        assert 'risk_rank' in ranking.columns
        assert len(ranking) == 2

    def test_generate_full_report(self, sample_prices):
        """Test full detailed report generation."""
        report = RiskReport()
        full_report = report.generate_full_report(sample_prices, symbol='TEST')

        assert 'summary' in full_report
        assert 'var_analysis' in full_report
        assert 'volatility_analysis' in full_report
        assert 'drawdown_analysis' in full_report
        assert 'risk_scores' in full_report

        assert full_report['summary']['symbol'] == 'TEST'

    def test_export_report_csv(self, sample_prices, tmp_path):
        """Test CSV export."""
        report = RiskReport()
        metrics = report.generate_report(sample_prices, symbol='TEST')

        filepath = tmp_path / "risk_report.csv"
        report.export_report(metrics, str(filepath), format='csv')

        assert filepath.exists()
        df = pd.read_csv(filepath)
        assert 'symbol' in df.columns

    def test_export_report_json(self, sample_prices, tmp_path):
        """Test JSON export."""
        report = RiskReport()
        metrics = report.generate_report(sample_prices, symbol='TEST')

        filepath = tmp_path / "risk_report.json"
        report.export_report(metrics, str(filepath), format='json')

        assert filepath.exists()


# ============================================================================
# Integration Tests with Real Data
# ============================================================================

class TestIntegrationWithRealData:
    """Integration tests using actual price data if available."""

    @pytest.fixture
    def real_prices(self):
        """Load real price data if available."""
        try:
            df = pd.read_csv('data/raw/AAPL_prices.csv')
            prices = df['Close']
            prices.index = pd.to_datetime(df['Date'])
            return prices
        except FileNotFoundError:
            pytest.skip("Real price data not available")

    def test_full_pipeline_real_data(self, real_prices):
        """Test complete risk analysis pipeline with real data."""
        report = RiskReport()
        metrics = report.generate_report(real_prices, symbol='AAPL')

        assert metrics.symbol == 'AAPL'
        assert 0 < metrics.var_95 < 0.2  # Reasonable daily VaR
        assert 0 < metrics.volatility_21d < 1.0  # Reasonable annualized vol
        assert 0 <= metrics.max_drawdown < 1.0

    def test_var_backtest_real_data(self, real_prices):
        """Test VaR backtesting with real data."""
        returns = real_prices.pct_change().dropna()

        calc = VaRCalculator()
        rolling_var = calc.rolling_var(returns, window=63)

        backtest = calc.var_backtest(
            returns[63:],
            rolling_var[63:],
            0.95
        )

        # Exceedance ratio should be reasonable
        assert 0 < backtest['exceedance_ratio'] < 5

    def test_garch_real_data(self, real_prices):
        """Test GARCH modeling with real data."""
        returns = real_prices.pct_change().dropna()

        forecaster = VolatilityForecaster()
        result = forecaster.garch_forecast(returns, horizon=5)

        assert result.converged or result.forecast is not None
        assert len(result.forecast) == 5
