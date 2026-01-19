"""
Comprehensive test suite for the feature engineering module.

Tests:
- PriceFeatureBuilder
- SentimentFeatureBuilder
- RiskFeatureBuilder
- FeatureBuilder (pipeline)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.features import (
    PriceFeatureBuilder,
    SentimentFeatureBuilder,
    RiskFeatureBuilder,
    FeatureBuilder,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_returns():
    """Generate sample return series for testing."""
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, 252)
    dates = pd.date_range(start='2024-01-01', periods=252, freq='B')
    return pd.Series(returns, index=dates, name='returns')


@pytest.fixture
def sample_prices(sample_returns):
    """Generate sample price series from returns."""
    prices = 100 * (1 + sample_returns).cumprod()
    prices.name = 'Close'
    return prices


@pytest.fixture
def sample_ohlcv(sample_prices):
    """Generate OHLCV DataFrame."""
    np.random.seed(42)
    n = len(sample_prices)

    # Create realistic OHLC from close
    high = sample_prices * (1 + np.abs(np.random.normal(0, 0.01, n)))
    low = sample_prices * (1 - np.abs(np.random.normal(0, 0.01, n)))
    open_price = sample_prices.shift(1).fillna(sample_prices.iloc[0])
    volume = np.random.randint(1000000, 10000000, n)

    df = pd.DataFrame({
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': sample_prices,
        'Volume': volume
    }, index=sample_prices.index)

    return df


@pytest.fixture
def sample_sentiment():
    """Generate sample sentiment DataFrame."""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='B')

    df = pd.DataFrame({
        'symbol': 'AAPL',
        'date': dates,
        'sentiment_score': np.random.uniform(-0.5, 0.5, 100),
        'sentiment_confidence': np.random.uniform(0.2, 0.8, 100),
        'article_count': np.random.randint(5, 50, 100),
        'bullish_ratio': np.random.uniform(0.3, 0.7, 100),
        'bearish_ratio': np.random.uniform(0.1, 0.4, 100),
        'sentiment_std': np.random.uniform(0.1, 0.4, 100),
        'signal_valid': np.random.choice([True, False], 100)
    })

    return df


@pytest.fixture
def price_builder():
    """Create price feature builder instance."""
    return PriceFeatureBuilder()


@pytest.fixture
def sentiment_builder():
    """Create sentiment feature builder instance."""
    return SentimentFeatureBuilder()


@pytest.fixture
def risk_builder():
    """Create risk feature builder instance."""
    return RiskFeatureBuilder()


@pytest.fixture
def feature_builder():
    """Create main feature builder instance."""
    return FeatureBuilder()


# ============================================================================
# PriceFeatureBuilder Tests
# ============================================================================

class TestPriceFeatureBuilder:
    """Tests for PriceFeatureBuilder class."""

    def test_init_default_windows(self):
        """Test default window sizes."""
        builder = PriceFeatureBuilder()
        assert builder.windows == [5, 10, 21, 63]
        assert builder.return_periods == [1, 5, 21]

    def test_init_custom_windows(self):
        """Test custom window sizes."""
        builder = PriceFeatureBuilder(windows=[10, 20], return_periods=[1, 5])
        assert builder.windows == [10, 20]
        assert builder.return_periods == [1, 5]

    def test_returns(self, price_builder, sample_ohlcv):
        """Test multi-period returns calculation."""
        returns_df = price_builder.returns(sample_ohlcv)

        assert 'return_1d' in returns_df.columns
        assert 'return_5d' in returns_df.columns
        assert 'return_21d' in returns_df.columns
        assert len(returns_df) == len(sample_ohlcv)

        # First row should be NaN for 1d return
        assert pd.isna(returns_df['return_1d'].iloc[0])

    def test_log_returns(self, price_builder, sample_ohlcv):
        """Test log returns calculation."""
        log_returns_df = price_builder.log_returns(sample_ohlcv)

        assert 'log_return_1d' in log_returns_df.columns
        assert len(log_returns_df) == len(sample_ohlcv)

    def test_rsi(self, price_builder, sample_ohlcv):
        """Test RSI calculation."""
        rsi = price_builder.rsi(sample_ohlcv, window=14)

        assert len(rsi) == len(sample_ohlcv)
        # RSI should be between 0 and 100
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()

    def test_macd(self, price_builder, sample_ohlcv):
        """Test MACD calculation."""
        macd_df = price_builder.macd(sample_ohlcv)

        assert 'macd' in macd_df.columns
        assert 'macd_signal' in macd_df.columns
        assert 'macd_histogram' in macd_df.columns

    def test_stochastic(self, price_builder, sample_ohlcv):
        """Test Stochastic Oscillator calculation."""
        stoch_df = price_builder.stochastic(sample_ohlcv)

        assert 'stoch_k' in stoch_df.columns
        assert 'stoch_d' in stoch_df.columns

        # Stochastic should be between 0 and 100
        valid_k = stoch_df['stoch_k'].dropna()
        assert (valid_k >= 0).all()
        assert (valid_k <= 100).all()

    def test_stochastic_missing_columns(self, price_builder, sample_prices):
        """Test stochastic with missing High/Low columns."""
        df = sample_prices.to_frame(name='Close')

        with pytest.raises(ValueError, match="Missing"):
            price_builder.stochastic(df)

    def test_volatility_features(self, price_builder, sample_ohlcv):
        """Test volatility features calculation."""
        vol_df = price_builder.volatility_features(sample_ohlcv)

        assert 'volatility_5' in vol_df.columns
        assert 'volatility_21' in vol_df.columns

        # Volatility should be positive
        valid_vol = vol_df['volatility_21'].dropna()
        assert (valid_vol >= 0).all()

    def test_bollinger_bands(self, price_builder, sample_ohlcv):
        """Test Bollinger Bands calculation."""
        bb_df = price_builder.bollinger_bands(sample_ohlcv)

        assert 'bb_middle' in bb_df.columns
        assert 'bb_upper' in bb_df.columns
        assert 'bb_lower' in bb_df.columns
        assert 'bb_pct_b' in bb_df.columns
        assert 'bb_bandwidth' in bb_df.columns

        # Upper band should be above middle, which should be above lower
        valid_idx = bb_df['bb_middle'].dropna().index
        assert (bb_df.loc[valid_idx, 'bb_upper'] >= bb_df.loc[valid_idx, 'bb_middle']).all()
        assert (bb_df.loc[valid_idx, 'bb_middle'] >= bb_df.loc[valid_idx, 'bb_lower']).all()

    def test_moving_averages(self, price_builder, sample_ohlcv):
        """Test moving averages calculation."""
        ma_df = price_builder.moving_averages(sample_ohlcv)

        assert 'sma_5' in ma_df.columns
        assert 'ema_5' in ma_df.columns
        assert 'sma_21' in ma_df.columns
        assert 'ema_21' in ma_df.columns

    def test_ma_crossover_features(self, price_builder, sample_ohlcv):
        """Test MA crossover features."""
        cross_df = price_builder.ma_crossover_features(sample_ohlcv)

        assert 'ma_diff' in cross_df.columns
        assert 'ma_ratio' in cross_df.columns
        assert 'ma_cross_up' in cross_df.columns
        assert 'ma_cross_down' in cross_df.columns

        # Crossover signals should be binary
        assert set(cross_df['ma_cross_up'].dropna().unique()).issubset({0, 1})
        assert set(cross_df['ma_cross_down'].dropna().unique()).issubset({0, 1})

    def test_volume_features(self, price_builder, sample_ohlcv):
        """Test volume features calculation."""
        vol_df = price_builder.volume_features(sample_ohlcv)

        assert 'volume_sma' in vol_df.columns
        assert 'volume_ratio' in vol_df.columns
        assert 'obv' in vol_df.columns

    def test_volume_features_missing_column(self, price_builder, sample_prices):
        """Test volume features with missing Volume column."""
        df = sample_prices.to_frame(name='Close')
        vol_df = price_builder.volume_features(df)

        # Should return empty DataFrame
        assert len(vol_df.columns) == 0

    def test_price_momentum(self, price_builder, sample_ohlcv):
        """Test price momentum features."""
        mom_df = price_builder.price_momentum(sample_ohlcv)

        assert 'momentum_5' in mom_df.columns
        assert 'roc_5' in mom_df.columns

    def test_atr(self, price_builder, sample_ohlcv):
        """Test ATR calculation."""
        atr = price_builder.atr(sample_ohlcv)

        assert len(atr) == len(sample_ohlcv)
        # ATR should be positive
        valid_atr = atr.dropna()
        assert (valid_atr >= 0).all()

    def test_build_all(self, price_builder, sample_ohlcv):
        """Test building all features."""
        features = price_builder.build_all(sample_ohlcv)

        # Should have many features
        assert len(features.columns) > 30
        assert len(features) == len(sample_ohlcv)

        # Check some key features are present
        assert 'return_1d' in features.columns
        assert 'rsi_14' in features.columns
        assert 'macd' in features.columns
        assert 'bb_pct_b' in features.columns

    def test_build_all_series_input(self, price_builder, sample_prices):
        """Test build_all with Series input."""
        features = price_builder.build_all(sample_prices.to_frame(name='Close'), include_volume=False)

        assert len(features) == len(sample_prices)
        assert 'return_1d' in features.columns


# ============================================================================
# SentimentFeatureBuilder Tests
# ============================================================================

class TestSentimentFeatureBuilder:
    """Tests for SentimentFeatureBuilder class."""

    def test_init_default_lags(self):
        """Test default lag values."""
        builder = SentimentFeatureBuilder()
        assert builder.lags == [1, 2, 3, 5, 7]
        assert builder.momentum_windows == [3, 5, 7]

    def test_init_custom_lags(self):
        """Test custom lag values."""
        builder = SentimentFeatureBuilder(lags=[1, 2, 3])
        assert builder.lags == [1, 2, 3]

    def test_sentiment_lags(self, sentiment_builder, sample_sentiment):
        """Test sentiment lags calculation."""
        lags_df = sentiment_builder.sentiment_lags(sample_sentiment, symbol='AAPL')

        assert 'sentiment_lag_1' in lags_df.columns
        assert 'sentiment_lag_5' in lags_df.columns

        # First lag rows should be NaN
        assert pd.isna(lags_df['sentiment_lag_1'].iloc[0])

    def test_sentiment_momentum(self, sentiment_builder, sample_sentiment):
        """Test sentiment momentum calculation."""
        mom_df = sentiment_builder.sentiment_momentum(sample_sentiment, symbol='AAPL')

        assert 'sentiment_change_3' in mom_df.columns
        assert 'sentiment_pct_change_3' in mom_df.columns

    def test_sentiment_ma(self, sentiment_builder, sample_sentiment):
        """Test sentiment moving averages."""
        ma_df = sentiment_builder.sentiment_ma(sample_sentiment, symbol='AAPL')

        assert 'sentiment_sma_3' in ma_df.columns
        assert 'sentiment_ema_5' in ma_df.columns

    def test_sentiment_disagreement(self, sentiment_builder, sample_sentiment):
        """Test sentiment disagreement features."""
        disag_df = sentiment_builder.sentiment_disagreement(sample_sentiment, symbol='AAPL')

        assert 'sentiment_std' in disag_df.columns
        assert 'disagreement_high' in disag_df.columns
        assert 'sentiment_range' in disag_df.columns

    def test_sentiment_zscore(self, sentiment_builder, sample_sentiment):
        """Test sentiment z-score calculation."""
        zscore_df = sentiment_builder.sentiment_zscore(sample_sentiment, symbol='AAPL')

        assert 'sentiment_zscore' in zscore_df.columns
        assert 'sentiment_percentile' in zscore_df.columns

    def test_article_count_features(self, sentiment_builder, sample_sentiment):
        """Test article count features."""
        count_df = sentiment_builder.article_count_features(sample_sentiment, symbol='AAPL')

        assert 'article_count' in count_df.columns
        assert 'article_count_sma_5' in count_df.columns
        assert 'high_attention' in count_df.columns

    def test_confidence_features(self, sentiment_builder, sample_sentiment):
        """Test confidence features."""
        conf_df = sentiment_builder.confidence_features(sample_sentiment, symbol='AAPL')

        assert 'sentiment_confidence' in conf_df.columns
        assert 'weighted_sentiment' in conf_df.columns

    def test_signal_features(self, sentiment_builder, sample_sentiment):
        """Test signal features."""
        signal_df = sentiment_builder.signal_features(sample_sentiment, symbol='AAPL')

        assert 'bullish_signal' in signal_df.columns
        assert 'bearish_signal' in signal_df.columns
        assert 'neutral_signal' in signal_df.columns

        # Signals should be binary
        assert set(signal_df['bullish_signal'].unique()).issubset({0, 1})

    def test_interaction_features(self, sentiment_builder, sample_sentiment):
        """Test interaction features."""
        inter_df = sentiment_builder.interaction_features(sample_sentiment, symbol='AAPL')

        assert 'sent_x_conf' in inter_df.columns
        assert 'sentiment_acceleration' in inter_df.columns

    def test_build_all(self, sentiment_builder, sample_sentiment):
        """Test building all sentiment features."""
        features = sentiment_builder.build_all(sample_sentiment, symbol='AAPL')

        # Should have many features
        assert len(features.columns) > 20

        # Check key features
        assert 'sentiment_score' in features.columns
        assert 'sentiment_lag_1' in features.columns
        assert 'sentiment_zscore' in features.columns

    def test_validate_missing_score(self, sentiment_builder):
        """Test validation with missing sentiment_score column."""
        df = pd.DataFrame({'date': pd.date_range('2024-01-01', periods=10)})

        with pytest.raises(ValueError, match="sentiment_score"):
            sentiment_builder.sentiment_lags(df)

    def test_symbol_filtering(self, sentiment_builder, sample_sentiment):
        """Test symbol filtering works correctly."""
        # Add another symbol
        sample_sentiment_multi = sample_sentiment.copy()
        msft = sample_sentiment.copy()
        msft['symbol'] = 'MSFT'
        msft['sentiment_score'] = msft['sentiment_score'] * -1
        sample_sentiment_multi = pd.concat([sample_sentiment_multi, msft])

        features = sentiment_builder.build_all(sample_sentiment_multi, symbol='AAPL')

        # Should only have AAPL data
        assert len(features) == len(sample_sentiment)


# ============================================================================
# RiskFeatureBuilder Tests
# ============================================================================

class TestRiskFeatureBuilder:
    """Tests for RiskFeatureBuilder class."""

    def test_init_default_values(self):
        """Test default initialization."""
        builder = RiskFeatureBuilder()
        assert builder.var_windows == [21, 63]
        assert builder.var_confidences == [0.95, 0.99]

    def test_init_custom_values(self):
        """Test custom initialization."""
        builder = RiskFeatureBuilder(var_windows=[10, 20], var_confidences=[0.90])
        assert builder.var_windows == [10, 20]
        assert builder.var_confidences == [0.90]

    def test_var_features(self, risk_builder, sample_ohlcv):
        """Test VaR features calculation."""
        var_df = risk_builder.var_features(sample_ohlcv)

        assert 'var_95_21' in var_df.columns
        assert 'var_99_63' in var_df.columns
        assert 'es_95_21' in var_df.columns

        # VaR should be positive (representing loss)
        valid_var = var_df['var_95_21'].dropna()
        # Most VaR values should be positive for a typical return series
        assert (valid_var > 0).mean() > 0.9

    def test_volatility_features(self, risk_builder, sample_ohlcv):
        """Test volatility features calculation."""
        vol_df = risk_builder.volatility_features(sample_ohlcv)

        assert 'realized_vol_5' in vol_df.columns
        assert 'ewma_vol_21' in vol_df.columns
        assert 'vol_ratio_5_10' in vol_df.columns

        # Volatility should be positive
        valid_vol = vol_df['realized_vol_21'].dropna()
        assert (valid_vol >= 0).all()

    def test_volatility_regime_features(self, risk_builder, sample_ohlcv):
        """Test volatility regime features."""
        regime_df = risk_builder.volatility_regime_features(sample_ohlcv)

        assert 'vol_regime' in regime_df.columns
        assert 'vol_regime_low' in regime_df.columns
        assert 'vol_regime_high' in regime_df.columns
        assert 'vol_regime_duration' in regime_df.columns

        # Regime dummies should be mutually exclusive
        valid_idx = regime_df.dropna(subset=['vol_regime']).index
        row_sums = (regime_df.loc[valid_idx, ['vol_regime_low', 'vol_regime_medium', 'vol_regime_high']].sum(axis=1))
        assert (row_sums == 1).all()

    def test_drawdown_features(self, risk_builder, sample_ohlcv):
        """Test drawdown features calculation."""
        dd_df = risk_builder.drawdown_features(sample_ohlcv)

        assert 'current_drawdown' in dd_df.columns
        assert 'days_since_peak' in dd_df.columns
        assert 'underwater' in dd_df.columns

        # Drawdown should be between 0 and 1
        valid_dd = dd_df['current_drawdown'].dropna()
        assert (valid_dd >= 0).all()
        assert (valid_dd <= 1).all()

    def test_garch_features(self, risk_builder, sample_ohlcv):
        """Test GARCH features calculation."""
        garch_df = risk_builder.garch_features(sample_ohlcv)

        assert 'garch_vol' in garch_df.columns
        assert 'garch_persistence' in garch_df.columns
        assert 'vol_surprise' in garch_df.columns

    def test_garch_features_insufficient_data(self, risk_builder):
        """Test GARCH with insufficient data."""
        # Only 50 days of data
        np.random.seed(42)
        prices = pd.Series(100 * (1 + np.random.normal(0, 0.02, 50)).cumprod())
        prices.name = 'Close'

        garch_df = risk_builder.garch_features(prices.to_frame(name='Close'))

        # Should have NaN values
        assert garch_df['garch_vol'].isna().all()

    def test_tail_risk_features(self, risk_builder, sample_ohlcv):
        """Test tail risk features."""
        tail_df = risk_builder.tail_risk_features(sample_ohlcv)

        assert 'skewness' in tail_df.columns
        assert 'kurtosis' in tail_df.columns
        assert 'tail_ratio' in tail_df.columns

    def test_risk_adjusted_features(self, risk_builder, sample_ohlcv):
        """Test risk-adjusted features."""
        risk_adj_df = risk_builder.risk_adjusted_features(sample_ohlcv)

        assert 'sharpe_ratio' in risk_adj_df.columns
        assert 'sortino_ratio' in risk_adj_df.columns
        assert 'calmar_ratio' in risk_adj_df.columns

    def test_build_all(self, risk_builder, sample_ohlcv):
        """Test building all risk features."""
        features = risk_builder.build_all(sample_ohlcv, include_garch=False)

        # Should have many features
        assert len(features.columns) > 15

        # Check key features
        assert 'var_95_21' in features.columns
        assert 'realized_vol_21' in features.columns
        assert 'current_drawdown' in features.columns

    def test_build_all_with_garch(self, risk_builder, sample_ohlcv):
        """Test building all features including GARCH."""
        features = risk_builder.build_all(sample_ohlcv, include_garch=True)

        assert 'garch_vol' in features.columns


# ============================================================================
# FeatureBuilder (Pipeline) Tests
# ============================================================================

class TestFeatureBuilder:
    """Tests for the main FeatureBuilder pipeline."""

    def test_init_default(self):
        """Test default initialization."""
        builder = FeatureBuilder()

        assert isinstance(builder.price_builder, PriceFeatureBuilder)
        assert isinstance(builder.sentiment_builder, SentimentFeatureBuilder)
        assert isinstance(builder.risk_builder, RiskFeatureBuilder)

    def test_init_custom_builders(self):
        """Test custom builder initialization."""
        price_builder = PriceFeatureBuilder(windows=[5, 10])
        builder = FeatureBuilder(price_builder=price_builder)

        assert builder.price_builder.windows == [5, 10]

    def test_build_price_features(self, feature_builder, sample_ohlcv):
        """Test building price features only."""
        features = feature_builder.build_price_features(sample_ohlcv)

        assert 'return_1d' in features.columns
        assert 'rsi_14' in features.columns

    def test_build_sentiment_features(self, feature_builder, sample_sentiment):
        """Test building sentiment features only."""
        features = feature_builder.build_sentiment_features(sample_sentiment, symbol='AAPL')

        assert 'sentiment_score' in features.columns
        assert 'sentiment_lag_1' in features.columns

    def test_build_risk_features(self, feature_builder, sample_ohlcv):
        """Test building risk features only."""
        features = feature_builder.build_risk_features(sample_ohlcv, include_garch=False)

        assert 'var_95_21' in features.columns
        assert 'current_drawdown' in features.columns

    def test_align_features(self, feature_builder):
        """Test feature alignment."""
        dates1 = pd.date_range('2024-01-01', periods=10, freq='B')
        dates2 = pd.date_range('2024-01-05', periods=10, freq='B')

        df1 = pd.DataFrame({'feature1': range(10)}, index=dates1)
        df2 = pd.DataFrame({'feature2': range(10)}, index=dates2)

        aligned = feature_builder.align_features(df1, df2, how='inner')

        # Should only have overlapping dates
        assert len(aligned) < 10
        assert 'feature1' in aligned.columns
        assert 'feature2' in aligned.columns

    def test_align_features_empty(self, feature_builder):
        """Test alignment with empty DataFrames."""
        result = feature_builder.align_features()
        assert len(result) == 0

    def test_build_features_complete(self, feature_builder, sample_ohlcv, sample_sentiment):
        """Test building complete feature set."""
        features = feature_builder.build_features(
            prices=sample_ohlcv,
            sentiment=sample_sentiment,
            symbol='AAPL',
            include_garch=False
        )

        # Should have features from all sources
        assert len(features.columns) > 50

        # Check features from different sources
        assert 'return_1d' in features.columns  # Price
        assert 'var_95_21' in features.columns  # Risk

        # Check symbol column
        assert 'symbol' in features.columns
        assert (features['symbol'] == 'AAPL').all()

    def test_build_features_no_sentiment(self, feature_builder, sample_ohlcv):
        """Test building features without sentiment."""
        features = feature_builder.build_features(
            prices=sample_ohlcv,
            sentiment=None,
            include_sentiment=False,
            include_garch=False
        )

        # Should still have price and risk features
        assert 'return_1d' in features.columns
        assert 'var_95_21' in features.columns

    def test_create_target_return(self, feature_builder, sample_ohlcv):
        """Test target creation - return type."""
        target = feature_builder.create_target(sample_ohlcv, horizon=1, target_type='return')

        assert target.name == 'target_return_1d'
        assert len(target) == len(sample_ohlcv)

        # Last row should be NaN (no future data)
        assert pd.isna(target.iloc[-1])

    def test_create_target_direction(self, feature_builder, sample_ohlcv):
        """Test target creation - direction type."""
        target = feature_builder.create_target(sample_ohlcv, horizon=1, target_type='direction')

        assert target.name == 'target_direction_1d'

        # Should be binary
        valid_target = target.dropna()
        assert set(valid_target.unique()).issubset({0, 1})

    def test_create_target_log_return(self, feature_builder, sample_ohlcv):
        """Test target creation - log return type."""
        target = feature_builder.create_target(sample_ohlcv, horizon=5, target_type='log_return')

        assert target.name == 'target_log_return_5d'

    def test_create_target_invalid_type(self, feature_builder, sample_ohlcv):
        """Test target creation with invalid type."""
        with pytest.raises(ValueError, match="Unknown target_type"):
            feature_builder.create_target(sample_ohlcv, target_type='invalid')

    def test_create_ml_dataset(self, feature_builder, sample_ohlcv):
        """Test ML dataset creation."""
        # Use only price features for this test to avoid alignment issues
        features = feature_builder.build_features(
            prices=sample_ohlcv,
            sentiment=None,
            include_sentiment=False,
            include_garch=False
        )

        X, y = feature_builder.create_ml_dataset(
            features=features,
            prices=sample_ohlcv,
            target_horizon=1,
            target_type='return',
            dropna=True
        )

        # Should have samples
        assert len(X) > 0
        assert len(y) > 0
        assert len(X) == len(y)

        # X should not contain target or symbol columns
        assert 'symbol' not in X.columns
        assert not any(c.startswith('target_') for c in X.columns)

        # X should be all numeric
        assert X.select_dtypes(include=[np.number]).shape == X.shape

    def test_get_feature_names(self, feature_builder, sample_ohlcv):
        """Test getting feature names."""
        features = feature_builder.build_price_features(sample_ohlcv)
        features['symbol'] = 'AAPL'

        names = feature_builder.get_feature_names(features)

        assert 'symbol' not in names
        assert 'return_1d' in names

    def test_feature_summary(self, feature_builder, sample_ohlcv):
        """Test feature summary generation."""
        features = feature_builder.build_price_features(sample_ohlcv)
        summary = feature_builder.feature_summary(features)

        assert 'count' in summary.columns
        assert 'missing' in summary.columns
        assert 'mean' in summary.columns
        assert 'std' in summary.columns


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for the complete feature pipeline."""

    def test_no_lookahead_bias(self, sample_ohlcv):
        """Test that features don't have lookahead bias."""
        builder = PriceFeatureBuilder()
        features = builder.build_all(sample_ohlcv)

        # Get the 100th row features
        idx = 100
        features_at_idx = features.iloc[idx]

        # Recalculate using only data up to idx
        partial_prices = sample_ohlcv.iloc[:idx+1]
        partial_features = builder.build_all(partial_prices)

        # Features should match (for features that don't depend on future data)
        # Check return_1d which uses data from t-1 and t
        assert np.isclose(
            features_at_idx['return_1d'],
            partial_features['return_1d'].iloc[-1],
            rtol=1e-10
        )

    def test_feature_alignment_consistency(self, sample_ohlcv, sample_sentiment):
        """Test that aligned features maintain consistency."""
        builder = FeatureBuilder()

        features = builder.build_features(
            prices=sample_ohlcv,
            sentiment=sample_sentiment,
            symbol='AAPL',
            include_garch=False
        )

        # All rows should have the same date for price and sentiment features
        # (no misalignment)
        assert not features.index.duplicated().any()

    def test_ml_dataset_shapes(self, sample_ohlcv, sample_sentiment):
        """Test ML dataset has consistent shapes."""
        builder = FeatureBuilder()

        features = builder.build_features(
            prices=sample_ohlcv,
            sentiment=sample_sentiment,
            symbol='AAPL',
            include_garch=False
        )

        X, y = builder.create_ml_dataset(
            features=features,
            prices=sample_ohlcv,
            target_horizon=1
        )

        # Shapes should match
        assert X.shape[0] == y.shape[0]

        # No NaN in final dataset
        assert not X.isna().any().any()
        assert not y.isna().any()

    def test_different_horizons(self, sample_ohlcv):
        """Test target creation with different horizons."""
        builder = FeatureBuilder()

        for horizon in [1, 5, 21]:
            target = builder.create_target(sample_ohlcv, horizon=horizon)

            # Should have correct number of NaN at the end
            assert target.iloc[-horizon:].isna().all()
            assert not target.iloc[:-horizon].isna().all()


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Edge case tests."""

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        builder = PriceFeatureBuilder()
        empty_df = pd.DataFrame(columns=['Close'])

        # Should return empty DataFrame with features
        features = builder.build_all(empty_df)
        assert len(features) == 0

    def test_single_row(self):
        """Test handling of single row DataFrame."""
        builder = PriceFeatureBuilder()
        single_df = pd.DataFrame({'Close': [100.0]}, index=[pd.Timestamp('2024-01-01')])

        features = builder.build_all(single_df, include_volume=False)

        # Should have many NaN since no history for rolling calculations
        # but EMA/MACD work with single point
        assert features.isna().mean().mean() > 0.5

    def test_constant_prices(self):
        """Test handling of constant prices (zero volatility)."""
        builder = PriceFeatureBuilder()
        dates = pd.date_range('2024-01-01', periods=100, freq='B')
        constant_df = pd.DataFrame({'Close': [100.0] * 100}, index=dates)

        features = builder.build_all(constant_df, include_volume=False)

        # Returns should be zero
        assert (features['return_1d'].dropna() == 0).all()

        # Volatility should be zero
        valid_vol = features['volatility_21'].dropna()
        assert (valid_vol == 0).all() or valid_vol.isna().all()

    def test_missing_values_in_input(self):
        """Test handling of missing values in input data."""
        builder = PriceFeatureBuilder()
        dates = pd.date_range('2024-01-01', periods=100, freq='B')
        prices = pd.Series(np.random.uniform(100, 110, 100), index=dates)
        prices.iloc[10:15] = np.nan  # Add some missing values

        features = builder.build_all(prices.to_frame(name='Close'), include_volume=False)

        # Should handle NaN and still produce features
        assert len(features) == 100

    def test_extreme_values(self):
        """Test handling of extreme price values."""
        builder = PriceFeatureBuilder()
        dates = pd.date_range('2024-01-01', periods=100, freq='B')

        # Very large price changes
        prices = pd.Series([100.0] * 50 + [200.0] * 50, index=dates)

        features = builder.build_all(prices.to_frame(name='Close'), include_volume=False)

        # Should handle the large change
        assert not features['return_1d'].replace([np.inf, -np.inf], np.nan).dropna().empty
