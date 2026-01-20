"""
Comprehensive test suite for the ML module.

Tests:
- DirectionClassifier
- ReturnRegressor
- EnsembleModel
- TimeSeriesSplit
- WalkForwardValidator
- SignalGenerator
- PredictionPipeline
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.ml import (
    # Model classes
    ModelMetrics,
    PredictionResult,
    DirectionClassifier,
    ReturnRegressor,
    EnsembleModel,
    # Validation classes
    FoldResult,
    ValidationResult,
    TimeSeriesSplit,
    PurgedTimeSeriesSplit,
    WalkForwardValidator,
    cross_validate,
    compare_models,
    # Signal classes
    Signal,
    SignalBatch,
    SignalGenerator,
    PredictionPipeline,
    generate_signals_from_features,
    rank_signals,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_features():
    """Generate sample feature matrix for testing."""
    np.random.seed(42)
    n_samples = 200
    n_features = 10

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)],
        index=pd.date_range(start='2024-01-01', periods=n_samples, freq='B')
    )
    return X


@pytest.fixture
def sample_direction_target(sample_features):
    """Generate binary direction target."""
    np.random.seed(42)
    # Create target with some predictable pattern
    noise = np.random.randn(len(sample_features)) * 0.3
    signal = sample_features['feature_0'].values + sample_features['feature_1'].values
    y = (signal + noise > 0).astype(int)
    return pd.Series(y, index=sample_features.index, name='target_direction')


@pytest.fixture
def sample_return_target(sample_features):
    """Generate return target."""
    np.random.seed(42)
    # Create continuous target with some predictable pattern
    noise = np.random.randn(len(sample_features)) * 0.01
    signal = (sample_features['feature_0'].values * 0.005 +
              sample_features['feature_1'].values * 0.003)
    y = signal + noise
    return pd.Series(y, index=sample_features.index, name='target_return')


@pytest.fixture
def trained_classifier(sample_features, sample_direction_target):
    """Create and train a classifier."""
    classifier = DirectionClassifier(params={'n_estimators': 50, 'max_depth': 3})
    classifier.fit(sample_features, sample_direction_target)
    return classifier


@pytest.fixture
def trained_regressor(sample_features, sample_return_target):
    """Create and train a regressor."""
    regressor = ReturnRegressor(params={'n_estimators': 50, 'max_depth': 3})
    regressor.fit(sample_features, sample_return_target)
    return regressor


@pytest.fixture
def temp_model_dir():
    """Create temporary directory for model storage."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


# ============================================================================
# DirectionClassifier Tests
# ============================================================================

class TestDirectionClassifier:
    """Tests for DirectionClassifier class."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        classifier = DirectionClassifier()
        assert classifier.is_fitted is False
        assert classifier.scale_features is True
        assert 'n_estimators' in classifier.params

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        classifier = DirectionClassifier(
            params={'n_estimators': 200, 'max_depth': 7},
            scale_features=False
        )
        assert classifier.params['n_estimators'] == 200
        assert classifier.params['max_depth'] == 7
        assert classifier.scale_features is False

    def test_fit(self, sample_features, sample_direction_target):
        """Test model training."""
        classifier = DirectionClassifier(params={'n_estimators': 50})
        result = classifier.fit(sample_features, sample_direction_target)

        assert classifier.is_fitted is True
        assert result is classifier  # Returns self
        assert len(classifier.feature_names) == sample_features.shape[1]

    def test_predict(self, trained_classifier, sample_features):
        """Test prediction."""
        predictions = trained_classifier.predict(sample_features)

        assert len(predictions) == len(sample_features)
        assert set(predictions).issubset({0, 1})

    def test_predict_proba(self, trained_classifier, sample_features):
        """Test probability prediction."""
        probabilities = trained_classifier.predict_proba(sample_features)

        assert probabilities.shape == (len(sample_features), 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)
        assert np.all(probabilities >= 0)
        assert np.all(probabilities <= 1)

    def test_predict_with_confidence(self, trained_classifier, sample_features):
        """Test prediction with confidence scores."""
        result = trained_classifier.predict_with_confidence(sample_features)

        assert isinstance(result, PredictionResult)
        assert len(result.predictions) == len(sample_features)
        assert len(result.confidence) == len(sample_features)
        assert result.probabilities.shape == (len(sample_features), 2)
        assert result.model_type == 'classifier'

    def test_evaluate(self, trained_classifier, sample_features, sample_direction_target):
        """Test model evaluation."""
        metrics = trained_classifier.evaluate(sample_features, sample_direction_target)

        assert isinstance(metrics, ModelMetrics)
        assert metrics.model_type == 'classifier'
        assert 'accuracy' in metrics.metrics
        assert 'precision' in metrics.metrics
        assert 'recall' in metrics.metrics
        assert 'f1' in metrics.metrics
        assert 'auc' in metrics.metrics
        assert 0 <= metrics.metrics['accuracy'] <= 1
        assert metrics.feature_importance is not None

    def test_feature_importance(self, trained_classifier):
        """Test feature importance extraction."""
        importance = trained_classifier.get_feature_importance()

        assert isinstance(importance, pd.DataFrame)
        assert 'feature' in importance.columns
        assert 'importance' in importance.columns
        assert len(importance) == len(trained_classifier.feature_names)

    def test_save_load(self, trained_classifier, temp_model_dir):
        """Test model persistence."""
        # Save
        save_path = trained_classifier.save(temp_model_dir / 'classifier')
        assert save_path.exists()

        # Load
        loaded = DirectionClassifier.load(save_path)
        assert loaded.is_fitted is True
        assert loaded.feature_names == trained_classifier.feature_names

    def test_predict_not_fitted(self, sample_features):
        """Test prediction fails when not fitted."""
        classifier = DirectionClassifier()
        with pytest.raises(ValueError, match="fitted"):
            classifier.predict(sample_features)

    def test_fit_with_numpy_arrays(self):
        """Test training with numpy arrays instead of DataFrames."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = (X[:, 0] > 0).astype(int)

        classifier = DirectionClassifier(params={'n_estimators': 50})
        classifier.fit(X, y)

        predictions = classifier.predict(X)
        assert len(predictions) == 100


# ============================================================================
# ReturnRegressor Tests
# ============================================================================

class TestReturnRegressor:
    """Tests for ReturnRegressor class."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        regressor = ReturnRegressor()
        assert regressor.is_fitted is False
        assert 'objective' in regressor.params
        assert regressor.params['objective'] == 'reg:squarederror'

    def test_fit(self, sample_features, sample_return_target):
        """Test model training."""
        regressor = ReturnRegressor(params={'n_estimators': 50})
        result = regressor.fit(sample_features, sample_return_target)

        assert regressor.is_fitted is True
        assert result is regressor

    def test_predict(self, trained_regressor, sample_features):
        """Test prediction."""
        predictions = trained_regressor.predict(sample_features)

        assert len(predictions) == len(sample_features)
        assert predictions.dtype == np.float64 or predictions.dtype == np.float32

    def test_predict_with_confidence(self, trained_regressor, sample_features):
        """Test prediction with confidence."""
        result = trained_regressor.predict_with_confidence(sample_features)

        assert isinstance(result, PredictionResult)
        assert len(result.predictions) == len(sample_features)
        assert len(result.confidence) == len(sample_features)
        assert result.model_type == 'regressor'

    def test_evaluate(self, trained_regressor, sample_features, sample_return_target):
        """Test model evaluation."""
        metrics = trained_regressor.evaluate(sample_features, sample_return_target)

        assert isinstance(metrics, ModelMetrics)
        assert metrics.model_type == 'regressor'
        assert 'mse' in metrics.metrics
        assert 'rmse' in metrics.metrics
        assert 'mae' in metrics.metrics
        assert 'r2' in metrics.metrics
        assert 'direction_accuracy' in metrics.metrics

    def test_save_load(self, trained_regressor, temp_model_dir):
        """Test model persistence."""
        save_path = trained_regressor.save(temp_model_dir / 'regressor')
        assert save_path.exists()

        loaded = ReturnRegressor.load(save_path)
        assert loaded.is_fitted is True


# ============================================================================
# EnsembleModel Tests
# ============================================================================

class TestEnsembleModel:
    """Tests for EnsembleModel class."""

    def test_init(self):
        """Test initialization."""
        ensemble = EnsembleModel()
        assert ensemble.classifier is not None
        assert ensemble.regressor is not None
        assert ensemble.is_fitted is False

    def test_fit(self, sample_features, sample_direction_target, sample_return_target):
        """Test ensemble training."""
        ensemble = EnsembleModel(
            classifier_params={'n_estimators': 50},
            regressor_params={'n_estimators': 50}
        )
        ensemble.fit(sample_features, sample_direction_target, sample_return_target)

        assert ensemble.is_fitted is True
        assert ensemble.classifier.is_fitted is True
        assert ensemble.regressor.is_fitted is True

    def test_predict(self, sample_features, sample_direction_target, sample_return_target):
        """Test ensemble prediction."""
        ensemble = EnsembleModel(
            classifier_params={'n_estimators': 50},
            regressor_params={'n_estimators': 50}
        )
        ensemble.fit(sample_features, sample_direction_target, sample_return_target)

        predictions = ensemble.predict(sample_features)

        assert 'direction' in predictions
        assert 'probability' in predictions
        assert 'return' in predictions
        assert 'signal' in predictions
        assert len(predictions['direction']) == len(sample_features)


# ============================================================================
# TimeSeriesSplit Tests
# ============================================================================

class TestTimeSeriesSplit:
    """Tests for TimeSeriesSplit class."""

    def test_init_defaults(self):
        """Test default initialization."""
        splitter = TimeSeriesSplit()
        assert splitter.n_splits == 5
        assert splitter.expanding is True

    def test_split_basic(self, sample_features):
        """Test basic split generation."""
        splitter = TimeSeriesSplit(n_splits=5, test_size=20, min_train_size=50)
        splits = list(splitter.split(sample_features))

        assert len(splits) == 5

        for train_idx, test_idx in splits:
            # Train comes before test
            assert train_idx[-1] < test_idx[0]
            # Test size is correct
            assert len(test_idx) == 20
            # Minimum train size met
            assert len(train_idx) >= 50

    def test_split_no_overlap(self, sample_features):
        """Test that train and test do not overlap."""
        splitter = TimeSeriesSplit(n_splits=5, test_size=20)

        for train_idx, test_idx in splitter.split(sample_features):
            overlap = set(train_idx) & set(test_idx)
            assert len(overlap) == 0

    def test_split_with_gap(self, sample_features):
        """Test split with gap between train and test."""
        splitter = TimeSeriesSplit(n_splits=3, test_size=20, gap=5)

        for train_idx, test_idx in splitter.split(sample_features):
            # Gap between train end and test start
            gap = test_idx[0] - train_idx[-1] - 1
            assert gap >= 5

    def test_expanding_window(self, sample_features):
        """Test expanding window behavior."""
        splitter = TimeSeriesSplit(n_splits=5, test_size=20, expanding=True)
        splits = list(splitter.split(sample_features))

        # Later folds should have more training data
        train_sizes = [len(train_idx) for train_idx, _ in splits]
        # First fold has most training data (since we work backwards)
        assert train_sizes[0] >= train_sizes[-1]

    def test_sliding_window(self, sample_features):
        """Test sliding window behavior."""
        splitter = TimeSeriesSplit(
            n_splits=5,
            test_size=20,
            train_size=50,
            expanding=False
        )
        splits = list(splitter.split(sample_features))

        for train_idx, _ in splits:
            # Training size should be capped
            assert len(train_idx) <= 50


class TestPurgedTimeSeriesSplit:
    """Tests for PurgedTimeSeriesSplit class."""

    def test_purge_removes_samples(self, sample_features):
        """Test that purge removes samples from training end."""
        splitter = PurgedTimeSeriesSplit(n_splits=3, test_size=20, purge_size=5)

        for train_idx, test_idx in splitter.split(sample_features):
            # Gap should include purge period
            gap = test_idx[0] - train_idx[-1] - 1
            assert gap >= 5


# ============================================================================
# WalkForwardValidator Tests
# ============================================================================

class TestWalkForwardValidator:
    """Tests for WalkForwardValidator class."""

    def test_init(self):
        """Test initialization."""
        validator = WalkForwardValidator(
            model_class=DirectionClassifier,
            n_splits=5
        )
        assert validator.n_splits == 5
        assert validator.model_class == DirectionClassifier

    def test_validate_classifier(self, sample_features, sample_direction_target):
        """Test validation with classifier."""
        validator = WalkForwardValidator(
            model_class=DirectionClassifier,
            model_params={'n_estimators': 50, 'max_depth': 3},
            n_splits=3,
            test_size=30,
            min_train_size=50,
            verbose=False
        )

        results = validator.validate(sample_features, sample_direction_target)

        assert isinstance(results, ValidationResult)
        assert results.n_folds == 3
        assert len(results.folds) == 3
        assert results.model_type == 'classifier'
        assert 'accuracy_mean' in results.aggregate_metrics

    def test_validate_regressor(self, sample_features, sample_return_target):
        """Test validation with regressor."""
        validator = WalkForwardValidator(
            model_class=ReturnRegressor,
            model_params={'n_estimators': 50, 'max_depth': 3},
            n_splits=3,
            test_size=30,
            min_train_size=50,
            verbose=False
        )

        results = validator.validate(sample_features, sample_return_target)

        assert isinstance(results, ValidationResult)
        assert results.model_type == 'regressor'
        assert 'mse_mean' in results.aggregate_metrics

    def test_validation_result_summary(self, sample_features, sample_direction_target):
        """Test validation result summary generation."""
        validator = WalkForwardValidator(
            model_class=DirectionClassifier,
            model_params={'n_estimators': 50},
            n_splits=3,
            test_size=30,
            verbose=False
        )

        results = validator.validate(sample_features, sample_direction_target)
        summary = results.summary()

        assert isinstance(summary, pd.DataFrame)
        assert 'fold' in summary.columns
        assert len(summary) == 4  # 3 folds + mean row

    def test_get_all_predictions(self, sample_features, sample_direction_target):
        """Test getting all predictions from validation."""
        validator = WalkForwardValidator(
            model_class=DirectionClassifier,
            model_params={'n_estimators': 50},
            n_splits=3,
            test_size=30,
            verbose=False
        )

        results = validator.validate(sample_features, sample_direction_target)
        predictions_df = results.get_all_predictions()

        assert isinstance(predictions_df, pd.DataFrame)
        assert 'prediction' in predictions_df.columns
        assert 'actual' in predictions_df.columns
        assert len(predictions_df) == results.total_test_samples


class TestCrossValidateFunction:
    """Tests for cross_validate convenience function."""

    def test_cross_validate(self, sample_features, sample_direction_target):
        """Test cross_validate function."""
        results = cross_validate(
            DirectionClassifier,
            sample_features,
            sample_direction_target,
            n_splits=3,
            model_params={'n_estimators': 50},
            test_size=30,
            verbose=False
        )

        assert isinstance(results, ValidationResult)
        assert results.n_folds == 3


class TestCompareModels:
    """Tests for compare_models function."""

    def test_compare_models(self, sample_features, sample_direction_target):
        """Test model comparison."""
        models = {
            'small': (DirectionClassifier, {'n_estimators': 30, 'max_depth': 2}),
            'medium': (DirectionClassifier, {'n_estimators': 50, 'max_depth': 3}),
        }

        comparison = compare_models(
            sample_features,
            sample_direction_target,
            models,
            n_splits=3,
            test_size=30,
            verbose=False
        )

        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 2
        assert 'model' in comparison.columns
        assert 'small' in comparison['model'].values
        assert 'medium' in comparison['model'].values


# ============================================================================
# SignalGenerator Tests
# ============================================================================

class TestSignalGenerator:
    """Tests for SignalGenerator class."""

    def test_init_defaults(self):
        """Test default initialization."""
        generator = SignalGenerator()
        assert generator.confidence_threshold == 0.5
        assert generator.neutral_zone == 0.05

    def test_generate_bullish_signal(self):
        """Test generating a bullish signal."""
        generator = SignalGenerator(confidence_threshold=0.3)
        signal = generator.generate(
            symbol='AAPL',
            direction=1,
            probability=0.75,
            expected_return=0.02
        )

        assert isinstance(signal, Signal)
        assert signal.symbol == 'AAPL'
        assert signal.direction == 1
        assert signal.is_bullish
        assert not signal.is_bearish
        assert signal.confidence > 0.3

    def test_generate_bearish_signal(self):
        """Test generating a bearish signal."""
        generator = SignalGenerator(confidence_threshold=0.3)
        signal = generator.generate(
            symbol='AAPL',
            direction=0,
            probability=0.25,
            expected_return=-0.02
        )

        assert signal.direction == -1
        assert signal.is_bearish
        assert not signal.is_bullish

    def test_generate_neutral_signal(self):
        """Test generating a neutral signal (in neutral zone)."""
        generator = SignalGenerator(neutral_zone=0.1)
        signal = generator.generate(
            symbol='AAPL',
            direction=1,
            probability=0.52  # Close to 0.5
        )

        assert signal.direction == 0
        assert signal.is_neutral

    def test_generate_below_confidence_threshold(self):
        """Test signal below confidence threshold becomes neutral."""
        generator = SignalGenerator(confidence_threshold=0.8)
        signal = generator.generate(
            symbol='AAPL',
            direction=1,
            probability=0.6  # Confidence = 0.2, below 0.8
        )

        assert signal.direction == 0

    def test_generate_batch(self):
        """Test batch signal generation."""
        generator = SignalGenerator()
        batch = generator.generate_batch(
            symbols=['AAPL', 'GOOGL', 'MSFT'],
            directions=np.array([1, 0, 1]),
            probabilities=np.array([0.7, 0.3, 0.8])
        )

        assert isinstance(batch, SignalBatch)
        assert len(batch) == 3
        assert batch.signals[0].symbol == 'AAPL'
        assert batch.signals[1].symbol == 'GOOGL'


class TestSignalBatch:
    """Tests for SignalBatch class."""

    @pytest.fixture
    def sample_batch(self):
        """Create sample signal batch."""
        signals = [
            Signal(pd.Timestamp.now(), 'AAPL', 1, 0.8, 0.7, 0.02),
            Signal(pd.Timestamp.now(), 'GOOGL', -1, 0.6, 0.5, -0.01),
            Signal(pd.Timestamp.now(), 'MSFT', 0, 0.2, 0.3, 0.001),
            Signal(pd.Timestamp.now(), 'TSLA', 1, 0.9, 0.8, 0.03),
        ]
        return SignalBatch(signals=signals)

    def test_to_dataframe(self, sample_batch):
        """Test conversion to DataFrame."""
        df = sample_batch.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4
        assert 'symbol' in df.columns
        assert 'direction' in df.columns
        assert 'confidence' in df.columns

    def test_filter_by_confidence(self, sample_batch):
        """Test filtering by confidence."""
        filtered = sample_batch.filter_by_confidence(min_confidence=0.6)

        assert len(filtered) == 2  # AAPL (0.7) and TSLA (0.8)

    def test_filter_by_strength(self, sample_batch):
        """Test filtering by strength."""
        filtered = sample_batch.filter_by_strength(min_strength=0.7)

        assert len(filtered) == 2  # AAPL (0.8) and TSLA (0.9)

    def test_get_bullish(self, sample_batch):
        """Test getting bullish signals."""
        bullish = sample_batch.get_bullish()

        assert len(bullish) == 2  # AAPL and TSLA
        assert all(s.is_bullish for s in bullish)

    def test_get_bearish(self, sample_batch):
        """Test getting bearish signals."""
        bearish = sample_batch.get_bearish()

        assert len(bearish) == 1  # GOOGL
        assert all(s.is_bearish for s in bearish)

    def test_get_by_symbol(self, sample_batch):
        """Test getting signals by symbol."""
        aapl_signals = sample_batch.get_by_symbol('AAPL')

        assert len(aapl_signals) == 1
        assert aapl_signals[0].symbol == 'AAPL'


# ============================================================================
# PredictionPipeline Tests
# ============================================================================

class TestPredictionPipeline:
    """Tests for PredictionPipeline class."""

    def test_init_with_classifier(self, trained_classifier):
        """Test initialization with classifier only."""
        pipeline = PredictionPipeline(classifier=trained_classifier)
        assert pipeline.classifier is not None
        assert pipeline.regressor is None

    def test_init_requires_model(self):
        """Test that initialization requires at least one model."""
        with pytest.raises(ValueError, match="at least one"):
            PredictionPipeline()

    def test_predict(self, trained_classifier, sample_features):
        """Test prediction generation."""
        pipeline = PredictionPipeline(classifier=trained_classifier)
        batch = pipeline.predict(sample_features, symbol='AAPL')

        assert isinstance(batch, SignalBatch)
        assert len(batch) == len(sample_features)
        assert all(s.symbol == 'AAPL' for s in batch.signals)

    def test_predict_latest(self, trained_classifier, sample_features):
        """Test generating signal for latest data point."""
        pipeline = PredictionPipeline(classifier=trained_classifier)
        signal = pipeline.predict_latest(sample_features, symbol='AAPL')

        assert isinstance(signal, Signal)
        assert signal.symbol == 'AAPL'

    def test_predict_with_regressor(
        self, trained_classifier, trained_regressor, sample_features
    ):
        """Test prediction with both classifier and regressor."""
        pipeline = PredictionPipeline(
            classifier=trained_classifier,
            regressor=trained_regressor
        )
        batch = pipeline.predict(sample_features, symbol='AAPL')

        assert len(batch) == len(sample_features)
        # Should have expected returns
        assert all(s.expected_return is not None for s in batch.signals)

    def test_predict_multiple_symbols(
        self, trained_classifier, sample_features
    ):
        """Test prediction for multiple symbols."""
        pipeline = PredictionPipeline(classifier=trained_classifier)

        features_dict = {
            'AAPL': sample_features,
            'GOOGL': sample_features,
        }

        batch = pipeline.predict_multiple_symbols(features_dict)

        assert len(batch) == 2
        symbols = [s.symbol for s in batch.signals]
        assert 'AAPL' in symbols
        assert 'GOOGL' in symbols


class TestGenerateSignalsFromFeatures:
    """Tests for generate_signals_from_features convenience function."""

    def test_generate_signals(self, trained_classifier, sample_features):
        """Test signal generation function."""
        signals_df = generate_signals_from_features(
            sample_features,
            'AAPL',
            trained_classifier,
            confidence_threshold=0.5
        )

        assert isinstance(signals_df, pd.DataFrame)
        assert len(signals_df) == len(sample_features)
        assert 'direction' in signals_df.columns
        assert 'confidence' in signals_df.columns


class TestRankSignals:
    """Tests for rank_signals function."""

    @pytest.fixture
    def sample_signals(self):
        """Create sample signals for ranking."""
        signals = [
            Signal(pd.Timestamp.now(), 'AAPL', 1, 0.5, 0.6, 0.01),
            Signal(pd.Timestamp.now(), 'GOOGL', 1, 0.9, 0.8, 0.03),
            Signal(pd.Timestamp.now(), 'MSFT', -1, 0.3, 0.4, -0.005),
            Signal(pd.Timestamp.now(), 'TSLA', 1, 0.7, 0.75, 0.02),
        ]
        return SignalBatch(signals=signals)

    def test_rank_by_strength(self, sample_signals):
        """Test ranking by strength."""
        ranked = rank_signals(sample_signals, by='strength')

        assert len(ranked) == 4
        # Highest strength first
        assert ranked[0].symbol == 'GOOGL'  # strength 0.9

    def test_rank_by_confidence(self, sample_signals):
        """Test ranking by confidence."""
        ranked = rank_signals(sample_signals, by='confidence')

        assert ranked[0].symbol == 'GOOGL'  # confidence 0.8

    def test_rank_top_n(self, sample_signals):
        """Test getting top N signals."""
        ranked = rank_signals(sample_signals, by='strength', top_n=2)

        assert len(ranked) == 2


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_pipeline(self, sample_features, sample_direction_target, sample_return_target):
        """Test complete training and prediction pipeline."""
        # Split data
        train_size = 150
        X_train = sample_features.iloc[:train_size]
        X_test = sample_features.iloc[train_size:]
        y_train_dir = sample_direction_target.iloc[:train_size]
        y_train_ret = sample_return_target.iloc[:train_size]

        # Train models
        classifier = DirectionClassifier(params={'n_estimators': 50})
        classifier.fit(X_train, y_train_dir)

        regressor = ReturnRegressor(params={'n_estimators': 50})
        regressor.fit(X_train, y_train_ret)

        # Create pipeline
        pipeline = PredictionPipeline(
            classifier=classifier,
            regressor=regressor
        )

        # Generate signals
        signals = pipeline.predict(X_test, symbol='AAPL')

        assert len(signals) == len(X_test)
        assert all(s.expected_return is not None for s in signals.signals)

    def test_walk_forward_to_signals(
        self, sample_features, sample_direction_target
    ):
        """Test walk-forward validation followed by signal generation."""
        # Validate model
        validator = WalkForwardValidator(
            model_class=DirectionClassifier,
            model_params={'n_estimators': 50},
            n_splits=3,
            test_size=30,
            verbose=False
        )
        results = validator.validate(sample_features, sample_direction_target)

        # Check validation completed
        assert results.n_folds == 3
        assert 'accuracy_mean' in results.aggregate_metrics

        # Train final model on all data
        classifier = DirectionClassifier(params={'n_estimators': 50})
        classifier.fit(sample_features, sample_direction_target)

        # Generate signals
        generator = SignalGenerator(confidence_threshold=0.5)
        predictions = classifier.predict(sample_features)
        probabilities = classifier.predict_proba(sample_features)[:, 1]

        batch = generator.generate_batch(
            symbols=['AAPL'] * len(sample_features),
            directions=predictions,
            probabilities=probabilities
        )

        # Filter by confidence
        high_confidence = batch.filter_by_confidence(min_confidence=0.6)
        assert len(high_confidence) <= len(batch)

    def test_model_save_load_predict(
        self, trained_classifier, sample_features, temp_model_dir
    ):
        """Test saving, loading, and predicting with model."""
        # Save model
        save_path = trained_classifier.save(temp_model_dir / 'test_model')

        # Load model
        loaded = DirectionClassifier.load(save_path)

        # Compare predictions
        orig_preds = trained_classifier.predict(sample_features)
        loaded_preds = loaded.predict(sample_features)

        np.testing.assert_array_equal(orig_preds, loaded_preds)


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_small_dataset(self):
        """Test with very small dataset."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(60, 5))
        y = (X.iloc[:, 0] > 0).astype(int)

        classifier = DirectionClassifier(params={'n_estimators': 30})
        classifier.fit(X, y)

        predictions = classifier.predict(X)
        assert len(predictions) == 60

    def test_single_class_target(self):
        """Test handling of single-class target raises appropriate error."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5))
        y = np.ones(100)  # All same class

        classifier = DirectionClassifier(params={'n_estimators': 30})

        # XGBoost requires at least 2 classes for classification
        with pytest.raises(ValueError, match="Invalid classes"):
            classifier.fit(X, y)

    def test_features_with_nan(self):
        """Test handling of NaN in features."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5))
        X.iloc[10, 2] = np.nan  # Add NaN
        y = (X.iloc[:, 0] > 0).astype(int)

        classifier = DirectionClassifier(params={'n_estimators': 30})

        # Should warn but not fail
        with pytest.warns(UserWarning):
            classifier.fit(X, y)

    def test_empty_signal_batch(self):
        """Test operations on empty signal batch."""
        batch = SignalBatch()

        assert len(batch) == 0
        df = batch.to_dataframe()
        assert len(df) == 0

        filtered = batch.filter_by_confidence(0.5)
        assert len(filtered) == 0

    def test_signal_to_dict(self):
        """Test signal serialization."""
        signal = Signal(
            timestamp=pd.Timestamp('2024-01-01'),
            symbol='AAPL',
            direction=1,
            strength=0.8,
            confidence=0.7,
            expected_return=0.02
        )

        d = signal.to_dict()
        assert d['symbol'] == 'AAPL'
        assert d['direction'] == 1
        assert d['strength'] == 0.8
