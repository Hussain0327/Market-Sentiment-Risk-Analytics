"""
Machine learning module.

Provides:
- XGBoost models for prediction (classifier and regressor)
- Walk-forward cross-validation
- Signal generation with confidence scores
- Model persistence (save/load)
"""

from .model import (
    ModelMetrics,
    PredictionResult,
    BaseModel,
    DirectionClassifier,
    ReturnRegressor,
    EnsembleModel,
)

from .validation import (
    FoldResult,
    ValidationResult,
    TimeSeriesSplit,
    PurgedTimeSeriesSplit,
    WalkForwardValidator,
    cross_validate,
    compare_models,
)

from .predictions import (
    Signal,
    SignalBatch,
    SignalGenerator,
    PredictionPipeline,
    generate_signals_from_features,
    rank_signals,
)


__all__ = [
    # Model classes
    'ModelMetrics',
    'PredictionResult',
    'BaseModel',
    'DirectionClassifier',
    'ReturnRegressor',
    'EnsembleModel',
    # Validation classes
    'FoldResult',
    'ValidationResult',
    'TimeSeriesSplit',
    'PurgedTimeSeriesSplit',
    'WalkForwardValidator',
    'cross_validate',
    'compare_models',
    # Signal classes
    'Signal',
    'SignalBatch',
    'SignalGenerator',
    'PredictionPipeline',
    'generate_signals_from_features',
    'rank_signals',
]
