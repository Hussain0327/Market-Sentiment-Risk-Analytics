"""
ML Predictions API routes.

Provides endpoints for ML-based market predictions.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Query
import pandas as pd
import numpy as np

from api.schemas import (
    MLPredictionResponse,
    MLPredictionDataPoint,
    MLModelsListResponse,
    ModelInfo,
)
from dashboard.data_loader import DashboardDataLoader
from dashboard.remote_loader import is_remote_mode, get_remote_loader


router = APIRouter()
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "data" / "models"

# Cache for loaded models
_model_cache = {}
_data_loader: Optional[DashboardDataLoader] = None


def get_data_loader() -> DashboardDataLoader:
    """Get or create the data loader instance."""
    global _data_loader
    if _data_loader is None:
        _data_loader = DashboardDataLoader()
    return _data_loader


def get_models_dir() -> Path:
    """Get the models directory, using remote loader if in cloud mode."""
    if is_remote_mode():
        try:
            loader = get_remote_loader()
            return loader.get_models_dir()
        except Exception as e:
            logger.warning(f"Remote model loading failed: {e}. Using local fallback.")

    return MODELS_DIR


def get_trained_symbols() -> List[str]:
    """Get list of symbols with trained models."""
    if is_remote_mode():
        try:
            loader = get_remote_loader()
            return loader.get_trained_symbols()
        except Exception:
            pass

    models_dir = get_models_dir()
    if not models_dir.exists():
        return []

    symbols = []
    for path in models_dir.iterdir():
        if path.is_dir():
            metadata = path / "metadata.json"
            if metadata.exists():
                symbols.append(path.name)

    return sorted(symbols)


def load_model_metadata(symbol: str) -> Optional[dict]:
    """Load metadata for a trained model."""
    if is_remote_mode():
        try:
            loader = get_remote_loader()
            return loader.load_model_metadata(symbol)
        except Exception:
            pass

    models_dir = get_models_dir()
    metadata_path = models_dir / symbol / "metadata.json"
    if not metadata_path.exists():
        return None

    with open(metadata_path) as f:
        return json.load(f)


def load_models(symbol: str):
    """Load classifier and regressor models for a symbol."""
    from src.ml import DirectionClassifier, ReturnRegressor

    if symbol in _model_cache:
        return _model_cache[symbol]

    models_dir = get_models_dir()
    model_dir = models_dir / symbol
    classifier_path = model_dir / "model_classifier.joblib"
    regressor_path = model_dir / "model_regressor.joblib"

    if not classifier_path.exists() or not regressor_path.exists():
        return None, None

    try:
        classifier = DirectionClassifier.load(classifier_path)
        regressor = ReturnRegressor.load(regressor_path)
        _model_cache[symbol] = (classifier, regressor)
        return classifier, regressor
    except Exception as e:
        logger.error(f"Error loading models for {symbol}: {e}")
        return None, None


def build_features_for_prediction(symbol: str, min_samples: int = 10) -> Optional[pd.DataFrame]:
    """
    Build features for making predictions.

    If sentiment data causes too few samples, falls back to price/risk features only.
    """
    from src.features import FeatureBuilder

    loader = get_data_loader()

    prices = loader.load_prices(symbol)
    sentiment = loader.load_daily_sentiment(symbol)

    if prices.empty:
        return None

    builder = FeatureBuilder()

    # Try with sentiment first if available
    if not sentiment.empty:
        features = builder.build_features(
            prices=prices,
            sentiment=sentiment,
            symbol=symbol,
            include_volume=True,
            include_garch=False,
            include_sentiment=True,
            include_risk=True
        )

        # Check if we got enough samples
        if not features.empty and len(features) >= min_samples:
            return features

        logger.info(f"Sentiment alignment resulted in {len(features) if not features.empty else 0} samples, using price/risk features only")

    # Fall back to price/risk features only
    features = builder.build_features(
        prices=prices,
        sentiment=None,
        symbol=symbol,
        include_volume=True,
        include_garch=False,
        include_sentiment=False,
        include_risk=True
    )

    return features


def generate_predictions(
    symbol: str,
    features: pd.DataFrame,
    classifier,
    regressor,
    days: int = 30
) -> List[MLPredictionDataPoint]:
    """Generate predictions for recent dates."""
    # Use last N days of data
    if len(features) > days:
        features = features.tail(days)

    # Get feature columns (exclude non-numeric and symbol)
    feature_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    X = features[feature_cols]

    # Handle any NaN values
    X = X.fillna(0)

    predictions = []

    try:
        # Get predictions
        directions = classifier.predict(X)
        probabilities = classifier.predict_proba(X)
        expected_returns = regressor.predict(X)

        for i, (idx, row) in enumerate(X.iterrows()):
            direction_val = int(directions[i])
            prob_up = float(probabilities[i, 1])

            # Determine direction string
            if prob_up > 0.55:
                direction_str = "bullish"
            elif prob_up < 0.45:
                direction_str = "bearish"
            else:
                direction_str = "neutral"

            # Confidence is distance from 0.5
            confidence = abs(prob_up - 0.5) * 2

            # Signal strength combines direction and confidence
            signal_strength = (2 * direction_val - 1) * confidence

            # Convert expected return to percentage
            exp_ret = float(expected_returns[i]) * 100

            # Get date
            if isinstance(idx, pd.Timestamp):
                date = idx.to_pydatetime()
            else:
                date = datetime.now()

            predictions.append(MLPredictionDataPoint(
                date=date,
                direction=direction_str,
                confidence=round(confidence, 3),
                expected_return=round(exp_ret, 3),
                signal_strength=round(signal_strength, 3)
            ))

    except Exception as e:
        logger.error(f"Error generating predictions for {symbol}: {e}")
        return []

    return predictions


@router.get("/", response_model=MLModelsListResponse)
def list_trained_models():
    """
    List all symbols with trained ML models.

    Returns a list of symbols that have trained models available
    for generating predictions.
    """
    symbols = get_trained_symbols()
    return MLModelsListResponse(symbols=symbols, count=len(symbols))


@router.get("/{symbol}", response_model=MLPredictionResponse)
def get_predictions(
    symbol: str,
    days: int = Query(default=30, ge=1, le=365, description="Days of prediction history")
):
    """
    Get ML predictions for a symbol.

    Returns the latest ML-based prediction along with model info
    and prediction history. If no trained model exists, returns
    helpful instructions for training.
    """
    symbol = symbol.upper()

    # Check if model exists
    metadata = load_model_metadata(symbol)

    if metadata is None:
        return MLPredictionResponse(
            symbol=symbol,
            model_available=False,
            latest_prediction=None,
            model_info=None,
            history=[],
            message=f"No trained model for {symbol}. Run: python scripts/train_models.py --symbol {symbol}"
        )

    # Load models
    classifier, regressor = load_models(symbol)

    if classifier is None or regressor is None:
        return MLPredictionResponse(
            symbol=symbol,
            model_available=False,
            latest_prediction=None,
            model_info=None,
            history=[],
            message=f"Error loading model for {symbol}. Try retraining: python scripts/train_models.py --symbol {symbol} --retrain"
        )

    # Build features
    features = build_features_for_prediction(symbol)

    if features is None or features.empty:
        return MLPredictionResponse(
            symbol=symbol,
            model_available=True,
            latest_prediction=None,
            model_info=None,
            history=[],
            message=f"No recent data available for {symbol} to generate predictions."
        )

    # Generate predictions
    predictions = generate_predictions(
        symbol, features, classifier, regressor, days=days
    )

    # Build model info
    classifier_metrics = metadata.get("classifier_metrics", {})
    regressor_metrics = metadata.get("regressor_metrics", {})

    trained_at = None
    if metadata.get("trained_at"):
        try:
            trained_at = datetime.fromisoformat(metadata["trained_at"])
        except (ValueError, TypeError):
            pass

    model_info = ModelInfo(
        trained_at=trained_at,
        accuracy=classifier_metrics.get("accuracy_mean"),
        r2_score=regressor_metrics.get("r2_mean"),
        direction_accuracy=regressor_metrics.get("direction_accuracy_mean"),
        n_samples=metadata.get("n_samples"),
        n_features=metadata.get("n_features")
    )

    # Get latest prediction
    latest = predictions[-1] if predictions else None

    return MLPredictionResponse(
        symbol=symbol,
        model_available=True,
        latest_prediction=latest,
        model_info=model_info,
        history=predictions,
        message=None
    )


@router.get("/{symbol}/latest", response_model=MLPredictionDataPoint)
def get_latest_prediction(symbol: str):
    """
    Get only the latest ML prediction for a symbol.

    Returns just the most recent prediction without history.
    Useful for quick status checks.
    """
    symbol = symbol.upper()

    # Check if model exists
    metadata = load_model_metadata(symbol)

    if metadata is None:
        raise HTTPException(
            status_code=404,
            detail=f"No trained model for {symbol}. Run: python scripts/train_models.py --symbol {symbol}"
        )

    # Load models
    classifier, regressor = load_models(symbol)

    if classifier is None or regressor is None:
        raise HTTPException(
            status_code=500,
            detail=f"Error loading model for {symbol}"
        )

    # Build features
    features = build_features_for_prediction(symbol)

    if features is None or features.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No recent data available for {symbol}"
        )

    # Generate prediction for latest date only
    predictions = generate_predictions(
        symbol, features, classifier, regressor, days=1
    )

    if not predictions:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate prediction for {symbol}"
        )

    return predictions[-1]
