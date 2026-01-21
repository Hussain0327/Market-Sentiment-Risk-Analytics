#!/usr/bin/env python3
"""
ML Model Training Script.

Trains direction classifier and return regressor models for market prediction.
Uses walk-forward validation to evaluate model performance without lookahead bias.

Usage:
    python scripts/train_models.py                    # Train all symbols
    python scripts/train_models.py --symbol AAPL     # Train specific symbol
    python scripts/train_models.py --retrain         # Force retrain existing models
    python scripts/train_models.py --list            # List available symbols

Models are saved to data/models/{symbol}/ with metadata.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from src.ml import (
    DirectionClassifier,
    ReturnRegressor,
    WalkForwardValidator,
)
from src.features import FeatureBuilder
from dashboard.data_loader import DashboardDataLoader


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models"


def get_available_symbols() -> List[str]:
    """Get list of symbols with available data."""
    loader = DashboardDataLoader()
    return loader.get_available_symbols()


def load_data_for_symbol(symbol: str) -> tuple:
    """
    Load price and sentiment data for a symbol.

    Returns:
        Tuple of (prices_df, sentiment_df) or (None, None) if data not available.
    """
    loader = DashboardDataLoader()

    prices = loader.load_prices(symbol)
    sentiment = loader.load_daily_sentiment(symbol)

    if prices.empty:
        logger.warning(f"No price data found for {symbol}")
        return None, None

    return prices, sentiment


def build_features_and_targets(
    prices: pd.DataFrame,
    sentiment: pd.DataFrame,
    symbol: str,
    target_horizon: int = 1,
    min_samples: int = 100
) -> tuple:
    """
    Build features and targets for ML training.

    If sentiment data causes too few samples (due to date alignment),
    falls back to price/risk features only.

    Returns:
        Tuple of (X_features, y_direction, y_return) DataFrames/Series.
    """
    builder = FeatureBuilder()

    # Create targets first (based on prices only)
    y_direction = builder.create_target(
        prices, horizon=target_horizon, target_type='direction'
    )
    y_return = builder.create_target(
        prices, horizon=target_horizon, target_type='return'
    )

    # Prepare price index for alignment
    prices_prep = builder._prepare_prices(prices)
    y_direction.index = prices_prep.index
    y_return.index = prices_prep.index

    # Try building with sentiment first if available
    use_sentiment = not sentiment.empty

    if use_sentiment:
        logger.info("Attempting to build features with sentiment data...")
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
        if not features.empty:
            if isinstance(features.index, pd.DatetimeIndex):
                features.index = features.index.normalize()

            combined = features.join(y_direction.to_frame('y_direction'), how='inner')
            combined = combined.join(y_return.to_frame('y_return'), how='inner')
            combined = combined.dropna()

            if len(combined) >= min_samples:
                logger.info(f"Using features with sentiment: {len(combined)} samples")
            else:
                logger.warning(
                    f"Sentiment alignment resulted in only {len(combined)} samples. "
                    f"Falling back to price/risk features only."
                )
                use_sentiment = False

    if not use_sentiment:
        # Build without sentiment
        logger.info("Building features without sentiment data...")
        features = builder.build_features(
            prices=prices,
            sentiment=None,
            symbol=symbol,
            include_volume=True,
            include_garch=False,
            include_sentiment=False,
            include_risk=True
        )

        if features.empty:
            logger.warning(f"No features built for {symbol}")
            return None, None, None

        if isinstance(features.index, pd.DatetimeIndex):
            features.index = features.index.normalize()

        combined = features.join(y_direction.to_frame('y_direction'), how='inner')
        combined = combined.join(y_return.to_frame('y_return'), how='inner')
        combined = combined.dropna()

    # Separate features and targets
    feature_cols = [c for c in combined.columns
                   if c not in ['y_direction', 'y_return', 'symbol']]
    X = combined[feature_cols]
    y_dir = combined['y_direction']
    y_ret = combined['y_return']

    # Filter to only numeric columns (remove volatility_regime, etc.)
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X = X[numeric_cols]

    logger.info(f"Built dataset: {X.shape[0]} samples, {X.shape[1]} features")

    return X, y_dir, y_ret


def validate_model(
    model_class,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5
) -> Dict[str, float]:
    """
    Run walk-forward validation and return metrics.
    """
    validator = WalkForwardValidator(
        model_class=model_class,
        n_splits=n_splits,
        min_train_size=50,
        gap=1,  # 1-day gap to avoid lookahead
        purge_size=0,
        expanding=True,
        verbose=False
    )

    try:
        result = validator.validate(X, y)
        return result.aggregate_metrics
    except Exception as e:
        logger.warning(f"Validation failed: {e}")
        return {}


def train_and_save_models(
    symbol: str,
    X: pd.DataFrame,
    y_direction: pd.Series,
    y_return: pd.Series,
    retrain: bool = False
) -> Dict[str, Any]:
    """
    Train classifier and regressor models and save to disk.

    Returns:
        Dictionary with training results and metadata.
    """
    model_dir = MODELS_DIR / symbol
    metadata_path = model_dir / "metadata.json"

    # Check if models already exist
    if not retrain and metadata_path.exists():
        logger.info(f"Models already exist for {symbol}. Use --retrain to override.")
        with open(metadata_path) as f:
            return json.load(f)

    model_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "symbol": symbol,
        "trained_at": datetime.now().isoformat(),
        "n_samples": len(X),
        "n_features": X.shape[1],
        "feature_names": X.columns.tolist(),
    }

    # Validate and train classifier
    logger.info(f"Training direction classifier for {symbol}...")

    classifier_metrics = validate_model(DirectionClassifier, X, y_direction)
    results["classifier_metrics"] = classifier_metrics

    if classifier_metrics:
        accuracy = classifier_metrics.get('accuracy_mean', 0)
        logger.info(f"  Validation accuracy: {accuracy:.3f}")

    # Train final classifier on all data
    classifier = DirectionClassifier()
    classifier.fit(X, y_direction)
    classifier.save(model_dir / "model_classifier")
    results["classifier_saved"] = True

    # Validate and train regressor
    logger.info(f"Training return regressor for {symbol}...")

    regressor_metrics = validate_model(ReturnRegressor, X, y_return)
    results["regressor_metrics"] = regressor_metrics

    if regressor_metrics:
        r2 = regressor_metrics.get('r2_mean', 0)
        dir_acc = regressor_metrics.get('direction_accuracy_mean', 0)
        logger.info(f"  Validation R2: {r2:.3f}, Direction accuracy: {dir_acc:.3f}")

    # Train final regressor on all data
    regressor = ReturnRegressor()
    regressor.fit(X, y_return)
    regressor.save(model_dir / "model_regressor")
    results["regressor_saved"] = True

    # Save metadata
    with open(metadata_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Models saved to {model_dir}")

    return results


def train_symbol(symbol: str, retrain: bool = False) -> Optional[Dict[str, Any]]:
    """
    Full training pipeline for a single symbol.
    """
    logger.info(f"=" * 50)
    logger.info(f"Training models for {symbol}")
    logger.info(f"=" * 50)

    # Load data
    prices, sentiment = load_data_for_symbol(symbol)
    if prices is None:
        logger.error(f"Failed to load data for {symbol}")
        return None

    logger.info(f"Loaded {len(prices)} price records")
    if sentiment is not None and not sentiment.empty:
        logger.info(f"Loaded {len(sentiment)} sentiment records")

    # Build features
    X, y_direction, y_return = build_features_and_targets(
        prices, sentiment, symbol
    )

    if X is None or len(X) < 100:
        logger.error(f"Insufficient data for {symbol} (need at least 100 samples)")
        return None

    # Train and save
    results = train_and_save_models(
        symbol, X, y_direction, y_return, retrain=retrain
    )

    return results


def train_all_symbols(retrain: bool = False) -> Dict[str, Any]:
    """
    Train models for all available symbols.
    """
    symbols = get_available_symbols()
    logger.info(f"Found {len(symbols)} symbols: {symbols}")

    results = {}
    for symbol in symbols:
        try:
            result = train_symbol(symbol, retrain=retrain)
            if result:
                results[symbol] = result
        except Exception as e:
            logger.error(f"Failed to train {symbol}: {e}")
            results[symbol] = {"error": str(e)}

    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 50)

    for symbol, result in results.items():
        if "error" in result:
            logger.info(f"  {symbol}: FAILED - {result['error']}")
        else:
            acc = result.get("classifier_metrics", {}).get("accuracy_mean", 0)
            logger.info(f"  {symbol}: OK - Accuracy: {acc:.3f}")

    return results


def list_trained_models() -> List[str]:
    """List symbols with trained models."""
    if not MODELS_DIR.exists():
        return []

    symbols = []
    for path in MODELS_DIR.iterdir():
        if path.is_dir():
            metadata = path / "metadata.json"
            if metadata.exists():
                symbols.append(path.name)

    return sorted(symbols)


def main():
    parser = argparse.ArgumentParser(
        description="Train ML models for market prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/train_models.py                    # Train all symbols
    python scripts/train_models.py --symbol AAPL     # Train specific symbol
    python scripts/train_models.py --retrain         # Force retrain
    python scripts/train_models.py --list            # List available symbols
        """
    )

    parser.add_argument(
        "--symbol", "-s",
        type=str,
        help="Specific symbol to train (default: all symbols)"
    )

    parser.add_argument(
        "--retrain", "-r",
        action="store_true",
        help="Force retrain even if models exist"
    )

    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available symbols and trained models"
    )

    args = parser.parse_args()

    if args.list:
        available = get_available_symbols()
        trained = list_trained_models()

        print("\nAvailable symbols (with data):")
        for s in available:
            status = "[TRAINED]" if s in trained else ""
            print(f"  {s} {status}")

        print(f"\nTrained models: {len(trained)}/{len(available)}")
        return

    if args.symbol:
        result = train_symbol(args.symbol.upper(), retrain=args.retrain)
        if result and "error" not in result:
            print(f"\nSuccess! Model trained for {args.symbol.upper()}")
            acc = result.get("classifier_metrics", {}).get("accuracy_mean", 0)
            print(f"Validation accuracy: {acc:.3f}")
        else:
            print(f"\nFailed to train model for {args.symbol.upper()}")
            sys.exit(1)
    else:
        results = train_all_symbols(retrain=args.retrain)
        successes = sum(1 for r in results.values() if "error" not in r)
        print(f"\nCompleted: {successes}/{len(results)} symbols trained successfully")


if __name__ == "__main__":
    main()
