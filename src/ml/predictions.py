"""
Signal Generation and Prediction Pipeline.

Provides:
- SignalGenerator: Convert model predictions to trading signals
- PredictionPipeline: End-to-end prediction workflow
- Confidence-weighted signal generation
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, Dict, List, Any, Tuple
import logging

import numpy as np
import pandas as pd

from .model import DirectionClassifier, ReturnRegressor, EnsembleModel, BaseModel


logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """
    Trading signal with metadata.

    Attributes:
        timestamp: When the signal was generated
        symbol: Asset symbol
        direction: Signal direction (1=long, -1=short, 0=neutral)
        strength: Signal strength [0, 1]
        confidence: Model confidence [0, 1]
        expected_return: Expected return from model
        features_used: Number of features in prediction
        model_type: Type of model that generated signal
    """
    timestamp: pd.Timestamp
    symbol: str
    direction: int
    strength: float
    confidence: float
    expected_return: Optional[float] = None
    features_used: int = 0
    model_type: str = 'classifier'

    @property
    def is_bullish(self) -> bool:
        """Check if signal is bullish."""
        return self.direction > 0

    @property
    def is_bearish(self) -> bool:
        """Check if signal is bearish."""
        return self.direction < 0

    @property
    def is_neutral(self) -> bool:
        """Check if signal is neutral."""
        return self.direction == 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'direction': self.direction,
            'strength': self.strength,
            'confidence': self.confidence,
            'expected_return': self.expected_return,
            'features_used': self.features_used,
            'model_type': self.model_type,
        }


@dataclass
class SignalBatch:
    """
    Collection of signals for multiple assets or time periods.

    Attributes:
        signals: List of individual signals
        generated_at: Timestamp of batch generation
        model_version: Version identifier for the model used
    """
    signals: List[Signal] = field(default_factory=list)
    generated_at: Optional[pd.Timestamp] = None
    model_version: Optional[str] = None

    def __len__(self) -> int:
        return len(self.signals)

    def __iter__(self):
        return iter(self.signals)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all signals to DataFrame."""
        if not self.signals:
            return pd.DataFrame()

        rows = [s.to_dict() for s in self.signals]
        df = pd.DataFrame(rows)
        return df

    def filter_by_confidence(self, min_confidence: float = 0.6) -> 'SignalBatch':
        """Filter signals by minimum confidence."""
        filtered = [s for s in self.signals if s.confidence >= min_confidence]
        return SignalBatch(
            signals=filtered,
            generated_at=self.generated_at,
            model_version=self.model_version
        )

    def filter_by_strength(self, min_strength: float = 0.5) -> 'SignalBatch':
        """Filter signals by minimum strength."""
        filtered = [s for s in self.signals if abs(s.strength) >= min_strength]
        return SignalBatch(
            signals=filtered,
            generated_at=self.generated_at,
            model_version=self.model_version
        )

    def get_by_symbol(self, symbol: str) -> List[Signal]:
        """Get signals for a specific symbol."""
        return [s for s in self.signals if s.symbol == symbol]

    def get_bullish(self) -> List[Signal]:
        """Get all bullish signals."""
        return [s for s in self.signals if s.is_bullish]

    def get_bearish(self) -> List[Signal]:
        """Get all bearish signals."""
        return [s for s in self.signals if s.is_bearish]


class SignalGenerator:
    """
    Convert model predictions to trading signals.

    Applies thresholds, confidence weighting, and signal normalization
    to transform raw predictions into actionable signals.

    Example:
        >>> generator = SignalGenerator(confidence_threshold=0.6)
        >>> signal = generator.generate(
        ...     symbol='AAPL',
        ...     direction=1,
        ...     probability=0.72,
        ...     expected_return=0.015
        ... )
        >>> print(f"Signal: {signal.direction}, Strength: {signal.strength:.2f}")
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        strength_threshold: float = 0.0,
        neutral_zone: float = 0.05,
        use_expected_return: bool = True
    ):
        """
        Initialize the signal generator.

        Args:
            confidence_threshold: Minimum confidence to generate signal.
            strength_threshold: Minimum strength to emit non-neutral signal.
            neutral_zone: Probability zone around 0.5 treated as neutral.
            use_expected_return: Weight signals by expected return magnitude.
        """
        self.confidence_threshold = confidence_threshold
        self.strength_threshold = strength_threshold
        self.neutral_zone = neutral_zone
        self.use_expected_return = use_expected_return

    def generate(
        self,
        symbol: str,
        direction: int,
        probability: float,
        expected_return: Optional[float] = None,
        timestamp: Optional[pd.Timestamp] = None,
        features_used: int = 0,
        model_type: str = 'classifier'
    ) -> Signal:
        """
        Generate a single trading signal.

        Args:
            symbol: Asset symbol.
            direction: Raw direction (0 or 1 from classifier).
            probability: Probability of up movement.
            expected_return: Expected return from regressor.
            timestamp: Signal timestamp (default: now).
            features_used: Number of features used in prediction.
            model_type: Type of model generating the signal.

        Returns:
            Signal object with computed strength and confidence.
        """
        if timestamp is None:
            timestamp = pd.Timestamp.now()

        # Calculate confidence as distance from 0.5
        confidence = abs(probability - 0.5) * 2  # Scale to [0, 1]

        # Determine direction with neutral zone
        if abs(probability - 0.5) < self.neutral_zone:
            final_direction = 0
        else:
            final_direction = 1 if probability > 0.5 else -1

        # Calculate signal strength
        strength = self._calculate_strength(
            probability=probability,
            expected_return=expected_return,
            confidence=confidence
        )

        # Apply thresholds
        if confidence < self.confidence_threshold:
            final_direction = 0
            strength = 0.0

        if abs(strength) < self.strength_threshold:
            final_direction = 0

        return Signal(
            timestamp=timestamp,
            symbol=symbol,
            direction=final_direction,
            strength=strength,
            confidence=confidence,
            expected_return=expected_return,
            features_used=features_used,
            model_type=model_type
        )

    def _calculate_strength(
        self,
        probability: float,
        expected_return: Optional[float],
        confidence: float
    ) -> float:
        """
        Calculate signal strength.

        Combines probability, expected return, and confidence
        into a single strength measure.
        """
        # Base strength from probability
        base_strength = (probability - 0.5) * 2  # [-1, 1]

        # Weight by expected return if available
        if self.use_expected_return and expected_return is not None:
            # Scale return by typical daily return magnitude (~2%)
            return_factor = np.clip(expected_return / 0.02, -2, 2)
            strength = base_strength * (1 + abs(return_factor)) / 2
        else:
            strength = base_strength

        # Weight by confidence
        strength = strength * confidence

        return float(np.clip(strength, -1, 1))

    def generate_batch(
        self,
        symbols: List[str],
        directions: np.ndarray,
        probabilities: np.ndarray,
        expected_returns: Optional[np.ndarray] = None,
        timestamps: Optional[List[pd.Timestamp]] = None,
        features_used: int = 0,
        model_type: str = 'classifier'
    ) -> SignalBatch:
        """
        Generate signals for multiple predictions.

        Args:
            symbols: List of asset symbols.
            directions: Array of direction predictions.
            probabilities: Array of probabilities.
            expected_returns: Array of expected returns (optional).
            timestamps: List of timestamps (optional).
            features_used: Number of features used.
            model_type: Model type identifier.

        Returns:
            SignalBatch containing all generated signals.
        """
        n = len(symbols)
        if timestamps is None:
            now = pd.Timestamp.now()
            timestamps = [now] * n

        if expected_returns is None:
            expected_returns = [None] * n

        signals = []
        for i in range(n):
            signal = self.generate(
                symbol=symbols[i],
                direction=int(directions[i]),
                probability=float(probabilities[i]),
                expected_return=expected_returns[i] if expected_returns[i] is not None else None,
                timestamp=timestamps[i],
                features_used=features_used,
                model_type=model_type
            )
            signals.append(signal)

        return SignalBatch(
            signals=signals,
            generated_at=pd.Timestamp.now(),
            model_version=model_type
        )


class PredictionPipeline:
    """
    End-to-end prediction and signal generation pipeline.

    Combines feature processing, model prediction, and signal generation
    into a single workflow.

    Example:
        >>> pipeline = PredictionPipeline(
        ...     classifier=trained_classifier,
        ...     regressor=trained_regressor
        ... )
        >>> signals = pipeline.predict(features, symbol='AAPL')
        >>> for signal in signals:
        ...     print(f"{signal.symbol}: {signal.direction} ({signal.confidence:.2f})")
    """

    def __init__(
        self,
        classifier: Optional[DirectionClassifier] = None,
        regressor: Optional[ReturnRegressor] = None,
        ensemble: Optional[EnsembleModel] = None,
        signal_generator: Optional[SignalGenerator] = None
    ):
        """
        Initialize the prediction pipeline.

        Args:
            classifier: Trained direction classifier.
            regressor: Trained return regressor.
            ensemble: Trained ensemble model (alternative to classifier+regressor).
            signal_generator: Signal generator (default created if not provided).
        """
        self.classifier = classifier
        self.regressor = regressor
        self.ensemble = ensemble
        self.signal_generator = signal_generator or SignalGenerator()

        # Validate we have at least one model
        if classifier is None and regressor is None and ensemble is None:
            raise ValueError(
                "Must provide at least one of: classifier, regressor, or ensemble"
            )

    def predict(
        self,
        features: pd.DataFrame,
        symbol: str,
        timestamps: Optional[pd.DatetimeIndex] = None
    ) -> SignalBatch:
        """
        Generate predictions and signals for a single symbol.

        Args:
            features: Feature DataFrame (rows are time periods).
            symbol: Asset symbol.
            timestamps: Timestamps for each row (default: use index).

        Returns:
            SignalBatch with signals for each time period.
        """
        n_samples = len(features)

        if timestamps is None:
            if isinstance(features.index, pd.DatetimeIndex):
                timestamps = features.index.tolist()
            else:
                timestamps = [pd.Timestamp.now()] * n_samples

        # Get predictions from available models
        if self.ensemble is not None:
            preds = self.ensemble.predict(features)
            directions = preds['direction']
            probabilities = preds['probability']
            expected_returns = preds['return']
            model_type = 'ensemble'
        else:
            # Use individual models
            directions = np.zeros(n_samples)
            probabilities = np.full(n_samples, 0.5)
            expected_returns = None

            if self.classifier is not None:
                directions = self.classifier.predict(features)
                proba = self.classifier.predict_proba(features)
                probabilities = proba[:, 1]
                model_type = 'classifier'

            if self.regressor is not None:
                expected_returns = self.regressor.predict(features)
                model_type = 'regressor' if self.classifier is None else 'combined'

        # Generate signals
        symbols = [symbol] * n_samples
        return self.signal_generator.generate_batch(
            symbols=symbols,
            directions=directions,
            probabilities=probabilities,
            expected_returns=expected_returns,
            timestamps=timestamps,
            features_used=features.shape[1],
            model_type=model_type
        )

    def predict_latest(
        self,
        features: pd.DataFrame,
        symbol: str
    ) -> Signal:
        """
        Generate signal for the most recent data point only.

        Args:
            features: Feature DataFrame (uses last row).
            symbol: Asset symbol.

        Returns:
            Single Signal for the latest time period.
        """
        # Get last row
        latest = features.iloc[[-1]]

        if isinstance(features.index, pd.DatetimeIndex):
            timestamp = features.index[-1]
        else:
            timestamp = pd.Timestamp.now()

        batch = self.predict(latest, symbol, timestamps=[timestamp])
        return batch.signals[0]

    def predict_multiple_symbols(
        self,
        features_dict: Dict[str, pd.DataFrame]
    ) -> SignalBatch:
        """
        Generate signals for multiple symbols.

        Args:
            features_dict: Dictionary of symbol -> features DataFrame.

        Returns:
            SignalBatch with signals for all symbols (latest only).
        """
        all_signals = []

        for symbol, features in features_dict.items():
            try:
                signal = self.predict_latest(features, symbol)
                all_signals.append(signal)
            except Exception as e:
                logger.warning(f"Failed to generate signal for {symbol}: {e}")

        return SignalBatch(
            signals=all_signals,
            generated_at=pd.Timestamp.now()
        )

    @classmethod
    def from_saved_models(
        cls,
        model_dir: Union[str, Path],
        model_name: str = 'model'
    ) -> 'PredictionPipeline':
        """
        Create pipeline from saved model files.

        Args:
            model_dir: Directory containing saved models.
            model_name: Base name for model files.

        Returns:
            PredictionPipeline with loaded models.
        """
        model_dir = Path(model_dir)

        classifier = None
        regressor = None

        # Try to load classifier
        classifier_path = model_dir / f"{model_name}_classifier.joblib"
        if classifier_path.exists():
            classifier = DirectionClassifier.load(classifier_path)

        # Try to load regressor
        regressor_path = model_dir / f"{model_name}_regressor.joblib"
        if regressor_path.exists():
            regressor = ReturnRegressor.load(regressor_path)

        if classifier is None and regressor is None:
            raise FileNotFoundError(
                f"No model files found in {model_dir} with name {model_name}"
            )

        return cls(classifier=classifier, regressor=regressor)


def generate_signals_from_features(
    features: pd.DataFrame,
    symbol: str,
    classifier: DirectionClassifier,
    regressor: Optional[ReturnRegressor] = None,
    confidence_threshold: float = 0.6,
    strength_threshold: float = 0.3
) -> pd.DataFrame:
    """
    Convenience function to generate signals from features.

    Args:
        features: Feature DataFrame.
        symbol: Asset symbol.
        classifier: Trained classifier.
        regressor: Trained regressor (optional).
        confidence_threshold: Minimum confidence for signal.
        strength_threshold: Minimum strength for signal.

    Returns:
        DataFrame with signals for each time period.

    Example:
        >>> signals_df = generate_signals_from_features(
        ...     features, 'AAPL', classifier,
        ...     confidence_threshold=0.6
        ... )
        >>> print(signals_df[signals_df['direction'] != 0])
    """
    generator = SignalGenerator(
        confidence_threshold=confidence_threshold,
        strength_threshold=strength_threshold
    )

    pipeline = PredictionPipeline(
        classifier=classifier,
        regressor=regressor,
        signal_generator=generator
    )

    batch = pipeline.predict(features, symbol)
    return batch.to_dataframe()


def rank_signals(
    signals: SignalBatch,
    by: str = 'strength',
    top_n: Optional[int] = None,
    ascending: bool = False
) -> List[Signal]:
    """
    Rank signals by a given metric.

    Args:
        signals: SignalBatch to rank.
        by: Metric to rank by ('strength', 'confidence', 'expected_return').
        top_n: Return only top N signals (None for all).
        ascending: Sort in ascending order.

    Returns:
        List of Signal objects sorted by the specified metric.
    """
    if not signals.signals:
        return []

    if by == 'strength':
        key = lambda s: abs(s.strength)
    elif by == 'confidence':
        key = lambda s: s.confidence
    elif by == 'expected_return':
        key = lambda s: abs(s.expected_return) if s.expected_return else 0
    else:
        raise ValueError(f"Unknown ranking metric: {by}")

    sorted_signals = sorted(signals.signals, key=key, reverse=not ascending)

    if top_n is not None:
        sorted_signals = sorted_signals[:top_n]

    return sorted_signals
