"""
XGBoost Models for Market Prediction.

Provides:
- DirectionClassifier: Predicts whether price goes up or down
- ReturnRegressor: Predicts expected returns
- Feature importance analysis
- Model persistence (save/load)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, Dict, List, Any, Tuple
import logging
import warnings

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

try:
    import xgboost as xgb
except ImportError:
    raise ImportError("xgboost is required. Install with: pip install xgboost")


logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """
    Container for model evaluation metrics.

    Attributes:
        model_type: Type of model ('classifier' or 'regressor')
        metrics: Dictionary of metric name to value
        feature_importance: DataFrame with feature importance scores
        n_samples: Number of samples used for evaluation
        timestamp: When the model was evaluated
    """
    model_type: str
    metrics: Dict[str, float]
    feature_importance: Optional[pd.DataFrame] = None
    n_samples: int = 0
    timestamp: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'model_type': self.model_type,
            'metrics': self.metrics,
            'n_samples': self.n_samples,
            'timestamp': self.timestamp,
        }
        if self.feature_importance is not None:
            result['feature_importance'] = self.feature_importance.to_dict()
        return result


@dataclass
class PredictionResult:
    """
    Container for model predictions.

    Attributes:
        predictions: Array of predictions
        probabilities: Array of class probabilities (classifier only)
        confidence: Array of prediction confidence scores
        feature_names: List of feature names used
        model_type: Type of model that generated predictions
    """
    predictions: np.ndarray
    probabilities: Optional[np.ndarray] = None
    confidence: Optional[np.ndarray] = None
    feature_names: List[str] = field(default_factory=list)
    model_type: str = 'classifier'

    def to_dataframe(self, index: Optional[pd.Index] = None) -> pd.DataFrame:
        """Convert predictions to DataFrame."""
        data = {'prediction': self.predictions}

        if self.probabilities is not None:
            if len(self.probabilities.shape) > 1:
                data['prob_down'] = self.probabilities[:, 0]
                data['prob_up'] = self.probabilities[:, 1]
            else:
                data['probability'] = self.probabilities

        if self.confidence is not None:
            data['confidence'] = self.confidence

        df = pd.DataFrame(data, index=index)
        return df


class BaseModel:
    """
    Base class for XGBoost models.

    Provides common functionality for training, evaluation,
    feature importance, and persistence.
    """

    DEFAULT_PARAMS = {
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
    }

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        scale_features: bool = True
    ):
        """
        Initialize the base model.

        Args:
            params: XGBoost parameters. Merged with defaults.
            scale_features: Whether to standardize features before training.
        """
        self.params = {**self.DEFAULT_PARAMS}
        if params:
            self.params.update(params)

        self.scale_features = scale_features
        self.scaler = StandardScaler() if scale_features else None
        self.model = None
        self.feature_names: List[str] = []
        self.is_fitted = False

    def _prepare_features(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        fit: bool = False
    ) -> np.ndarray:
        """Prepare features for model input."""
        if isinstance(X, pd.DataFrame):
            if fit:
                self.feature_names = X.columns.tolist()
            X = X.values
        else:
            X = np.asarray(X)
            if fit:
                # Set default feature names for numpy arrays
                self.feature_names = [f'f{i}' for i in range(X.shape[1])]

        if self.scale_features and self.scaler is not None:
            if fit:
                X = self.scaler.fit_transform(X)
            else:
                X = self.scaler.transform(X)

        return X

    def _validate_data(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Validate input data."""
        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = np.asarray(X)

        if np.any(np.isnan(X_arr)):
            warnings.warn("Features contain NaN values. Consider cleaning data first.")

        y_arr = None
        if y is not None:
            if isinstance(y, pd.Series):
                y_arr = y.values
            else:
                y_arr = np.asarray(y)

        return X_arr, y_arr

    def get_feature_importance(
        self,
        importance_type: str = 'gain'
    ) -> pd.DataFrame:
        """
        Get feature importance scores.

        Args:
            importance_type: Type of importance ('gain', 'weight', 'cover').

        Returns:
            DataFrame with feature names and importance scores.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first.")

        # Get importance from the underlying booster
        booster = self.model.get_booster()
        importance = booster.get_score(importance_type=importance_type)

        # If no feature names set, use the keys from importance
        if not self.feature_names:
            # XGBoost uses 'f0', 'f1', etc. as feature names by default
            n_features = len(importance) if importance else 0
            self.feature_names = [f'f{i}' for i in range(n_features)]

        # Map feature indices to names
        result = []
        for i, name in enumerate(self.feature_names):
            # XGBoost uses 'f0', 'f1', etc. as feature names by default
            key = f'f{i}'
            score = importance.get(key, 0.0)
            result.append({'feature': name, 'importance': score})

        # Handle empty result
        if not result:
            return pd.DataFrame(columns=['feature', 'importance', 'importance_pct'])

        df = pd.DataFrame(result)
        df = df.sort_values('importance', ascending=False).reset_index(drop=True)

        # Normalize to sum to 1
        total = df['importance'].sum()
        if total > 0:
            df['importance_pct'] = df['importance'] / total * 100
        else:
            df['importance_pct'] = 0.0

        return df

    def save(self, path: Union[str, Path]) -> Path:
        """
        Save model to disk.

        Args:
            path: File path (will add .joblib extension if needed).

        Returns:
            Path to saved file.
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model.")

        path = Path(path)
        if path.suffix != '.joblib':
            path = path.with_suffix('.joblib')

        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'params': self.params,
            'scale_features': self.scale_features,
            'model_type': self.__class__.__name__,
        }

        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")

        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'BaseModel':
        """
        Load model from disk.

        Args:
            path: Path to saved model file.

        Returns:
            Loaded model instance.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        model_data = joblib.load(path)

        # Create new instance
        instance = cls(
            params=model_data['params'],
            scale_features=model_data['scale_features']
        )
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.feature_names = model_data['feature_names']
        instance.is_fitted = True

        logger.info(f"Model loaded from {path}")
        return instance


class DirectionClassifier(BaseModel):
    """
    XGBoost classifier for predicting price direction.

    Predicts whether the next period's price will go up (1) or down (0).

    Example:
        >>> classifier = DirectionClassifier()
        >>> classifier.fit(X_train, y_train)
        >>> predictions = classifier.predict(X_test)
        >>> metrics = classifier.evaluate(X_test, y_test)
    """

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        scale_features: bool = True,
        class_weight: Optional[str] = 'balanced'
    ):
        """
        Initialize the direction classifier.

        Args:
            params: XGBoost parameters.
            scale_features: Whether to standardize features.
            class_weight: How to handle class imbalance ('balanced' or None).
        """
        super().__init__(params, scale_features)
        self.class_weight = class_weight

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        eval_set: Optional[List[Tuple]] = None,
        early_stopping_rounds: Optional[int] = None,
        verbose: bool = False
    ) -> 'DirectionClassifier':
        """
        Train the classifier.

        Args:
            X: Feature matrix.
            y: Target labels (0 or 1).
            eval_set: List of (X, y) tuples for evaluation during training.
            early_stopping_rounds: Stop if no improvement after N rounds.
            verbose: Whether to print training progress.

        Returns:
            self
        """
        X_prepared = self._prepare_features(X, fit=True)
        _, y_arr = self._validate_data(X, y)

        # Handle class weights
        scale_pos_weight = 1.0
        if self.class_weight == 'balanced':
            n_neg = np.sum(y_arr == 0)
            n_pos = np.sum(y_arr == 1)
            if n_pos > 0:
                scale_pos_weight = n_neg / n_pos

        model_params = {**self.params, 'scale_pos_weight': scale_pos_weight}

        self.model = xgb.XGBClassifier(**model_params)

        fit_params = {'verbose': verbose}
        if eval_set is not None:
            # Prepare eval set
            prepared_eval = []
            for X_eval, y_eval in eval_set:
                X_eval_prep = self._prepare_features(X_eval, fit=False)
                prepared_eval.append((X_eval_prep, y_eval))
            fit_params['eval_set'] = prepared_eval

        if early_stopping_rounds is not None:
            fit_params['early_stopping_rounds'] = early_stopping_rounds

        self.model.fit(X_prepared, y_arr, **fit_params)
        self.is_fitted = True

        logger.info(f"Classifier trained on {len(y_arr)} samples")
        return self

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Feature matrix.

        Returns:
            Array of predicted labels (0 or 1).
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first.")

        X_prepared = self._prepare_features(X, fit=False)
        return self.model.predict(X_prepared)

    def predict_proba(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Feature matrix.

        Returns:
            Array of shape (n_samples, 2) with probabilities for each class.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first.")

        X_prepared = self._prepare_features(X, fit=False)
        return self.model.predict_proba(X_prepared)

    def predict_with_confidence(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> PredictionResult:
        """
        Predict with full details including probabilities and confidence.

        Args:
            X: Feature matrix.

        Returns:
            PredictionResult with predictions, probabilities, and confidence.
        """
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)

        # Confidence is the probability of the predicted class
        confidence = np.max(probabilities, axis=1)

        return PredictionResult(
            predictions=predictions,
            probabilities=probabilities,
            confidence=confidence,
            feature_names=self.feature_names,
            model_type='classifier'
        )

    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> ModelMetrics:
        """
        Evaluate the classifier.

        Args:
            X: Feature matrix.
            y: True labels.

        Returns:
            ModelMetrics with accuracy, precision, recall, F1, and AUC.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first.")

        X_prepared = self._prepare_features(X, fit=False)
        _, y_arr = self._validate_data(X, y)

        y_pred = self.model.predict(X_prepared)
        y_proba = self.model.predict_proba(X_prepared)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_arr, y_pred),
            'precision': precision_score(y_arr, y_pred, zero_division=0),
            'recall': recall_score(y_arr, y_pred, zero_division=0),
            'f1': f1_score(y_arr, y_pred, zero_division=0),
        }

        # AUC requires both classes present
        if len(np.unique(y_arr)) > 1:
            metrics['auc'] = roc_auc_score(y_arr, y_proba)
        else:
            metrics['auc'] = 0.5

        feature_importance = self.get_feature_importance()

        return ModelMetrics(
            model_type='classifier',
            metrics=metrics,
            feature_importance=feature_importance,
            n_samples=len(y_arr),
            timestamp=pd.Timestamp.now().isoformat()
        )


class ReturnRegressor(BaseModel):
    """
    XGBoost regressor for predicting returns.

    Predicts the expected return for the next period.

    Example:
        >>> regressor = ReturnRegressor()
        >>> regressor.fit(X_train, y_train)
        >>> predictions = regressor.predict(X_test)
        >>> metrics = regressor.evaluate(X_test, y_test)
    """

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        scale_features: bool = True
    ):
        """
        Initialize the return regressor.

        Args:
            params: XGBoost parameters.
            scale_features: Whether to standardize features.
        """
        # Override some defaults for regression
        default_regression_params = {
            'objective': 'reg:squarederror',
        }
        combined_params = {**default_regression_params}
        if params:
            combined_params.update(params)

        super().__init__(combined_params, scale_features)

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        eval_set: Optional[List[Tuple]] = None,
        early_stopping_rounds: Optional[int] = None,
        verbose: bool = False
    ) -> 'ReturnRegressor':
        """
        Train the regressor.

        Args:
            X: Feature matrix.
            y: Target returns.
            eval_set: List of (X, y) tuples for evaluation during training.
            early_stopping_rounds: Stop if no improvement after N rounds.
            verbose: Whether to print training progress.

        Returns:
            self
        """
        X_prepared = self._prepare_features(X, fit=True)
        _, y_arr = self._validate_data(X, y)

        self.model = xgb.XGBRegressor(**self.params)

        fit_params = {'verbose': verbose}
        if eval_set is not None:
            prepared_eval = []
            for X_eval, y_eval in eval_set:
                X_eval_prep = self._prepare_features(X_eval, fit=False)
                prepared_eval.append((X_eval_prep, y_eval))
            fit_params['eval_set'] = prepared_eval

        if early_stopping_rounds is not None:
            fit_params['early_stopping_rounds'] = early_stopping_rounds

        self.model.fit(X_prepared, y_arr, **fit_params)
        self.is_fitted = True

        logger.info(f"Regressor trained on {len(y_arr)} samples")
        return self

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Predict returns.

        Args:
            X: Feature matrix.

        Returns:
            Array of predicted returns.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first.")

        X_prepared = self._prepare_features(X, fit=False)
        return self.model.predict(X_prepared)

    def predict_with_confidence(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        n_iterations: int = 100
    ) -> PredictionResult:
        """
        Predict with confidence intervals using bootstrap.

        Args:
            X: Feature matrix.
            n_iterations: Number of bootstrap iterations for confidence estimation.

        Returns:
            PredictionResult with predictions and confidence scores.
        """
        predictions = self.predict(X)

        # Estimate confidence based on prediction magnitude relative to training variance
        # Higher absolute prediction = lower confidence (more extreme)
        pred_std = np.std(predictions)
        if pred_std > 0:
            z_scores = np.abs(predictions - np.mean(predictions)) / pred_std
            # Convert z-score to confidence (higher z = lower confidence)
            confidence = 1 / (1 + z_scores)
        else:
            confidence = np.ones(len(predictions)) * 0.5

        return PredictionResult(
            predictions=predictions,
            probabilities=None,
            confidence=confidence,
            feature_names=self.feature_names,
            model_type='regressor'
        )

    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> ModelMetrics:
        """
        Evaluate the regressor.

        Args:
            X: Feature matrix.
            y: True returns.

        Returns:
            ModelMetrics with MSE, MAE, RMSE, and R-squared.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first.")

        X_prepared = self._prepare_features(X, fit=False)
        _, y_arr = self._validate_data(X, y)

        y_pred = self.model.predict(X_prepared)

        mse = mean_squared_error(y_arr, y_pred)
        metrics = {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'mae': mean_absolute_error(y_arr, y_pred),
            'r2': r2_score(y_arr, y_pred),
        }

        # Direction accuracy (did we get the sign right?)
        direction_correct = np.sign(y_pred) == np.sign(y_arr)
        metrics['direction_accuracy'] = np.mean(direction_correct)

        feature_importance = self.get_feature_importance()

        return ModelMetrics(
            model_type='regressor',
            metrics=metrics,
            feature_importance=feature_importance,
            n_samples=len(y_arr),
            timestamp=pd.Timestamp.now().isoformat()
        )


class EnsembleModel:
    """
    Ensemble combining classifier and regressor.

    Uses the classifier for direction and regressor for magnitude,
    combining them for more robust predictions.

    Example:
        >>> ensemble = EnsembleModel()
        >>> ensemble.fit(X_train, y_direction, y_return)
        >>> signals = ensemble.predict(X_test)
    """

    def __init__(
        self,
        classifier_params: Optional[Dict[str, Any]] = None,
        regressor_params: Optional[Dict[str, Any]] = None,
        scale_features: bool = True
    ):
        """
        Initialize the ensemble model.

        Args:
            classifier_params: Parameters for the direction classifier.
            regressor_params: Parameters for the return regressor.
            scale_features: Whether to standardize features.
        """
        self.classifier = DirectionClassifier(
            params=classifier_params,
            scale_features=scale_features
        )
        self.regressor = ReturnRegressor(
            params=regressor_params,
            scale_features=scale_features
        )
        self.is_fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y_direction: Union[pd.Series, np.ndarray],
        y_return: Union[pd.Series, np.ndarray],
        **kwargs
    ) -> 'EnsembleModel':
        """
        Train both models.

        Args:
            X: Feature matrix.
            y_direction: Direction labels (0 or 1).
            y_return: Return targets.
            **kwargs: Additional arguments passed to fit methods.

        Returns:
            self
        """
        self.classifier.fit(X, y_direction, **kwargs)
        self.regressor.fit(X, y_return, **kwargs)
        self.is_fitted = True
        return self

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Generate predictions from both models.

        Args:
            X: Feature matrix.

        Returns:
            Dictionary with 'direction', 'probability', 'return', and 'signal'.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first.")

        direction = self.classifier.predict(X)
        proba = self.classifier.predict_proba(X)
        returns = self.regressor.predict(X)

        # Combined signal: direction * probability * magnitude
        prob_up = proba[:, 1]
        signal = (2 * direction - 1) * prob_up * np.abs(returns)

        return {
            'direction': direction,
            'probability': prob_up,
            'return': returns,
            'signal': signal
        }

    def save(self, path: Union[str, Path]) -> Path:
        """Save both models."""
        path = Path(path)
        base = path.stem
        parent = path.parent
        parent.mkdir(parents=True, exist_ok=True)

        self.classifier.save(parent / f"{base}_classifier.joblib")
        self.regressor.save(parent / f"{base}_regressor.joblib")

        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'EnsembleModel':
        """Load both models."""
        path = Path(path)
        base = path.stem
        parent = path.parent

        instance = cls()
        instance.classifier = DirectionClassifier.load(
            parent / f"{base}_classifier.joblib"
        )
        instance.regressor = ReturnRegressor.load(
            parent / f"{base}_regressor.joblib"
        )
        instance.is_fitted = True

        return instance
