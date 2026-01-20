"""
Walk-Forward Cross-Validation for Time Series.

Provides:
- TimeSeriesSplit: Custom time series cross-validation
- WalkForwardValidator: Full validation pipeline with metrics
- No lookahead bias: Train only on past data

Key principle: Never train on future data to predict the past.
"""

from dataclasses import dataclass, field
from typing import Optional, Union, List, Generator, Tuple, Dict, Any, Type
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score

from .model import BaseModel, DirectionClassifier, ReturnRegressor, ModelMetrics


logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    """
    Result from a single validation fold.

    Attributes:
        fold_idx: Index of this fold
        train_start: Start index of training period
        train_end: End index of training period
        test_start: Start index of test period
        test_end: End index of test period
        metrics: Model evaluation metrics
        predictions: Test set predictions
        actuals: Test set actual values
        train_size: Number of training samples
        test_size: Number of test samples
    """
    fold_idx: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    metrics: ModelMetrics
    predictions: np.ndarray
    actuals: np.ndarray
    train_size: int = 0
    test_size: int = 0

    @property
    def train_dates(self) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
        """Return training date range if available."""
        return None  # Set externally if dates are available

    @property
    def test_dates(self) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
        """Return test date range if available."""
        return None


@dataclass
class ValidationResult:
    """
    Complete validation result across all folds.

    Attributes:
        folds: List of individual fold results
        aggregate_metrics: Averaged metrics across folds
        model_type: Type of model validated
        n_folds: Number of folds
        total_test_samples: Total samples used for testing
    """
    folds: List[FoldResult]
    aggregate_metrics: Dict[str, float] = field(default_factory=dict)
    model_type: str = 'classifier'
    n_folds: int = 0
    total_test_samples: int = 0

    def summary(self) -> pd.DataFrame:
        """Generate summary DataFrame of all folds."""
        rows = []
        for fold in self.folds:
            row = {
                'fold': fold.fold_idx,
                'train_size': fold.train_size,
                'test_size': fold.test_size,
                **fold.metrics.metrics
            }
            rows.append(row)

        df = pd.DataFrame(rows)

        # Add aggregate row
        agg_row = {'fold': 'mean', 'train_size': '', 'test_size': ''}
        for col in df.columns:
            if col not in ['fold', 'train_size', 'test_size']:
                agg_row[col] = df[col].mean()
        df = pd.concat([df, pd.DataFrame([agg_row])], ignore_index=True)

        return df

    def get_all_predictions(self) -> pd.DataFrame:
        """Combine predictions from all folds."""
        all_preds = []
        all_actuals = []
        all_folds = []

        for fold in self.folds:
            all_preds.extend(fold.predictions)
            all_actuals.extend(fold.actuals)
            all_folds.extend([fold.fold_idx] * len(fold.predictions))

        return pd.DataFrame({
            'fold': all_folds,
            'prediction': all_preds,
            'actual': all_actuals
        })


class TimeSeriesSplit:
    """
    Time series cross-validator with expanding or sliding window.

    Unlike sklearn's TimeSeriesSplit, this provides more flexibility:
    - Expanding window: Train on all historical data
    - Sliding window: Train on fixed-size recent window
    - Configurable gap between train and test
    - Minimum train size requirement

    Example:
        >>> splitter = TimeSeriesSplit(n_splits=5, test_size=20)
        >>> for train_idx, test_idx in splitter.split(X):
        ...     X_train, X_test = X[train_idx], X[test_idx]
        ...     # Train and evaluate
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: Optional[int] = None,
        train_size: Optional[int] = None,
        min_train_size: int = 50,
        gap: int = 0,
        expanding: bool = True
    ):
        """
        Initialize the time series splitter.

        Args:
            n_splits: Number of folds.
            test_size: Size of each test set. If None, automatically calculated.
            train_size: Maximum training size (for sliding window). If None, use all.
            min_train_size: Minimum required training samples.
            gap: Number of samples to skip between train and test (avoid leakage).
            expanding: If True, use all past data. If False, use sliding window.
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.min_train_size = min_train_size
        self.gap = gap
        self.expanding = expanding

    def split(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test indices for each fold.

        Args:
            X: Feature matrix (used only for length).
            y: Target (optional, used only for length if X not provided).

        Yields:
            Tuple of (train_indices, test_indices) for each fold.
        """
        n_samples = len(X)

        # Calculate test size if not provided
        test_size = self.test_size
        if test_size is None:
            # Reserve enough for all folds plus minimum training
            available = n_samples - self.min_train_size - self.gap
            test_size = max(1, available // self.n_splits)

        # Calculate split points
        # We work backwards from the end
        for fold_idx in range(self.n_splits):
            # Test set position (working backwards from most recent)
            test_end = n_samples - (fold_idx * test_size)
            test_start = test_end - test_size

            # Training set ends before the gap
            train_end = test_start - self.gap

            # Training set start
            if self.expanding:
                train_start = 0
            else:
                # Sliding window
                if self.train_size is not None:
                    train_start = max(0, train_end - self.train_size)
                else:
                    train_start = 0

            # Validate sizes
            actual_train_size = train_end - train_start
            if actual_train_size < self.min_train_size:
                logger.warning(
                    f"Fold {fold_idx}: Training size {actual_train_size} "
                    f"below minimum {self.min_train_size}, skipping."
                )
                continue

            if test_start >= train_end and test_start >= 0:
                # Valid fold - test starts after train ends
                train_indices = np.arange(train_start, train_end)
                test_indices = np.arange(test_start, test_end)

                yield train_indices, test_indices

    def get_n_splits(self) -> int:
        """Return the number of splits."""
        return self.n_splits


class PurgedTimeSeriesSplit(TimeSeriesSplit):
    """
    Time series split with purging to prevent leakage.

    Adds a purge period after training to remove samples that might
    have information about the test period (e.g., overlapping windows
    in feature calculation).

    Example:
        >>> splitter = PurgedTimeSeriesSplit(n_splits=5, purge_size=5)
        >>> for train_idx, test_idx in splitter.split(X):
        ...     # train_idx excludes last 5 samples before test
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: Optional[int] = None,
        purge_size: int = 5,
        embargo_size: int = 1,
        **kwargs
    ):
        """
        Initialize purged splitter.

        Args:
            n_splits: Number of folds.
            test_size: Size of each test set.
            purge_size: Samples to remove from end of training (before gap).
            embargo_size: Additional samples to skip after test before next train.
            **kwargs: Additional arguments for TimeSeriesSplit.
        """
        super().__init__(n_splits=n_splits, test_size=test_size, **kwargs)
        self.purge_size = purge_size
        self.embargo_size = embargo_size

    def split(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate purged train/test indices."""
        for train_indices, test_indices in super().split(X, y):
            # Remove purge period from training
            if self.purge_size > 0 and len(train_indices) > self.purge_size:
                train_indices = train_indices[:-self.purge_size]

            if len(train_indices) >= self.min_train_size:
                yield train_indices, test_indices


class WalkForwardValidator:
    """
    Complete walk-forward validation pipeline.

    Handles the full validation workflow:
    1. Split data into time-ordered folds
    2. Train model on each training set
    3. Evaluate on corresponding test set
    4. Aggregate metrics across folds

    This ensures no lookahead bias - models only see past data.

    Example:
        >>> validator = WalkForwardValidator(
        ...     model_class=DirectionClassifier,
        ...     n_splits=5
        ... )
        >>> results = validator.validate(X, y)
        >>> print(results.summary())
    """

    def __init__(
        self,
        model_class: Type[BaseModel] = DirectionClassifier,
        model_params: Optional[Dict[str, Any]] = None,
        n_splits: int = 5,
        test_size: Optional[int] = None,
        min_train_size: int = 50,
        gap: int = 0,
        purge_size: int = 0,
        expanding: bool = True,
        verbose: bool = True
    ):
        """
        Initialize the validator.

        Args:
            model_class: Model class to instantiate for each fold.
            model_params: Parameters to pass to model constructor.
            n_splits: Number of validation folds.
            test_size: Size of each test set.
            min_train_size: Minimum training samples required.
            gap: Gap between train and test sets.
            purge_size: Purge period to remove from training.
            expanding: Use expanding window (vs sliding).
            verbose: Print progress information.
        """
        self.model_class = model_class
        self.model_params = model_params or {}
        self.n_splits = n_splits
        self.test_size = test_size
        self.min_train_size = min_train_size
        self.gap = gap
        self.purge_size = purge_size
        self.expanding = expanding
        self.verbose = verbose

        # Create appropriate splitter
        if purge_size > 0:
            self.splitter = PurgedTimeSeriesSplit(
                n_splits=n_splits,
                test_size=test_size,
                min_train_size=min_train_size,
                gap=gap,
                purge_size=purge_size,
                expanding=expanding
            )
        else:
            self.splitter = TimeSeriesSplit(
                n_splits=n_splits,
                test_size=test_size,
                min_train_size=min_train_size,
                gap=gap,
                expanding=expanding
            )

    def validate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        **fit_kwargs
    ) -> ValidationResult:
        """
        Run walk-forward validation.

        Args:
            X: Feature matrix.
            y: Target values.
            **fit_kwargs: Additional arguments passed to model.fit().

        Returns:
            ValidationResult with all fold results and aggregated metrics.
        """
        if isinstance(X, pd.DataFrame):
            X_arr = X.values
            feature_names = X.columns.tolist()
        else:
            X_arr = np.asarray(X)
            feature_names = [f'f{i}' for i in range(X_arr.shape[1])]

        if isinstance(y, pd.Series):
            y_arr = y.values
        else:
            y_arr = np.asarray(y)

        fold_results = []
        is_classifier = issubclass(self.model_class, DirectionClassifier)

        for fold_idx, (train_idx, test_idx) in enumerate(self.splitter.split(X_arr)):
            if self.verbose:
                logger.info(
                    f"Fold {fold_idx + 1}: Train {len(train_idx)} samples, "
                    f"Test {len(test_idx)} samples"
                )

            # Split data
            X_train, X_test = X_arr[train_idx], X_arr[test_idx]
            y_train, y_test = y_arr[train_idx], y_arr[test_idx]

            # Create and train model
            model = self.model_class(params=self.model_params)
            model.fit(X_train, y_train, **fit_kwargs)

            # Generate predictions
            predictions = model.predict(X_test)

            # Evaluate
            metrics = model.evaluate(X_test, y_test)

            # Store fold result
            fold_result = FoldResult(
                fold_idx=fold_idx,
                train_start=int(train_idx[0]),
                train_end=int(train_idx[-1]),
                test_start=int(test_idx[0]),
                test_end=int(test_idx[-1]),
                metrics=metrics,
                predictions=predictions,
                actuals=y_test,
                train_size=len(train_idx),
                test_size=len(test_idx)
            )
            fold_results.append(fold_result)

        # Aggregate metrics
        aggregate_metrics = self._aggregate_metrics(fold_results)

        return ValidationResult(
            folds=fold_results,
            aggregate_metrics=aggregate_metrics,
            model_type='classifier' if is_classifier else 'regressor',
            n_folds=len(fold_results),
            total_test_samples=sum(f.test_size for f in fold_results)
        )

    def _aggregate_metrics(
        self,
        fold_results: List[FoldResult]
    ) -> Dict[str, float]:
        """Aggregate metrics across folds."""
        if not fold_results:
            return {}

        # Collect all metric names
        metric_names = set()
        for fold in fold_results:
            metric_names.update(fold.metrics.metrics.keys())

        # Calculate mean and std for each metric
        aggregate = {}
        for metric in metric_names:
            values = [
                fold.metrics.metrics.get(metric, np.nan)
                for fold in fold_results
            ]
            values = [v for v in values if not np.isnan(v)]
            if values:
                aggregate[f'{metric}_mean'] = np.mean(values)
                aggregate[f'{metric}_std'] = np.std(values)

        return aggregate

    def validate_multiple_horizons(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y_dict: Dict[int, Union[pd.Series, np.ndarray]],
        **fit_kwargs
    ) -> Dict[int, ValidationResult]:
        """
        Validate across multiple prediction horizons.

        Args:
            X: Feature matrix.
            y_dict: Dictionary mapping horizon -> target values.
            **fit_kwargs: Additional arguments for model.fit().

        Returns:
            Dictionary mapping horizon -> ValidationResult.
        """
        results = {}
        for horizon, y in y_dict.items():
            if self.verbose:
                logger.info(f"\nValidating horizon: {horizon}")
            results[horizon] = self.validate(X, y, **fit_kwargs)
        return results


def cross_validate(
    model_class: Type[BaseModel],
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    n_splits: int = 5,
    **kwargs
) -> ValidationResult:
    """
    Convenience function for quick cross-validation.

    Args:
        model_class: Model class to use.
        X: Feature matrix.
        y: Target values.
        n_splits: Number of folds.
        **kwargs: Additional arguments for WalkForwardValidator.

    Returns:
        ValidationResult with all metrics.

    Example:
        >>> results = cross_validate(DirectionClassifier, X, y, n_splits=5)
        >>> print(f"Mean accuracy: {results.aggregate_metrics['accuracy_mean']:.3f}")
    """
    validator = WalkForwardValidator(
        model_class=model_class,
        n_splits=n_splits,
        **kwargs
    )
    return validator.validate(X, y)


def compare_models(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    models: Dict[str, Tuple[Type[BaseModel], Dict[str, Any]]],
    n_splits: int = 5,
    **kwargs
) -> pd.DataFrame:
    """
    Compare multiple models using walk-forward validation.

    Args:
        X: Feature matrix.
        y: Target values.
        models: Dictionary of model_name -> (model_class, model_params).
        n_splits: Number of validation folds.
        **kwargs: Additional arguments for validators.

    Returns:
        DataFrame comparing model performance.

    Example:
        >>> models = {
        ...     'baseline': (DirectionClassifier, {'n_estimators': 50}),
        ...     'complex': (DirectionClassifier, {'n_estimators': 200, 'max_depth': 7}),
        ... }
        >>> comparison = compare_models(X, y, models)
        >>> print(comparison)
    """
    results = []

    for name, (model_class, model_params) in models.items():
        logger.info(f"\nValidating model: {name}")

        validator = WalkForwardValidator(
            model_class=model_class,
            model_params=model_params,
            n_splits=n_splits,
            **kwargs
        )
        result = validator.validate(X, y)

        row = {'model': name, 'n_folds': result.n_folds}
        row.update(result.aggregate_metrics)
        results.append(row)

    return pd.DataFrame(results)
