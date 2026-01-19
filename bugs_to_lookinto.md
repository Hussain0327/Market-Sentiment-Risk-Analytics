# Bugs & Issues to Look Into

## Critical: FutureWarnings (Will Break in Future Pandas Versions)

### 1. Deprecated Index Assignment
**File:** `src/features/price_features.py:473-474`

```python
obv_sign[0] = 0  # First value has no direction
```

**Warning:**
```
FutureWarning: Series.__setitem__ treating keys as positions is deprecated.
In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior).
To set a value by position, use `ser.iloc[pos] = value`
```

**Fix:**
```python
obv_sign.iloc[0] = 0  # First value has no direction
```

---

### 2. fillna Downcasting Deprecation
**File:** `src/features/price_features.py:427`

```python
fast_above_prev = fast_above.shift(1).fillna(False).astype(bool)
```

**Warning:**
```
FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated
and will change in a future version. Call result.infer_objects(copy=False) instead.
```

**Fix:**
```python
fast_above_prev = fast_above.shift(1).fillna(False).infer_objects(copy=False).astype(bool)
```

---

## Medium Priority

### 3. Missing Module Exports
**File:** `src/features/__init__.py:11`

```python
__all__ = []  # Empty - PriceFeatureBuilder, SentimentFeatureBuilder not exported
```

**Impact:** Users cannot do `from src.features import PriceFeatureBuilder`

**Fix:**
```python
from .price_features import PriceFeatureBuilder
from .sentiment_features import SentimentFeatureBuilder

__all__ = [
    "PriceFeatureBuilder",
    "SentimentFeatureBuilder",
]
```

---

### 4. Datetime Parsing Warning in Tests
**File:** `tests/test_risk.py:559`

```python
prices.index = pd.to_datetime(df['Date'])
```

**Warning:**
```
FutureWarning: In a future version of pandas, parsing datetimes with mixed time zones
will raise an error unless `utc=True`.
```

**Fix:**
```python
prices.index = pd.to_datetime(df['Date'], utc=True)
```

---

## Low Priority / Code Quality

### 5. Deprecated numpy.random.seed() Pattern
**File:** `src/risk/var.py:187-188`

```python
if seed is not None:
    np.random.seed(seed)
```

**Recommendation:** Use `numpy.random.Generator` for better practice:
```python
rng = np.random.default_rng(seed)
simulated_returns = rng.normal(mean, std, simulations)
```

---

### 6. Broad Exception Handling
**Files:** Multiple locations

- `src/risk/risk_report.py:135-140` - GARCH forecast
- `src/risk/risk_report.py:143-147` - Volatility regime
- `src/sentiment/finbert.py:263-264` - Analyze method

**Current:**
```python
except Exception:
    # silent fallback
```

**Recommendation:** Catch specific exceptions for better debugging:
```python
except (ValueError, RuntimeError) as e:
    logger.warning(f"GARCH fitting failed: {e}")
```

---

## Incomplete Implementation (Placeholder Files)

The following files are empty stubs that need implementation:

### Features Module
- [ ] `src/features/risk_features.py` - VaR/volatility features for ML
- [ ] `src/features/builder.py` - Aggregate all feature builders

### Database Module
- [ ] `src/db/connection.py`
- [ ] `src/db/models.py`
- [ ] `src/db/queries.py`

### ML Module
- [ ] `src/ml/model.py`
- [ ] `src/ml/validation.py`
- [ ] `src/ml/predictions.py`

### Dashboard
- [ ] `dashboard/app.py`
- [ ] `dashboard/pages/sentiment.py`
- [ ] `dashboard/pages/risk.py`
- [ ] `dashboard/pages/signals.py`
- [ ] `dashboard/pages/backtest.py`
- [ ] `dashboard/components/charts.py`
- [ ] `dashboard/components/tables.py`

### Scripts
- [ ] `scripts/run_daily_pipeline.py`
- [ ] `scripts/backfill_historical.py`
- [ ] `scripts/init_db.py`

### Tests
- [ ] `tests/test_var.py` - Empty
- [ ] `tests/test_features.py` - Empty
- [ ] Tests for `src/sentiment/` module - Missing

---

## Test Status

All 46 existing tests pass:
```
tests/test_risk.py ... 46 passed, 5 warnings in 12.26s
```

---

## Notes

- The Kupiec likelihood ratio test formula in `var.py:415-420` is **correct** (verified mathematically)
- Drawdown validation properly catches `prices <= 0`
- Overall code quality is good with proper docstrings, type hints, and error handling
