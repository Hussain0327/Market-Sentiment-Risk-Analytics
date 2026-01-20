# Market Sentiment & Risk Analytics Engine

A comprehensive financial analytics platform that combines sentiment analysis, risk metrics, and machine learning to generate actionable trading signals.

## Overview

```
┌─────────────────────┐
│   Risk Analytics    │
│  - VaR (3 methods)  │
│  - Volatility fcst  │
│  - Drawdown metrics │
└──────────┬──────────┘
           │
[Finnhub API] → [Sentiment Model] → [Feature Engineering] → [ML Signal Model] → [Dashboard]
           │
┌──────────┴──────────┐
│     Price Data      │
│  - Returns/Volume   │
│  - Technicals       │
└─────────────────────┘
```

## Features

| Component | Tools | Purpose |
|-----------|-------|---------|
| **Sentiment Analysis** | FinBERT, VADER | Extract market sentiment from financial news |
| **VaR Calculation** | numpy, scipy | Measure portfolio risk exposure |
| **Volatility Forecasting** | GARCH (arch) | Model time-varying risk |
| **Feature Engineering** | pandas | Prepare data for ML models |
| **ML Models** | XGBoost | Direction prediction & signal generation |
| **Backtesting** | vectorbt | Validate strategies out-of-sample |
| **Dashboard** | Streamlit, Plotly | Interactive visualization |

## Tech Stack

- **Data Sources:** Finnhub API (news), yfinance (prices)
- **Database:** SQLite with SQLAlchemy
- **Sentiment:** FinBERT (primary), VADER (fallback)
- **Risk:** numpy, scipy, arch (GARCH)
- **ML:** XGBoost, scikit-learn
- **Dashboard:** Streamlit, Plotly

## Project Structure

```
Market-Sentiment-Risk-Analytics/
├── config.py                 # Configuration management
├── requirements.txt          # Python dependencies
├── src/
│   ├── data/                 # Data collection (Finnhub, yfinance)
│   ├── sentiment/            # FinBERT & VADER analysis
│   ├── risk/                 # VaR, volatility, drawdown
│   ├── features/             # Feature engineering (100+ features)
│   ├── ml/                   # XGBoost models & signal generation
│   └── db/                   # SQLite database operations
├── dashboard/
│   ├── app.py                # Streamlit entry point
│   ├── components/           # Reusable UI components
│   └── pages/                # Dashboard pages
├── notebooks/                # Jupyter notebooks for EDA
├── scripts/                  # Pipeline scripts
├── tests/                    # Test suite (270 tests)
└── docs/                     # Documentation
```

## Quick Start

### 1. Clone and Setup Environment

```bash
git clone <repository-url>
cd Market-Sentiment-Risk-Analytics

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
cp .env.example .env
# Edit .env and add your Finnhub API key
```

Get a free API key at: https://finnhub.io/register

### 3. Verify Installation

```bash
python -c "from config import Config; print(Config.PROJECT_ROOT)"
python -c "from src import data, sentiment, risk, features, ml, db"
```

### 4. Run Sentiment Analysis

```python
from src.sentiment import VaderAnalyzer, SentimentAggregator
import pandas as pd

# Load news data
news = pd.read_csv('data/raw/AAPL_news.csv')

# Analyze sentiment
analyzer = VaderAnalyzer()
sentiment = analyzer.analyze_batch(news, text_column='headline')

# Aggregate to daily signals
agg = SentimentAggregator()
daily = agg.aggregate_daily(sentiment)
signals = agg.generate_signals(daily, method='zscore')

print(signals[['symbol', 'date', 'sentiment_score', 'signal']])
```

### 5. Train ML Model and Generate Signals

```python
from src.features import FeatureBuilder
from src.ml import DirectionClassifier, WalkForwardValidator, PredictionPipeline
import pandas as pd

# Build features
builder = FeatureBuilder()
prices = pd.read_csv('data/raw/AAPL_prices.csv')
features = builder.build_features(prices, symbol='AAPL', include_sentiment=False)
X, y = builder.create_ml_dataset(features, prices, target_horizon=1, target_type='direction')

# Walk-forward validation (no lookahead bias)
validator = WalkForwardValidator(
    model_class=DirectionClassifier,
    model_params={'n_estimators': 100, 'max_depth': 4},
    n_splits=5,
    test_size=25
)
results = validator.validate(X, y)
print(f"CV Accuracy: {results.aggregate_metrics['accuracy_mean']:.1%}")

# Train final model and generate signals
clf = DirectionClassifier(params={'n_estimators': 100, 'max_depth': 4})
clf.fit(X, y)

pipeline = PredictionPipeline(classifier=clf)
signals = pipeline.predict(X, symbol='AAPL')
print(f"Bullish: {len(signals.get_bullish())}, Bearish: {len(signals.get_bearish())}")

# Get latest signal
latest = pipeline.predict_latest(X, symbol='AAPL')
print(f"Latest: {'BULLISH' if latest.is_bullish else 'BEARISH'} ({latest.confidence:.1%} confidence)")
```

### 6. Run Dashboard (coming soon)

```bash
streamlit run dashboard/app.py
```

## Configuration

Key settings in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `FINNHUB_API_KEY` | (from .env) | Finnhub API key |
| `SENTIMENT_MODEL` | ProsusAI/finbert | HuggingFace model for sentiment |
| `VAR_CONFIDENCE_LEVELS` | [0.95, 0.99] | VaR confidence levels |
| `LOOKBACK_WINDOWS` | [21, 63, 252] | Trading days for analysis |
| `DEFAULT_WATCHLIST` | AAPL, MSFT, ... | Default stock symbols |

## Development Status

**Current Progress: 75% Complete (Phases 1-6 of 8)**

| Phase | Status | Description |
|-------|--------|-------------|
| 1. Foundation | Complete | Project structure, config, dependencies |
| 2. Data Collection | Complete | 1,731 news articles + 1yr prices for 7 symbols |
| 3. Sentiment Analysis | Complete | FinBERT + VADER analyzers, daily aggregation |
| 4. Risk Metrics | Complete | VaR (3 methods), GARCH volatility, drawdown analysis |
| 5. Feature Engineering | Complete | 100+ features from price, sentiment, and risk data |
| 6. ML Models | Complete | XGBoost classifier/regressor, walk-forward validation, signal generation |
| 7. Database | Next | SQLite persistence |
| 8. Dashboard | Planned | Streamlit visualization |

### Test Coverage

```
270 tests passing across 5 test modules:
  - test_data.py       (data collection)
  - test_sentiment.py  (39 tests - sentiment analysis)
  - test_risk.py       (46 tests - risk metrics)
  - test_features.py   (69 tests - feature engineering)
  - test_ml.py         (64 tests - ML models)
```

Run tests with: `pytest tests/ -v`

See [docs/README.md](docs/README.md) for detailed implementation progress.

## What's Next

### Phase 7: Database (Next)
- SQLAlchemy ORM models for news, prices, sentiment, signals
- SQLite connection management
- CRUD operations and aggregation queries

### Phase 8: Dashboard (Planned)
- Streamlit multi-page application
- Interactive Plotly visualizations
- Real-time signal monitoring
- Backtesting interface

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
