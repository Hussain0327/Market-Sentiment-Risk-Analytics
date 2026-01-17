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
| **Signal Classification** | XGBoost, scikit-learn | Build predictive trading signals |
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
│   ├── features/             # Feature engineering
│   ├── ml/                   # XGBoost models
│   └── db/                   # SQLite database operations
├── dashboard/
│   ├── app.py                # Streamlit entry point
│   ├── components/           # Reusable UI components
│   └── pages/                # Dashboard pages
├── notebooks/                # Jupyter notebooks for EDA
├── scripts/                  # Pipeline scripts
├── tests/                    # Test suite
└── docs/                     # Documentation
```

## Quick Start

### 1. Clone and Setup Environment

```bash
git clone <repository-url>
cd Market-Sentiment-Risk-Analytics

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
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

### 4. Run Dashboard (coming soon)

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

See [docs/README.md](docs/README.md) for current implementation progress.

## License

MIT
