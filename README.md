<p align="center">
  <h1 align="center">Market Sentiment & Risk Analytics</h1>
  <p align="center">
    NLP-powered sentiment analysis meets quantitative risk metrics and machine learning for smarter trading signals.
  </p>
</p>

<p align="center">
  <a href="https://web-iota-three-51.vercel.app">
    <img src="https://img.shields.io/badge/demo-live-brightgreen?style=for-the-badge" alt="Live Demo">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-blue?style=for-the-badge" alt="License">
  </a>
  <img src="https://img.shields.io/badge/python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Next.js-16-black?style=for-the-badge&logo=next.js" alt="Next.js">
</p>

<p align="center">
  <a href="#live-demos">View Demos</a> •
  <a href="#features">Features</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#api-reference">API</a>
</p>

---

## Live Demos

| Platform | URL | Description |
|----------|-----|-------------|
| **Next.js Dashboard** | [web-iota-three-51.vercel.app](https://web-iota-three-51.vercel.app) | Fast, modern dashboard with instant loads |
| **Streamlit App** | [streamlit.app](https://hussain0327-market-sentiment-risk-analytics.streamlit.app) | Interactive Python dashboard |

---

## Overview

This platform processes financial news through transformer-based sentiment models, calculates institutional-grade risk metrics, and trains ML models to predict market direction—all feeding into interactive dashboards.

**Tracked Symbols:** AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA PIPELINE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐       │
│   │   Finnhub    │────▶│     FinBERT      │────▶│    Sentiment     │       │
│   │  News API    │     │  Transformer     │     │    Signals       │       │
│   └──────────────┘     └──────────────────┘     └────────┬─────────┘       │
│                                                          │                  │
│   ┌──────────────┐     ┌──────────────────┐              │                  │
│   │   yfinance   │────▶│   Risk Engine    │              │                  │
│   │  Price Data  │     │  VaR • GARCH     │              │                  │
│   └──────────────┘     └────────┬─────────┘              │                  │
│                                 │                        │                  │
│                                 ▼                        ▼                  │
│                        ┌──────────────────────────────────┐                 │
│                        │      Feature Engineering         │                 │
│                        │   80+ technical + risk + NLP     │                 │
│                        └────────────────┬─────────────────┘                 │
│                                         │                                   │
│                                         ▼                                   │
│                        ┌──────────────────────────────────┐                 │
│                        │         XGBoost Models           │                 │
│                        │   Direction + Return Prediction  │                 │
│                        └────────────────┬─────────────────┘                 │
│                                         │                                   │
│                                         ▼                                   │
│                 ┌───────────────────────┴───────────────────────┐           │
│                 │                                               │           │
│          ┌──────▼──────┐                               ┌────────▼────────┐  │
│          │   Next.js   │                               │    Streamlit    │  │
│          │  Dashboard  │                               │    Dashboard    │  │
│          │   (Vercel)  │                               │     (Cloud)     │  │
│          └─────────────┘                               └─────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Features

### Sentiment Analysis

- **FinBERT Transformer** — Fine-tuned BERT model specifically for financial text
- **VADER Fallback** — Lightweight sentiment scoring for high-throughput scenarios
- **Signal Generation** — Z-score normalized signals with confidence thresholds
- **Daily Aggregation** — Time-weighted sentiment combining multiple news sources

### Risk Analytics

| Metric | Description |
|--------|-------------|
| **Value at Risk** | Historical, Parametric, and Monte Carlo VaR at 95%/99% confidence |
| **Expected Shortfall** | CVaR measuring tail risk beyond VaR threshold |
| **GARCH Volatility** | GARCH(1,1) forecasting for volatility clustering |
| **Drawdown Analysis** | Maximum drawdown, recovery periods, underwater curves |
| **Risk Scoring** | Composite 0-100 score combining multiple risk factors |

### Machine Learning Pipeline

- **80 Engineered Features** — Technical indicators, risk metrics, sentiment momentum
- **XGBoost Models** — Direction classifier + return regressor per symbol
- **Walk-Forward Validation** — Time-series aware backtesting to prevent lookahead bias
- **Model Persistence** — Trained models stored in GitHub Releases for cloud deployment

### Dashboards

**Next.js (Vercel)**
- Instant page loads with pre-rendered static data
- Modern React/TypeScript stack
- Interactive Recharts visualizations
- Symbol comparison across all metrics

**Streamlit (Python)**
- Real-time data updates
- ML model backtesting interface
- Detailed risk metric breakdowns
- News sentiment drill-down

---

## Tech Stack

| Layer | Technologies |
|-------|--------------|
| **Data** | Finnhub API, yfinance, pandas |
| **Sentiment** | HuggingFace Transformers, FinBERT, VADER |
| **Risk** | NumPy, SciPy, arch (GARCH) |
| **ML** | XGBoost, scikit-learn, walk-forward validation |
| **Backend** | FastAPI, SQLAlchemy, Pydantic |
| **Frontend** | Next.js 16, Streamlit, Recharts, Plotly, Tailwind CSS |
| **Deployment** | Vercel, Streamlit Cloud, GitHub Releases |

---

## Project Structure

```
Market-Sentiment-Risk-Analytics/
│
├── api/                          # FastAPI REST backend
│   ├── main.py                   # Application entry point
│   └── routes/                   # Endpoint handlers
│
├── dashboard/                    # Streamlit application
│   ├── app.py                    # Main entry point
│   ├── views/                    # Page components
│   ├── components/               # Reusable UI elements
│   ├── data_loader.py            # Unified data access
│   └── remote_loader.py          # GitHub Releases integration
│
├── web/                          # Next.js dashboard
│   ├── src/app/                  # App router pages
│   ├── src/components/           # React components
│   ├── src/lib/                  # Data utilities
│   ├── public/data/              # Static JSON data
│   └── scripts/export-data.py    # CSV to JSON export
│
├── src/                          # Core Python modules
│   ├── data/                     # Data collection clients
│   │   ├── watchlist.py          # Symbol management
│   │   ├── news_client.py        # Finnhub integration
│   │   └── price_client.py       # yfinance wrapper
│   │
│   ├── sentiment/                # NLP pipeline
│   │   ├── finbert.py            # Transformer model
│   │   ├── vader_fallback.py     # Lightweight fallback
│   │   └── aggregator.py         # Daily aggregation
│   │
│   ├── risk/                     # Risk calculations
│   │   ├── var.py                # Value at Risk
│   │   ├── volatility.py         # GARCH forecasting
│   │   └── drawdown.py           # Drawdown analysis
│   │
│   ├── features/                 # Feature engineering
│   │   ├── builder.py            # Pipeline orchestration
│   │   ├── price_features.py     # Technical indicators
│   │   ├── risk_features.py      # Risk-based features
│   │   └── sentiment_features.py # NLP features
│   │
│   ├── ml/                       # Machine learning
│   │   ├── model.py              # XGBoost training
│   │   ├── predictions.py        # Inference
│   │   └── validation.py         # Walk-forward CV
│   │
│   └── db/                       # Database layer
│       ├── models.py             # SQLAlchemy ORM
│       └── queries.py            # Prepared queries
│
├── scripts/                      # Automation
│   ├── train_models.py           # Full training pipeline
│   └── upload_models.py          # Deploy to GitHub Releases
│
├── data/                         # Data storage (gitignored)
│   ├── raw/                      # Price and news CSVs
│   ├── processed/                # Sentiment scores
│   └── models/                   # Trained model files
│
└── tests/                        # Test suite
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+ (for Next.js dashboard)
- Finnhub API key ([free tier available](https://finnhub.io))

### Installation

```bash
# Clone the repository
git clone https://github.com/Hussain0327/Market-Sentiment-Risk-Analytics.git
cd Market-Sentiment-Risk-Analytics

# Set up Python environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your FINNHUB_API_KEY
```

### Run Locally

```bash
# Streamlit Dashboard
streamlit run dashboard/app.py

# Next.js Dashboard
cd web && npm install && npm run dev

# FastAPI Backend
uvicorn api.main:app --reload
```

### Train Models

```bash
# Train XGBoost models for all symbols
python scripts/train_models.py

# Upload to GitHub Releases (requires GITHUB_TOKEN)
python scripts/upload_models.py
```

---

## API Reference

The FastAPI backend exposes RESTful endpoints for all data and predictions.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/prices/{symbol}` | GET | Historical OHLCV price data |
| `/api/sentiment/{symbol}` | GET | Sentiment scores and signals |
| `/api/risk/{symbol}` | GET | VaR, volatility, drawdown metrics |
| `/api/predictions/{symbol}` | GET | ML model predictions |
| `/api/signals/{symbol}` | GET | Trading signal history |
| `/health` | GET | Service health check |

Interactive API documentation available at `/docs` when running locally.

---

## Configuration

### Environment Variables

```env
# Required
FINNHUB_API_KEY=your_api_key_here

# Optional
SENTIMENT_MODEL=ProsusAI/finbert
DATABASE_URL=sqlite:///data/market_sentiment.db
GITHUB_TOKEN=your_token_for_model_uploads
```

### Customization

**Add New Symbols**
```python
# src/data/watchlist.py
SYMBOLS = ["AAPL", "MSFT", "GOOGL", ...]  # Add your symbols
```

**Adjust Risk Parameters**
```python
# src/risk/var.py
confidence_levels = [0.95, 0.99]  # VaR confidence levels
lookback_window = 252             # Trading days for historical VaR
```

---

## Deployment

### Next.js Dashboard (Vercel)

The Next.js dashboard deploys automatically with bundled JSON data—no cold starts.

```bash
cd web
vercel --prod
```

### Streamlit Dashboard

Deployed on Streamlit Cloud with models fetched from GitHub Releases at runtime.

### Model Updates

Update ML models without redeploying the dashboard:

```bash
# Retrain models
python scripts/train_models.py --retrain

# Upload to GitHub Releases
export GITHUB_TOKEN="your_token"
python scripts/upload_models.py
```

Models are automatically picked up on the next dashboard page load.

---

## Model Performance

Current model metrics (walk-forward validated):

| Symbol | AUC | Accuracy | Direction Acc | Features |
|--------|-----|----------|---------------|----------|
| AAPL | 55.2% | 48.6% | 48.6% | 80 |
| MSFT | 54.8% | 47.1% | 50.0% | 80 |
| GOOGL | 53.1% | 45.7% | 48.6% | 80 |
| AMZN | 52.9% | 48.6% | 51.4% | 80 |
| META | 51.7% | 45.7% | 47.1% | 80 |
| NVDA | 56.3% | 51.4% | 52.9% | 80 |
| TSLA | 54.2% | 48.6% | 50.0% | 80 |

*Note: Stock returns are inherently difficult to predict. These metrics reflect the challenge of forecasting in efficient markets.*

---

## Roadmap

- [ ] Real-time WebSocket price feeds
- [ ] Portfolio optimization module
- [ ] Options-implied sentiment integration
- [ ] Multi-timeframe analysis
- [ ] Alerting system for signal changes

---

## Contributing

Contributions are welcome. Please open an issue first to discuss what you'd like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [FinBERT](https://huggingface.co/ProsusAI/finbert) by ProsusAI for financial sentiment analysis
- [Finnhub](https://finnhub.io) for market news data
- [yfinance](https://github.com/ranaroussi/yfinance) for price data

---

<p align="center">
  Built by <a href="https://github.com/Hussain0327">Raja Hussain</a>
</p>
