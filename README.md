# Market Sentiment & Risk Analytics

A financial analytics platform that combines NLP-based sentiment analysis, quantitative risk metrics, and machine learning to generate trading signals. Built with Python, deployed on Streamlit Cloud.

**Live Demo:** [View Dashboard](https://market-sentiment-risk-analytics.streamlit.app/)

## What It Does

The system processes financial news through FinBERT sentiment models, calculates risk metrics (VaR, volatility forecasting, drawdown analysis), engineers 100+ features, and trains XGBoost models to predict market direction. Everything feeds into an interactive dashboard.

```
News Data (Finnhub) --> Sentiment Analysis (FinBERT) --\
                                                        --> Feature Engineering --> ML Models --> Dashboard
Price Data (yfinance) --> Risk Metrics (VaR, GARCH) --/
```

## Tech Stack

| Layer | Technology |
|-------|------------|
| Data Sources | Finnhub API, yfinance |
| Sentiment | FinBERT (HuggingFace), VADER |
| Risk | NumPy, SciPy, arch (GARCH) |
| ML | XGBoost, scikit-learn |
| Backend | FastAPI, SQLAlchemy |
| Frontend | Streamlit, Plotly |
| Deployment | Streamlit Cloud, GitHub Releases |

## Features

**Sentiment Analysis**
- FinBERT transformer model for financial text
- VADER as lightweight fallback
- Daily aggregation with z-score signal generation

**Risk Metrics**
- Value at Risk (Historical, Parametric, Monte Carlo)
- GARCH(1,1) volatility forecasting
- Maximum drawdown and recovery analysis

**ML Pipeline**
- 100+ engineered features (technical, sentiment, risk)
- XGBoost direction classifier and return regressor
- Walk-forward validation to prevent lookahead bias

**Dashboard**
- Overview with portfolio summary
- Sentiment trends and news analysis
- Risk metrics visualization
- ML predictions with confidence scores
- Backtesting interface

## Project Structure

```
Market-Sentiment-Risk-Analytics/
├── api/                      # FastAPI backend
│   ├── main.py
│   └── routes/               # REST endpoints
├── dashboard/
│   ├── app.py                # Streamlit entry point
│   ├── views/                # Dashboard pages
│   ├── components/           # Reusable charts/tables
│   ├── data_loader.py        # Data access layer
│   └── remote_loader.py      # GitHub Releases model loader
├── src/
│   ├── data/                 # Data collection
│   ├── sentiment/            # FinBERT & VADER
│   ├── risk/                 # VaR, volatility, drawdown
│   ├── features/             # Feature engineering
│   ├── ml/                   # XGBoost models
│   └── db/                   # Database operations
├── scripts/
│   ├── train_models.py       # Model training pipeline
│   └── upload_models.py      # Deploy models to GitHub Releases
├── data/
│   ├── raw/                  # Price and news CSVs
│   ├── processed/            # Sentiment scores, signals
│   └── models/               # Trained model files
└── tests/                    # Test suite
```

## Setup

### Prerequisites

- Python 3.10+
- Finnhub API key (free at finnhub.io)

### Installation

```bash
git clone https://github.com/Hussain0327/Market-Sentiment-Risk-Analytics.git
cd Market-Sentiment-Risk-Analytics

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Add your FINNHUB_API_KEY to .env
```

### Run Locally

```bash
# Dashboard
streamlit run dashboard/app.py

# API server
uvicorn api.main:app --reload
```

### Train Models

```bash
python scripts/train_models.py
```

## Deployment

The dashboard is deployed on Streamlit Cloud. Models are stored in GitHub Releases and fetched at runtime, so retraining doesn't require redeployment.

**Update models without redeploying:**
```bash
python scripts/train_models.py --retrain
export GITHUB_TOKEN="your_token"
python scripts/upload_models.py
```

The dashboard automatically picks up new models on the next page load.

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/prices/{symbol}` | Historical price data |
| `GET /api/sentiment/{symbol}` | Sentiment scores and signals |
| `GET /api/risk/{symbol}` | VaR and volatility metrics |
| `GET /api/predictions/{symbol}` | ML predictions |

Full API docs at `/docs` when running locally.

## Configuration

Environment variables (`.env`):

```
FINNHUB_API_KEY=your_key
SENTIMENT_MODEL=ProsusAI/finbert
DATABASE_URL=sqlite:///data/market_sentiment.db
```

## Coverage

7 symbols tracked: AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA

## License

MIT
