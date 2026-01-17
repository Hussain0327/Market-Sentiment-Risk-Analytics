"""
Configuration management for Market Sentiment & Risk Analytics project.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Main configuration class."""

    # Project paths
    PROJECT_ROOT = Path(__file__).parent.resolve()
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    MODELS_DIR = DATA_DIR / "models"

    # API Configuration
    FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
    FINNHUB_BASE_URL = "https://finnhub.io/api/v1"

    # Database Configuration
    DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR}/market_sentiment.db")

    # Model Settings
    SENTIMENT_MODEL = os.getenv("SENTIMENT_MODEL", "ProsusAI/finbert")
    USE_GPU = os.getenv("USE_GPU", "false").lower() == "true"
    DEVICE = "cuda" if USE_GPU else "cpu"

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # Risk Parameters
    VAR_CONFIDENCE_LEVELS = [0.95, 0.99]
    LOOKBACK_WINDOWS = [21, 63, 252]  # 1 month, 3 months, 1 year (trading days)
    GARCH_ORDER = (1, 1)

    # Feature Engineering Parameters
    SENTIMENT_LOOKBACK_DAYS = 7
    PRICE_LOOKBACK_DAYS = 30
    TECHNICAL_INDICATORS = ["sma", "ema", "rsi", "macd", "bollinger"]

    # Default Watchlist
    DEFAULT_WATCHLIST = [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "META",
        "NVDA",
        "TSLA",
        "JPM",
        "V",
        "SPY",
    ]

    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist."""
        cls.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def validate(cls):
        """Validate critical configuration."""
        warnings = []

        if not cls.FINNHUB_API_KEY:
            warnings.append("FINNHUB_API_KEY not set - news data collection will not work")

        return warnings


# Ensure directories exist on import
Config.ensure_directories()
