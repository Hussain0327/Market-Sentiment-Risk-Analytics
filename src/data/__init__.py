"""
Data collection and processing module.

Handles:
- News data fetching from Finnhub API
- Price data fetching from yfinance
- Symbol watchlist management
"""

from src.data.news_client import FinnhubNewsClient
from src.data.price_client import PriceClient
from src.data.watchlist import Watchlist

__all__ = ["FinnhubNewsClient", "PriceClient", "Watchlist"]
