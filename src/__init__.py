"""
Market Sentiment & Risk Analytics - Source Package.

This package contains the core modules for:
- data: Data collection and processing
- sentiment: Sentiment analysis (FinBERT, VADER)
- risk: Risk metrics and calculations
- features: Feature engineering
- ml: Machine learning models
- db: Database operations
"""

from src import data, sentiment, risk, features, ml, db

__all__ = ["data", "sentiment", "risk", "features", "ml", "db"]
