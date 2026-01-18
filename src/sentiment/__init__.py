"""
Sentiment analysis module.

Provides:
- FinBERT-based financial sentiment analysis (primary)
- VADER sentiment analysis (fallback)
- Sentiment aggregation and signal generation
"""

from .finbert import FinBertAnalyzer, SentimentResult
from .vader_fallback import VaderAnalyzer, get_analyzer
from .aggregator import SentimentAggregator

__all__ = [
    "FinBertAnalyzer",
    "SentimentResult",
    "VaderAnalyzer",
    "SentimentAggregator",
    "get_analyzer",
]
