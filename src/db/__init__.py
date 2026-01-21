"""
Database operations module.

Provides:
- SQLite database management
- Data models and schemas
- CRUD operations for sentiment and price data
"""

from .models import (
    Base,
    Symbol,
    Article,
    Price,
    ArticleSentiment,
    DailySentiment,
    Feature,
    Prediction,
    Signal,
)

from .connection import (
    DatabaseManager,
    init_db,
    get_db,
)

from .queries import (
    # Symbol operations
    get_or_create_symbol,
    upsert_symbol,
    get_all_symbols,
    get_symbol_by_ticker,
    # Price operations
    upsert_price,
    bulk_upsert_prices,
    get_prices_df,
    # Daily sentiment operations
    upsert_daily_sentiment,
    bulk_upsert_daily_sentiment,
    get_daily_sentiment_df,
    # Feature operations
    upsert_feature,
    get_features_df,
    # Signal operations
    insert_signal,
    get_latest_signals,
    get_signals_df,
    # Prediction operations
    upsert_prediction,
    # Aggregation queries
    get_sentiment_summary,
    get_latest_sentiment_by_symbol,
)

__all__ = [
    # Models
    'Base',
    'Symbol',
    'Article',
    'Price',
    'ArticleSentiment',
    'DailySentiment',
    'Feature',
    'Prediction',
    'Signal',
    # Connection
    'DatabaseManager',
    'init_db',
    'get_db',
    # Symbol operations
    'get_or_create_symbol',
    'upsert_symbol',
    'get_all_symbols',
    'get_symbol_by_ticker',
    # Price operations
    'upsert_price',
    'bulk_upsert_prices',
    'get_prices_df',
    # Daily sentiment operations
    'upsert_daily_sentiment',
    'bulk_upsert_daily_sentiment',
    'get_daily_sentiment_df',
    # Feature operations
    'upsert_feature',
    'get_features_df',
    # Signal operations
    'insert_signal',
    'get_latest_signals',
    'get_signals_df',
    # Prediction operations
    'upsert_prediction',
    # Aggregation queries
    'get_sentiment_summary',
    'get_latest_sentiment_by_symbol',
]
