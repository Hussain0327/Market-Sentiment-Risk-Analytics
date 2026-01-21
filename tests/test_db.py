"""
Tests for database module.

Tests:
- Table creation
- CRUD operations for each entity
- Upsert behavior
- Date range queries
- Aggregations
"""

import pytest
from datetime import date, datetime, timedelta

import pandas as pd
import numpy as np

from src.db import (
    DatabaseManager,
    Base,
    Symbol,
    Article,
    Price,
    ArticleSentiment,
    DailySentiment,
    Feature,
    Prediction,
    Signal,
    get_or_create_symbol,
    upsert_symbol,
    get_all_symbols,
    get_symbol_by_ticker,
    upsert_price,
    bulk_upsert_prices,
    get_prices_df,
    upsert_daily_sentiment,
    bulk_upsert_daily_sentiment,
    get_daily_sentiment_df,
    upsert_feature,
    get_features_df,
    insert_signal,
    get_latest_signals,
    get_signals_df,
    upsert_prediction,
    get_sentiment_summary,
    get_latest_sentiment_by_symbol,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def db():
    """Create an in-memory database for testing."""
    db_manager = DatabaseManager(":memory:")
    db_manager.init_db()
    return db_manager


@pytest.fixture
def session(db):
    """Create a database session."""
    with db.session() as session:
        yield session


@pytest.fixture
def sample_symbol(session):
    """Create a sample symbol."""
    symbol = get_or_create_symbol(session, "AAPL", name="Apple Inc.", sector="Technology")
    return symbol


@pytest.fixture
def sample_prices_df():
    """Create sample price DataFrame."""
    dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
    return pd.DataFrame({
        "Date": dates,
        "Open": np.random.uniform(100, 110, 10),
        "High": np.random.uniform(110, 120, 10),
        "Low": np.random.uniform(90, 100, 10),
        "Close": np.random.uniform(100, 110, 10),
        "Volume": np.random.randint(1000000, 10000000, 10),
        "Dividends": np.zeros(10),
        "Stock_Splits": np.zeros(10),
    })


@pytest.fixture
def sample_sentiment_df():
    """Create sample daily sentiment DataFrame."""
    dates = pd.date_range(start="2024-01-01", periods=5, freq="D")
    return pd.DataFrame({
        "date": dates,
        "sentiment_score": np.random.uniform(-1, 1, 5),
        "sentiment_confidence": np.random.uniform(0, 1, 5),
        "article_count": np.random.randint(5, 50, 5),
        "bullish_ratio": np.random.uniform(0, 1, 5),
        "bearish_ratio": np.random.uniform(0, 0.5, 5),
        "sentiment_std": np.random.uniform(0, 0.5, 5),
        "signal_valid": [True, False, True, True, False],
    })


# =============================================================================
# Table Creation Tests
# =============================================================================

class TestTableCreation:
    """Tests for database table creation."""

    def test_create_all_tables(self, db):
        """Test that all tables are created."""
        # Tables should already be created by fixture
        from sqlalchemy import inspect
        inspector = inspect(db.engine)
        tables = inspector.get_table_names()

        expected_tables = [
            'symbols', 'articles', 'prices', 'article_sentiments',
            'daily_sentiments', 'features', 'predictions', 'signals'
        ]

        for table in expected_tables:
            assert table in tables, f"Table {table} not found"

    def test_drop_and_recreate(self, db):
        """Test dropping and recreating tables."""
        from sqlalchemy import inspect

        db.drop_all()

        # Verify tables are dropped
        inspector = inspect(db.engine)
        tables = inspector.get_table_names()
        assert len(tables) == 0

        # Recreate tables
        db.init_db()
        inspector = inspect(db.engine)
        tables = inspector.get_table_names()
        assert len(tables) >= 8


# =============================================================================
# Symbol Tests
# =============================================================================

class TestSymbol:
    """Tests for Symbol CRUD operations."""

    def test_get_or_create_symbol_new(self, session):
        """Test creating a new symbol."""
        symbol = get_or_create_symbol(session, "MSFT", name="Microsoft", sector="Technology")

        assert symbol.ticker == "MSFT"
        assert symbol.name == "Microsoft"
        assert symbol.sector == "Technology"
        assert symbol.is_active is True
        assert symbol.id is not None

    def test_get_or_create_symbol_existing(self, session, sample_symbol):
        """Test getting an existing symbol."""
        symbol2 = get_or_create_symbol(session, "AAPL")

        assert symbol2.id == sample_symbol.id
        assert symbol2.ticker == "AAPL"

    def test_upsert_symbol_update(self, session, sample_symbol):
        """Test updating a symbol."""
        updated = upsert_symbol(
            session, "AAPL",
            name="Apple Inc. (Updated)",
            sector="Tech",
            is_active=False
        )

        assert updated.id == sample_symbol.id
        assert updated.name == "Apple Inc. (Updated)"
        assert updated.sector == "Tech"
        assert updated.is_active is False

    def test_get_all_symbols(self, session):
        """Test getting all symbols."""
        get_or_create_symbol(session, "AAPL")
        get_or_create_symbol(session, "MSFT")
        get_or_create_symbol(session, "GOOGL")

        symbols = get_all_symbols(session)
        tickers = [s.ticker for s in symbols]

        assert len(symbols) == 3
        assert "AAPL" in tickers
        assert "MSFT" in tickers
        assert "GOOGL" in tickers

    def test_get_all_symbols_active_only(self, session):
        """Test filtering active symbols."""
        get_or_create_symbol(session, "AAPL")
        symbol2 = get_or_create_symbol(session, "MSFT")
        symbol2.is_active = False

        active_symbols = get_all_symbols(session, active_only=True)
        all_symbols = get_all_symbols(session, active_only=False)

        assert len(active_symbols) == 1
        assert len(all_symbols) == 2

    def test_get_symbol_by_ticker(self, session, sample_symbol):
        """Test getting symbol by ticker."""
        symbol = get_symbol_by_ticker(session, "AAPL")
        assert symbol is not None
        assert symbol.ticker == "AAPL"

        not_found = get_symbol_by_ticker(session, "NONEXISTENT")
        assert not_found is None


# =============================================================================
# Price Tests
# =============================================================================

class TestPrice:
    """Tests for Price CRUD operations."""

    def test_upsert_price_new(self, session, sample_symbol):
        """Test creating a new price record."""
        price = upsert_price(
            session, sample_symbol.id, date(2024, 1, 15),
            open_price=100.0, high=105.0, low=99.0, close=103.0,
            volume=1000000
        )

        assert price.symbol_id == sample_symbol.id
        assert price.date == date(2024, 1, 15)
        assert price.open == 100.0
        assert price.close == 103.0

    def test_upsert_price_update(self, session, sample_symbol):
        """Test updating an existing price record."""
        # Create initial
        price1 = upsert_price(
            session, sample_symbol.id, date(2024, 1, 15),
            open_price=100.0, high=105.0, low=99.0, close=103.0,
            volume=1000000
        )

        # Update
        price2 = upsert_price(
            session, sample_symbol.id, date(2024, 1, 15),
            open_price=101.0, high=106.0, low=98.0, close=104.0,
            volume=1500000
        )

        assert price2.id == price1.id
        assert price2.close == 104.0
        assert price2.volume == 1500000

    def test_bulk_upsert_prices(self, session, sample_symbol, sample_prices_df):
        """Test bulk price insertion."""
        count = bulk_upsert_prices(session, sample_prices_df, sample_symbol.id)

        assert count == 10

        # Verify data
        prices_back = get_prices_df(session, sample_symbol.id)
        assert len(prices_back) == 10

    def test_get_prices_df(self, session, sample_symbol, sample_prices_df):
        """Test getting prices as DataFrame."""
        bulk_upsert_prices(session, sample_prices_df, sample_symbol.id)

        # Get all
        df = get_prices_df(session, sample_symbol.id)
        assert len(df) == 10
        assert 'date' in df.columns
        assert 'close' in df.columns

    def test_get_prices_df_date_range(self, session, sample_symbol, sample_prices_df):
        """Test getting prices with date filter."""
        bulk_upsert_prices(session, sample_prices_df, sample_symbol.id)

        # Filter by date range
        df = get_prices_df(
            session, sample_symbol.id,
            start_date=date(2024, 1, 3),
            end_date=date(2024, 1, 7)
        )

        assert len(df) == 5  # Days 3-7 inclusive

    def test_get_prices_df_empty(self, session, sample_symbol):
        """Test getting prices when none exist."""
        df = get_prices_df(session, sample_symbol.id)
        assert len(df) == 0
        assert 'date' in df.columns


# =============================================================================
# Daily Sentiment Tests
# =============================================================================

class TestDailySentiment:
    """Tests for DailySentiment CRUD operations."""

    def test_upsert_daily_sentiment_new(self, session, sample_symbol):
        """Test creating a new daily sentiment record."""
        sentiment = upsert_daily_sentiment(
            session, sample_symbol.id, date(2024, 1, 15),
            sentiment_score=0.5, sentiment_confidence=0.8,
            article_count=25, bullish_ratio=0.6, bearish_ratio=0.2,
            sentiment_std=0.3, signal_valid=True
        )

        assert sentiment.symbol_id == sample_symbol.id
        assert sentiment.sentiment_score == 0.5
        assert sentiment.article_count == 25
        assert sentiment.signal_valid is True

    def test_upsert_daily_sentiment_update(self, session, sample_symbol):
        """Test updating existing daily sentiment."""
        # Create initial
        sent1 = upsert_daily_sentiment(
            session, sample_symbol.id, date(2024, 1, 15),
            sentiment_score=0.5, sentiment_confidence=0.8
        )

        # Update
        sent2 = upsert_daily_sentiment(
            session, sample_symbol.id, date(2024, 1, 15),
            sentiment_score=0.7, sentiment_confidence=0.9,
            article_count=30
        )

        assert sent2.id == sent1.id
        assert sent2.sentiment_score == 0.7
        assert sent2.article_count == 30

    def test_bulk_upsert_daily_sentiment(self, session, sample_symbol, sample_sentiment_df):
        """Test bulk sentiment insertion."""
        count = bulk_upsert_daily_sentiment(session, sample_sentiment_df, sample_symbol.id)

        assert count == 5

        # Verify data
        df_back = get_daily_sentiment_df(session, sample_symbol.id)
        assert len(df_back) == 5

    def test_get_daily_sentiment_df(self, session, sample_symbol, sample_sentiment_df):
        """Test getting daily sentiment as DataFrame."""
        bulk_upsert_daily_sentiment(session, sample_sentiment_df, sample_symbol.id)

        df = get_daily_sentiment_df(session, sample_symbol.id)
        assert len(df) == 5
        assert 'sentiment_score' in df.columns
        assert 'bullish_ratio' in df.columns

    def test_get_daily_sentiment_df_date_range(self, session, sample_symbol, sample_sentiment_df):
        """Test getting sentiment with date filter."""
        bulk_upsert_daily_sentiment(session, sample_sentiment_df, sample_symbol.id)

        df = get_daily_sentiment_df(
            session, sample_symbol.id,
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 4)
        )

        assert len(df) == 3  # Days 2-4 inclusive


# =============================================================================
# Feature Tests
# =============================================================================

class TestFeature:
    """Tests for Feature CRUD operations."""

    def test_upsert_feature_new(self, session, sample_symbol):
        """Test creating a new feature record."""
        feature_data = {
            "sentiment_score": 0.5,
            "momentum_5d": 0.02,
            "volatility_21d": 0.15,
            "rsi_14": 55.0
        }

        feature = upsert_feature(
            session, sample_symbol.id, date(2024, 1, 15),
            feature_data=feature_data, feature_version="v1"
        )

        assert feature.symbol_id == sample_symbol.id
        assert feature.feature_data["sentiment_score"] == 0.5
        assert feature.feature_version == "v1"

    def test_upsert_feature_update(self, session, sample_symbol):
        """Test updating feature data."""
        feature_data_v1 = {"sentiment_score": 0.5}
        feature_data_v2 = {"sentiment_score": 0.7, "new_feature": 1.0}

        feat1 = upsert_feature(
            session, sample_symbol.id, date(2024, 1, 15),
            feature_data=feature_data_v1, feature_version="v1"
        )

        feat2 = upsert_feature(
            session, sample_symbol.id, date(2024, 1, 15),
            feature_data=feature_data_v2, feature_version="v2"
        )

        assert feat2.id == feat1.id
        assert feat2.feature_data["sentiment_score"] == 0.7
        assert feat2.feature_data["new_feature"] == 1.0
        assert feat2.feature_version == "v2"

    def test_get_features_df(self, session, sample_symbol):
        """Test getting features as DataFrame."""
        for i in range(5):
            upsert_feature(
                session, sample_symbol.id, date(2024, 1, i + 1),
                feature_data={"sentiment": 0.1 * i, "momentum": 0.02 * i}
            )

        df = get_features_df(session, sample_symbol.id)
        assert len(df) == 5
        assert 'sentiment' in df.columns
        assert 'momentum' in df.columns

    def test_get_features_df_empty(self, session, sample_symbol):
        """Test getting features when none exist."""
        df = get_features_df(session, sample_symbol.id)
        assert len(df) == 0


# =============================================================================
# Signal Tests
# =============================================================================

class TestSignal:
    """Tests for Signal CRUD operations."""

    def test_insert_signal(self, session, sample_symbol):
        """Test inserting a signal."""
        signal = insert_signal(
            session, sample_symbol.id,
            timestamp=datetime(2024, 1, 15, 10, 30),
            direction=1, strength=0.75, confidence=0.8,
            expected_return=0.02, signal_type="sentiment"
        )

        assert signal.symbol_id == sample_symbol.id
        assert signal.direction == 1
        assert signal.strength == 0.75
        assert signal.is_bullish is True
        assert signal.is_bearish is False

    def test_get_latest_signals(self, session, sample_symbol):
        """Test getting latest signals."""
        for i in range(5):
            insert_signal(
                session, sample_symbol.id,
                timestamp=datetime(2024, 1, 15, 10 + i, 0),
                direction=1 if i % 2 == 0 else -1,
                strength=0.5 + 0.1 * i,
                confidence=0.7
            )

        signals = get_latest_signals(session, sample_symbol.id, limit=3)
        assert len(signals) == 3

        # Should be in descending timestamp order
        assert signals[0].timestamp > signals[1].timestamp

    def test_get_signals_df(self, session, sample_symbol):
        """Test getting signals as DataFrame."""
        for i in range(5):
            insert_signal(
                session, sample_symbol.id,
                timestamp=datetime(2024, 1, i + 1, 10, 0),
                direction=1, strength=0.5, confidence=0.7
            )

        df = get_signals_df(session, sample_symbol.id)
        assert len(df) == 5
        assert 'direction' in df.columns
        assert 'strength' in df.columns

    def test_get_signals_df_date_range(self, session, sample_symbol):
        """Test getting signals with timestamp filter."""
        for i in range(5):
            insert_signal(
                session, sample_symbol.id,
                timestamp=datetime(2024, 1, i + 1, 10, 0),
                direction=1, strength=0.5, confidence=0.7
            )

        # Filter from Jan 2 at 00:00 to Jan 4 at 23:59
        df = get_signals_df(
            session, sample_symbol.id,
            start_date=datetime(2024, 1, 2),
            end_date=datetime(2024, 1, 4, 23, 59)
        )

        assert len(df) == 3  # Signals on 2nd, 3rd, 4th at 10:00


# =============================================================================
# Prediction Tests
# =============================================================================

class TestPrediction:
    """Tests for Prediction CRUD operations."""

    def test_upsert_prediction_new(self, session, sample_symbol):
        """Test creating a new prediction."""
        pred = upsert_prediction(
            session, sample_symbol.id, date(2024, 1, 15),
            direction=1, probability=0.72, expected_return=0.015,
            model_name="xgboost", features_used=100
        )

        assert pred.symbol_id == sample_symbol.id
        assert pred.direction == 1
        assert pred.probability == 0.72

    def test_upsert_prediction_update(self, session, sample_symbol):
        """Test updating a prediction."""
        pred1 = upsert_prediction(
            session, sample_symbol.id, date(2024, 1, 15),
            direction=1, probability=0.72, model_name="xgboost"
        )

        pred2 = upsert_prediction(
            session, sample_symbol.id, date(2024, 1, 15),
            direction=1, probability=0.75, model_name="xgboost"
        )

        assert pred2.id == pred1.id
        assert pred2.probability == 0.75

    def test_different_models_same_date(self, session, sample_symbol):
        """Test predictions from different models on same date."""
        pred1 = upsert_prediction(
            session, sample_symbol.id, date(2024, 1, 15),
            direction=1, probability=0.72, model_name="xgboost"
        )

        pred2 = upsert_prediction(
            session, sample_symbol.id, date(2024, 1, 15),
            direction=0, probability=0.45, model_name="logistic"
        )

        assert pred1.id != pred2.id  # Different records


# =============================================================================
# Aggregation Tests
# =============================================================================

class TestAggregations:
    """Tests for aggregation queries."""

    def test_get_sentiment_summary(self, session):
        """Test sentiment summary aggregation."""
        # Create symbols
        sym1 = get_or_create_symbol(session, "AAPL")
        sym2 = get_or_create_symbol(session, "MSFT")

        # Add sentiment data
        for i in range(5):
            upsert_daily_sentiment(
                session, sym1.id, date(2024, 1, i + 1),
                sentiment_score=0.1 * i, sentiment_confidence=0.7,
                article_count=10 + i, bullish_ratio=0.5, bearish_ratio=0.2
            )
            upsert_daily_sentiment(
                session, sym2.id, date(2024, 1, i + 1),
                sentiment_score=-0.1 * i, sentiment_confidence=0.6,
                article_count=5 + i, bullish_ratio=0.3, bearish_ratio=0.4
            )

        summary = get_sentiment_summary(session)

        assert len(summary) == 2
        assert 'ticker' in summary.columns
        assert 'avg_sentiment' in summary.columns
        assert 'total_articles' in summary.columns

    def test_get_sentiment_summary_date_filter(self, session, sample_symbol, sample_sentiment_df):
        """Test sentiment summary with date filter."""
        bulk_upsert_daily_sentiment(session, sample_sentiment_df, sample_symbol.id)

        summary = get_sentiment_summary(
            session,
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 4)
        )

        assert len(summary) == 1
        assert summary.iloc[0]['days_count'] == 3

    def test_get_latest_sentiment_by_symbol(self, session):
        """Test getting latest sentiment per symbol."""
        sym1 = get_or_create_symbol(session, "AAPL")
        sym2 = get_or_create_symbol(session, "MSFT")

        # Add multiple days of sentiment
        for i in range(3):
            upsert_daily_sentiment(
                session, sym1.id, date(2024, 1, i + 1),
                sentiment_score=0.1 * i, sentiment_confidence=0.7
            )
            upsert_daily_sentiment(
                session, sym2.id, date(2024, 1, i + 1),
                sentiment_score=-0.1 * i, sentiment_confidence=0.6
            )

        latest = get_latest_sentiment_by_symbol(session)

        assert len(latest) == 2

        aapl_row = latest[latest['ticker'] == 'AAPL'].iloc[0]
        assert aapl_row['date'] == date(2024, 1, 3)  # Latest date

        msft_row = latest[latest['ticker'] == 'MSFT'].iloc[0]
        assert msft_row['date'] == date(2024, 1, 3)


# =============================================================================
# Relationship Tests
# =============================================================================

class TestRelationships:
    """Tests for model relationships."""

    def test_symbol_prices_relationship(self, session, sample_symbol, sample_prices_df):
        """Test Symbol -> Prices relationship."""
        bulk_upsert_prices(session, sample_prices_df, sample_symbol.id)

        # Refresh symbol from session
        symbol = get_symbol_by_ticker(session, "AAPL")
        assert len(symbol.prices) == 10

    def test_symbol_daily_sentiments_relationship(self, session, sample_symbol, sample_sentiment_df):
        """Test Symbol -> DailySentiments relationship."""
        bulk_upsert_daily_sentiment(session, sample_sentiment_df, sample_symbol.id)

        symbol = get_symbol_by_ticker(session, "AAPL")
        assert len(symbol.daily_sentiments) == 5

    def test_cascade_delete(self, session):
        """Test cascade delete on symbol removal."""
        symbol = get_or_create_symbol(session, "TEST")
        symbol_id = symbol.id

        # Add related data
        upsert_price(
            session, symbol_id, date(2024, 1, 15),
            100, 105, 99, 103, 1000000
        )
        upsert_daily_sentiment(
            session, symbol_id, date(2024, 1, 15),
            0.5, 0.7
        )

        # Delete symbol
        session.delete(symbol)
        session.flush()

        # Verify related data is deleted
        prices = session.query(Price).filter(Price.symbol_id == symbol_id).all()
        sentiments = session.query(DailySentiment).filter(
            DailySentiment.symbol_id == symbol_id
        ).all()

        assert len(prices) == 0
        assert len(sentiments) == 0


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_string_date_handling(self, session, sample_symbol):
        """Test that string dates are properly converted."""
        # String date
        price = upsert_price(
            session, sample_symbol.id, "2024-01-15",
            100, 105, 99, 103, 1000000
        )
        assert price.date == date(2024, 1, 15)

        # Datetime date
        sentiment = upsert_daily_sentiment(
            session, sample_symbol.id, datetime(2024, 1, 16, 10, 30),
            0.5, 0.7
        )
        assert sentiment.date == date(2024, 1, 16)

    def test_empty_dataframe_queries(self, session, sample_symbol):
        """Test queries return proper empty DataFrames."""
        prices = get_prices_df(session, sample_symbol.id)
        assert len(prices) == 0
        assert 'date' in prices.columns

        sentiments = get_daily_sentiment_df(session, sample_symbol.id)
        assert len(sentiments) == 0
        assert 'sentiment_score' in sentiments.columns

        signals = get_signals_df(session, sample_symbol.id)
        assert len(signals) == 0
        assert 'direction' in signals.columns

    def test_null_optional_fields(self, session, sample_symbol):
        """Test that optional fields can be null."""
        signal = insert_signal(
            session, sample_symbol.id,
            timestamp=datetime(2024, 1, 15),
            direction=0, strength=0.0, confidence=0.5,
            expected_return=None  # Optional
        )

        assert signal.expected_return is None

    def test_json_feature_data(self, session, sample_symbol):
        """Test JSON feature data with various types."""
        feature_data = {
            "string_val": "test",
            "int_val": 42,
            "float_val": 3.14,
            "list_val": [1, 2, 3],
            "nested": {"a": 1, "b": 2}
        }

        feature = upsert_feature(
            session, sample_symbol.id, date(2024, 1, 15),
            feature_data=feature_data
        )

        # Refresh from database
        session.expire(feature)
        assert feature.feature_data["string_val"] == "test"
        assert feature.feature_data["nested"]["a"] == 1
