"""
SQLAlchemy ORM models for Market Sentiment & Risk Analytics.

Provides 8 core models:
- Symbol: Reference table for tracked assets
- Article: News articles from Finnhub
- Price: OHLCV price data
- ArticleSentiment: Per-article sentiment scores
- DailySentiment: Aggregated daily sentiment
- Feature: JSON blob for ML features
- Prediction: ML model outputs
- Signal: Trading signals
"""

from datetime import datetime, date
from typing import Optional, Dict, Any

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Date, Boolean,
    ForeignKey, Text, JSON, UniqueConstraint, Index
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Symbol(Base):
    """
    Reference table for tracked assets.

    Attributes:
        id: Primary key
        ticker: Stock symbol (e.g., 'AAPL')
        name: Company name
        sector: Industry sector
        is_active: Whether symbol is actively tracked
        created_at: When the symbol was added
        updated_at: Last update timestamp
    """
    __tablename__ = 'symbols'

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(20), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=True)
    sector = Column(String(100), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    articles = relationship("Article", back_populates="symbol_rel", cascade="all, delete-orphan")
    prices = relationship("Price", back_populates="symbol_rel", cascade="all, delete-orphan")
    article_sentiments = relationship("ArticleSentiment", back_populates="symbol_rel", cascade="all, delete-orphan")
    daily_sentiments = relationship("DailySentiment", back_populates="symbol_rel", cascade="all, delete-orphan")
    features = relationship("Feature", back_populates="symbol_rel", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="symbol_rel", cascade="all, delete-orphan")
    signals = relationship("Signal", back_populates="symbol_rel", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Symbol(ticker='{self.ticker}', name='{self.name}')>"


class Article(Base):
    """
    News articles from Finnhub.

    Attributes:
        id: Primary key
        symbol_id: Foreign key to symbols
        finnhub_id: Unique article ID from Finnhub
        headline: Article headline
        summary: Article summary/description
        source: News source
        category: Article category
        url: Article URL
        image_url: Image URL if available
        article_datetime: When the article was published
        created_at: When the record was created
    """
    __tablename__ = 'articles'

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol_id = Column(Integer, ForeignKey('symbols.id', ondelete='CASCADE'), nullable=False)
    finnhub_id = Column(String(50), nullable=False, index=True)
    headline = Column(Text, nullable=False)
    summary = Column(Text, nullable=True)
    source = Column(String(100), nullable=True)
    category = Column(String(50), nullable=True)
    url = Column(Text, nullable=True)
    image_url = Column(Text, nullable=True)
    article_datetime = Column(DateTime, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Composite unique constraint
    __table_args__ = (
        UniqueConstraint('symbol_id', 'finnhub_id', name='uq_article_symbol_finnhub'),
        Index('ix_article_symbol_datetime', 'symbol_id', 'article_datetime'),
    )

    # Relationships
    symbol_rel = relationship("Symbol", back_populates="articles")
    sentiment = relationship("ArticleSentiment", back_populates="article", uselist=False, cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Article(id={self.finnhub_id}, headline='{self.headline[:50]}...')>"


class Price(Base):
    """
    OHLCV price data.

    Attributes:
        id: Primary key
        symbol_id: Foreign key to symbols
        date: Trading date
        open: Opening price
        high: High price
        low: Low price
        close: Closing price
        volume: Trading volume
        dividends: Dividend amount
        stock_splits: Stock split ratio
        created_at: When the record was created
    """
    __tablename__ = 'prices'

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol_id = Column(Integer, ForeignKey('symbols.id', ondelete='CASCADE'), nullable=False)
    date = Column(Date, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    dividends = Column(Float, default=0.0)
    stock_splits = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Composite unique constraint
    __table_args__ = (
        UniqueConstraint('symbol_id', 'date', name='uq_price_symbol_date'),
        Index('ix_price_symbol_date', 'symbol_id', 'date'),
    )

    # Relationships
    symbol_rel = relationship("Symbol", back_populates="prices")

    def __repr__(self) -> str:
        return f"<Price(symbol_id={self.symbol_id}, date={self.date}, close={self.close})>"


class ArticleSentiment(Base):
    """
    Per-article sentiment scores.

    Attributes:
        id: Primary key
        article_id: Foreign key to articles
        symbol_id: Foreign key to symbols (denormalized for queries)
        sentiment_score: Overall sentiment score [-1, 1]
        confidence: Model confidence [0, 1]
        prob_positive: Probability of positive sentiment
        prob_negative: Probability of negative sentiment
        prob_neutral: Probability of neutral sentiment
        sentiment_signal: Categorical signal (bullish/bearish/neutral)
        model_name: Name of the model used
        created_at: When the record was created
    """
    __tablename__ = 'article_sentiments'

    id = Column(Integer, primary_key=True, autoincrement=True)
    article_id = Column(Integer, ForeignKey('articles.id', ondelete='CASCADE'), nullable=False, unique=True)
    symbol_id = Column(Integer, ForeignKey('symbols.id', ondelete='CASCADE'), nullable=False)
    sentiment_score = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    prob_positive = Column(Float, default=0.0)
    prob_negative = Column(Float, default=0.0)
    prob_neutral = Column(Float, default=0.0)
    sentiment_signal = Column(String(20), nullable=True)
    model_name = Column(String(100), default='finbert')
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index('ix_article_sentiment_symbol', 'symbol_id'),
    )

    # Relationships
    article = relationship("Article", back_populates="sentiment")
    symbol_rel = relationship("Symbol", back_populates="article_sentiments")

    def __repr__(self) -> str:
        return f"<ArticleSentiment(article_id={self.article_id}, score={self.sentiment_score:.2f})>"


class DailySentiment(Base):
    """
    Aggregated daily sentiment.

    Attributes:
        id: Primary key
        symbol_id: Foreign key to symbols
        date: Date of aggregation
        sentiment_score: Time-weighted average sentiment
        sentiment_confidence: Average confidence
        article_count: Number of articles (buzz metric)
        bullish_ratio: Fraction of bullish articles
        bearish_ratio: Fraction of bearish articles
        sentiment_std: Standard deviation (disagreement metric)
        signal_valid: Whether signal meets quality thresholds
        created_at: When the record was created
        updated_at: Last update timestamp
    """
    __tablename__ = 'daily_sentiments'

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol_id = Column(Integer, ForeignKey('symbols.id', ondelete='CASCADE'), nullable=False)
    date = Column(Date, nullable=False, index=True)
    sentiment_score = Column(Float, nullable=False)
    sentiment_confidence = Column(Float, nullable=False)
    article_count = Column(Integer, default=0)
    bullish_ratio = Column(Float, default=0.0)
    bearish_ratio = Column(Float, default=0.0)
    sentiment_std = Column(Float, default=0.0)
    signal_valid = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Composite unique constraint
    __table_args__ = (
        UniqueConstraint('symbol_id', 'date', name='uq_daily_sentiment_symbol_date'),
        Index('ix_daily_sentiment_symbol_date', 'symbol_id', 'date'),
    )

    # Relationships
    symbol_rel = relationship("Symbol", back_populates="daily_sentiments")

    def __repr__(self) -> str:
        return f"<DailySentiment(symbol_id={self.symbol_id}, date={self.date}, score={self.sentiment_score:.2f})>"


class Feature(Base):
    """
    JSON blob for ML features.

    Uses flexible JSON storage for feature data to avoid migrations
    when adding new features.

    Attributes:
        id: Primary key
        symbol_id: Foreign key to symbols
        date: Feature date
        feature_data: JSON blob with all features
        feature_version: Version identifier for feature schema
        created_at: When the record was created
        updated_at: Last update timestamp
    """
    __tablename__ = 'features'

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol_id = Column(Integer, ForeignKey('symbols.id', ondelete='CASCADE'), nullable=False)
    date = Column(Date, nullable=False, index=True)
    feature_data = Column(JSON, nullable=False)
    feature_version = Column(String(50), default='v1')
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Composite unique constraint
    __table_args__ = (
        UniqueConstraint('symbol_id', 'date', name='uq_feature_symbol_date'),
        Index('ix_feature_symbol_date', 'symbol_id', 'date'),
    )

    # Relationships
    symbol_rel = relationship("Symbol", back_populates="features")

    def __repr__(self) -> str:
        return f"<Feature(symbol_id={self.symbol_id}, date={self.date})>"


class Prediction(Base):
    """
    ML model outputs.

    Attributes:
        id: Primary key
        symbol_id: Foreign key to symbols
        date: Prediction date
        direction: Predicted direction (1=up, 0=down)
        probability: Probability of up movement
        expected_return: Expected return from regressor
        model_name: Name of the model used
        model_version: Version of the model
        features_used: Number of features used
        created_at: When the record was created
    """
    __tablename__ = 'predictions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol_id = Column(Integer, ForeignKey('symbols.id', ondelete='CASCADE'), nullable=False)
    date = Column(Date, nullable=False, index=True)
    direction = Column(Integer, nullable=False)
    probability = Column(Float, nullable=False)
    expected_return = Column(Float, nullable=True)
    model_name = Column(String(100), default='xgboost')
    model_version = Column(String(50), nullable=True)
    features_used = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Composite unique constraint
    __table_args__ = (
        UniqueConstraint('symbol_id', 'date', 'model_name', name='uq_prediction_symbol_date_model'),
        Index('ix_prediction_symbol_date', 'symbol_id', 'date'),
    )

    # Relationships
    symbol_rel = relationship("Symbol", back_populates="predictions")

    def __repr__(self) -> str:
        return f"<Prediction(symbol_id={self.symbol_id}, date={self.date}, direction={self.direction})>"


class Signal(Base):
    """
    Trading signals.

    Attributes:
        id: Primary key
        symbol_id: Foreign key to symbols
        timestamp: Signal generation timestamp
        direction: Signal direction (1=long, -1=short, 0=neutral)
        strength: Signal strength [0, 1]
        confidence: Model confidence [0, 1]
        expected_return: Expected return
        signal_type: Type of signal (sentiment/ml/combined)
        created_at: When the record was created
    """
    __tablename__ = 'signals'

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol_id = Column(Integer, ForeignKey('symbols.id', ondelete='CASCADE'), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    direction = Column(Integer, nullable=False)
    strength = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    expected_return = Column(Float, nullable=True)
    signal_type = Column(String(50), default='combined')
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index('ix_signal_symbol_timestamp', 'symbol_id', 'timestamp'),
    )

    # Relationships
    symbol_rel = relationship("Symbol", back_populates="signals")

    @property
    def is_bullish(self) -> bool:
        """Check if signal is bullish."""
        return self.direction > 0

    @property
    def is_bearish(self) -> bool:
        """Check if signal is bearish."""
        return self.direction < 0

    @property
    def is_neutral(self) -> bool:
        """Check if signal is neutral."""
        return self.direction == 0

    def __repr__(self) -> str:
        return f"<Signal(symbol_id={self.symbol_id}, direction={self.direction}, strength={self.strength:.2f})>"
