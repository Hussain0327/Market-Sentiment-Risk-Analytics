"""
Database CRUD operations for Market Sentiment & Risk Analytics.

Provides query functions for all database entities with upsert pattern
for idempotent data loading.
"""

from datetime import date, datetime
from typing import Optional, List, Dict, Any, Union

import pandas as pd
from sqlalchemy import select, func, and_
from sqlalchemy.orm import Session
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from .models import (
    Symbol, Article, Price, ArticleSentiment,
    DailySentiment, Feature, Prediction, Signal
)


# =============================================================================
# Symbol Operations
# =============================================================================

def get_or_create_symbol(
    session: Session,
    ticker: str,
    name: Optional[str] = None,
    sector: Optional[str] = None
) -> Symbol:
    """
    Get existing symbol or create new one.

    Args:
        session: Database session.
        ticker: Stock symbol.
        name: Company name (optional).
        sector: Industry sector (optional).

    Returns:
        Symbol object (existing or newly created).
    """
    symbol = session.query(Symbol).filter(Symbol.ticker == ticker).first()

    if symbol is None:
        symbol = Symbol(ticker=ticker, name=name, sector=sector)
        session.add(symbol)
        session.flush()

    return symbol


def upsert_symbol(
    session: Session,
    ticker: str,
    name: Optional[str] = None,
    sector: Optional[str] = None,
    is_active: bool = True
) -> Symbol:
    """
    Insert or update a symbol.

    Args:
        session: Database session.
        ticker: Stock symbol.
        name: Company name.
        sector: Industry sector.
        is_active: Whether symbol is actively tracked.

    Returns:
        Symbol object.
    """
    symbol = session.query(Symbol).filter(Symbol.ticker == ticker).first()

    if symbol is None:
        symbol = Symbol(ticker=ticker, name=name, sector=sector, is_active=is_active)
        session.add(symbol)
    else:
        if name is not None:
            symbol.name = name
        if sector is not None:
            symbol.sector = sector
        symbol.is_active = is_active
        symbol.updated_at = datetime.utcnow()

    session.flush()
    return symbol


def get_all_symbols(session: Session, active_only: bool = True) -> List[Symbol]:
    """Get all symbols."""
    query = session.query(Symbol)
    if active_only:
        query = query.filter(Symbol.is_active == True)
    return query.order_by(Symbol.ticker).all()


def get_symbol_by_ticker(session: Session, ticker: str) -> Optional[Symbol]:
    """Get symbol by ticker."""
    return session.query(Symbol).filter(Symbol.ticker == ticker).first()


# =============================================================================
# Price Operations
# =============================================================================

def upsert_price(
    session: Session,
    symbol_id: int,
    date_val: Union[date, datetime, str],
    open_price: float,
    high: float,
    low: float,
    close: float,
    volume: float,
    dividends: float = 0.0,
    stock_splits: float = 0.0
) -> Price:
    """
    Insert or update a price record.

    Args:
        session: Database session.
        symbol_id: Foreign key to symbols.
        date_val: Trading date.
        open_price: Opening price.
        high: High price.
        low: Low price.
        close: Closing price.
        volume: Trading volume.
        dividends: Dividend amount.
        stock_splits: Stock split ratio.

    Returns:
        Price object.
    """
    if isinstance(date_val, str):
        date_val = pd.to_datetime(date_val).date()
    elif isinstance(date_val, datetime):
        date_val = date_val.date()

    price = session.query(Price).filter(
        and_(Price.symbol_id == symbol_id, Price.date == date_val)
    ).first()

    if price is None:
        price = Price(
            symbol_id=symbol_id, date=date_val,
            open=open_price, high=high, low=low, close=close, volume=volume,
            dividends=dividends, stock_splits=stock_splits
        )
        session.add(price)
    else:
        price.open = open_price
        price.high = high
        price.low = low
        price.close = close
        price.volume = volume
        price.dividends = dividends
        price.stock_splits = stock_splits

    session.flush()
    return price


def bulk_upsert_prices(session: Session, prices_df: pd.DataFrame, symbol_id: int) -> int:
    """
    Bulk insert or update prices from DataFrame.

    Args:
        session: Database session.
        prices_df: DataFrame with columns: Date, Open, High, Low, Close, Volume.
        symbol_id: Foreign key to symbols.

    Returns:
        Number of records processed.
    """
    count = 0
    for _, row in prices_df.iterrows():
        date_val = pd.to_datetime(row.get('Date', row.name)).date()

        upsert_price(
            session, symbol_id, date_val,
            open_price=float(row['Open']),
            high=float(row['High']),
            low=float(row['Low']),
            close=float(row['Close']),
            volume=float(row['Volume']),
            dividends=float(row.get('Dividends', 0.0)),
            stock_splits=float(row.get('Stock_Splits', 0.0))
        )
        count += 1

    return count


def get_prices_df(
    session: Session,
    symbol_id: int,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None
) -> pd.DataFrame:
    """
    Get prices as DataFrame.

    Args:
        session: Database session.
        symbol_id: Foreign key to symbols.
        start_date: Start date filter (inclusive).
        end_date: End date filter (inclusive).

    Returns:
        DataFrame with price data.
    """
    query = session.query(Price).filter(Price.symbol_id == symbol_id)

    if start_date:
        query = query.filter(Price.date >= start_date)
    if end_date:
        query = query.filter(Price.date <= end_date)

    query = query.order_by(Price.date)
    prices = query.all()

    if not prices:
        return pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume'])

    data = [{
        'date': p.date,
        'open': p.open,
        'high': p.high,
        'low': p.low,
        'close': p.close,
        'volume': p.volume,
        'dividends': p.dividends,
        'stock_splits': p.stock_splits
    } for p in prices]

    return pd.DataFrame(data)


# =============================================================================
# Daily Sentiment Operations
# =============================================================================

def upsert_daily_sentiment(
    session: Session,
    symbol_id: int,
    date_val: Union[date, datetime, str],
    sentiment_score: float,
    sentiment_confidence: float,
    article_count: int = 0,
    bullish_ratio: float = 0.0,
    bearish_ratio: float = 0.0,
    sentiment_std: float = 0.0,
    signal_valid: bool = False
) -> DailySentiment:
    """
    Insert or update daily sentiment.

    Args:
        session: Database session.
        symbol_id: Foreign key to symbols.
        date_val: Date of aggregation.
        sentiment_score: Time-weighted average sentiment.
        sentiment_confidence: Average confidence.
        article_count: Number of articles.
        bullish_ratio: Fraction of bullish articles.
        bearish_ratio: Fraction of bearish articles.
        sentiment_std: Standard deviation.
        signal_valid: Whether signal meets quality thresholds.

    Returns:
        DailySentiment object.
    """
    if isinstance(date_val, str):
        date_val = pd.to_datetime(date_val).date()
    elif isinstance(date_val, datetime):
        date_val = date_val.date()

    sentiment = session.query(DailySentiment).filter(
        and_(DailySentiment.symbol_id == symbol_id, DailySentiment.date == date_val)
    ).first()

    if sentiment is None:
        sentiment = DailySentiment(
            symbol_id=symbol_id, date=date_val,
            sentiment_score=sentiment_score, sentiment_confidence=sentiment_confidence,
            article_count=article_count, bullish_ratio=bullish_ratio,
            bearish_ratio=bearish_ratio, sentiment_std=sentiment_std,
            signal_valid=signal_valid
        )
        session.add(sentiment)
    else:
        sentiment.sentiment_score = sentiment_score
        sentiment.sentiment_confidence = sentiment_confidence
        sentiment.article_count = article_count
        sentiment.bullish_ratio = bullish_ratio
        sentiment.bearish_ratio = bearish_ratio
        sentiment.sentiment_std = sentiment_std
        sentiment.signal_valid = signal_valid
        sentiment.updated_at = datetime.utcnow()

    session.flush()
    return sentiment


def bulk_upsert_daily_sentiment(
    session: Session,
    sentiment_df: pd.DataFrame,
    symbol_id: int
) -> int:
    """
    Bulk insert or update daily sentiment from DataFrame.

    Args:
        session: Database session.
        sentiment_df: DataFrame with daily sentiment columns.
        symbol_id: Foreign key to symbols.

    Returns:
        Number of records processed.
    """
    count = 0
    for _, row in sentiment_df.iterrows():
        date_val = pd.to_datetime(row.get('date', row.name)).date()

        upsert_daily_sentiment(
            session, symbol_id, date_val,
            sentiment_score=float(row['sentiment_score']),
            sentiment_confidence=float(row['sentiment_confidence']),
            article_count=int(row.get('article_count', 0)),
            bullish_ratio=float(row.get('bullish_ratio', 0.0)),
            bearish_ratio=float(row.get('bearish_ratio', 0.0)),
            sentiment_std=float(row.get('sentiment_std', 0.0)),
            signal_valid=bool(row.get('signal_valid', False))
        )
        count += 1

    return count


def get_daily_sentiment_df(
    session: Session,
    symbol_id: int,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None
) -> pd.DataFrame:
    """
    Get daily sentiment as DataFrame.

    Args:
        session: Database session.
        symbol_id: Foreign key to symbols.
        start_date: Start date filter (inclusive).
        end_date: End date filter (inclusive).

    Returns:
        DataFrame with daily sentiment data.
    """
    query = session.query(DailySentiment).filter(DailySentiment.symbol_id == symbol_id)

    if start_date:
        query = query.filter(DailySentiment.date >= start_date)
    if end_date:
        query = query.filter(DailySentiment.date <= end_date)

    query = query.order_by(DailySentiment.date)
    records = query.all()

    if not records:
        return pd.DataFrame(columns=[
            'date', 'sentiment_score', 'sentiment_confidence',
            'article_count', 'bullish_ratio', 'bearish_ratio',
            'sentiment_std', 'signal_valid'
        ])

    data = [{
        'date': r.date,
        'sentiment_score': r.sentiment_score,
        'sentiment_confidence': r.sentiment_confidence,
        'article_count': r.article_count,
        'bullish_ratio': r.bullish_ratio,
        'bearish_ratio': r.bearish_ratio,
        'sentiment_std': r.sentiment_std,
        'signal_valid': r.signal_valid
    } for r in records]

    return pd.DataFrame(data)


# =============================================================================
# Feature Operations
# =============================================================================

def upsert_feature(
    session: Session,
    symbol_id: int,
    date_val: Union[date, datetime, str],
    feature_data: Dict[str, Any],
    feature_version: str = 'v1'
) -> Feature:
    """
    Insert or update feature data.

    Args:
        session: Database session.
        symbol_id: Foreign key to symbols.
        date_val: Feature date.
        feature_data: JSON blob with all features.
        feature_version: Version identifier.

    Returns:
        Feature object.
    """
    if isinstance(date_val, str):
        date_val = pd.to_datetime(date_val).date()
    elif isinstance(date_val, datetime):
        date_val = date_val.date()

    feature = session.query(Feature).filter(
        and_(Feature.symbol_id == symbol_id, Feature.date == date_val)
    ).first()

    if feature is None:
        feature = Feature(
            symbol_id=symbol_id, date=date_val,
            feature_data=feature_data, feature_version=feature_version
        )
        session.add(feature)
    else:
        feature.feature_data = feature_data
        feature.feature_version = feature_version
        feature.updated_at = datetime.utcnow()

    session.flush()
    return feature


def get_features_df(
    session: Session,
    symbol_id: int,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None
) -> pd.DataFrame:
    """
    Get features as DataFrame.

    Args:
        session: Database session.
        symbol_id: Foreign key to symbols.
        start_date: Start date filter (inclusive).
        end_date: End date filter (inclusive).

    Returns:
        DataFrame with feature data (JSON unpacked).
    """
    query = session.query(Feature).filter(Feature.symbol_id == symbol_id)

    if start_date:
        query = query.filter(Feature.date >= start_date)
    if end_date:
        query = query.filter(Feature.date <= end_date)

    query = query.order_by(Feature.date)
    records = query.all()

    if not records:
        return pd.DataFrame(columns=['date'])

    data = []
    for r in records:
        row = {'date': r.date, **r.feature_data}
        data.append(row)

    return pd.DataFrame(data)


# =============================================================================
# Signal Operations
# =============================================================================

def insert_signal(
    session: Session,
    symbol_id: int,
    timestamp: Union[datetime, str],
    direction: int,
    strength: float,
    confidence: float,
    expected_return: Optional[float] = None,
    signal_type: str = 'combined'
) -> Signal:
    """
    Insert a new signal.

    Args:
        session: Database session.
        symbol_id: Foreign key to symbols.
        timestamp: Signal generation timestamp.
        direction: Signal direction (1=long, -1=short, 0=neutral).
        strength: Signal strength [0, 1].
        confidence: Model confidence [0, 1].
        expected_return: Expected return.
        signal_type: Type of signal.

    Returns:
        Signal object.
    """
    if isinstance(timestamp, str):
        timestamp = pd.to_datetime(timestamp)

    signal = Signal(
        symbol_id=symbol_id, timestamp=timestamp,
        direction=direction, strength=strength, confidence=confidence,
        expected_return=expected_return, signal_type=signal_type
    )
    session.add(signal)
    session.flush()
    return signal


def get_latest_signals(
    session: Session,
    symbol_id: Optional[int] = None,
    limit: int = 10
) -> List[Signal]:
    """
    Get the latest signals.

    Args:
        session: Database session.
        symbol_id: Filter by symbol (optional).
        limit: Maximum number of signals to return.

    Returns:
        List of Signal objects.
    """
    query = session.query(Signal)

    if symbol_id is not None:
        query = query.filter(Signal.symbol_id == symbol_id)

    return query.order_by(Signal.timestamp.desc()).limit(limit).all()


def get_signals_df(
    session: Session,
    symbol_id: Optional[int] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Get signals as DataFrame.

    Args:
        session: Database session.
        symbol_id: Filter by symbol (optional).
        start_date: Start timestamp filter (inclusive).
        end_date: End timestamp filter (inclusive).

    Returns:
        DataFrame with signal data.
    """
    query = session.query(Signal)

    if symbol_id is not None:
        query = query.filter(Signal.symbol_id == symbol_id)
    if start_date:
        query = query.filter(Signal.timestamp >= start_date)
    if end_date:
        query = query.filter(Signal.timestamp <= end_date)

    query = query.order_by(Signal.timestamp)
    records = query.all()

    if not records:
        return pd.DataFrame(columns=[
            'timestamp', 'symbol_id', 'direction', 'strength',
            'confidence', 'expected_return', 'signal_type'
        ])

    data = [{
        'timestamp': r.timestamp,
        'symbol_id': r.symbol_id,
        'direction': r.direction,
        'strength': r.strength,
        'confidence': r.confidence,
        'expected_return': r.expected_return,
        'signal_type': r.signal_type
    } for r in records]

    return pd.DataFrame(data)


# =============================================================================
# Prediction Operations
# =============================================================================

def upsert_prediction(
    session: Session,
    symbol_id: int,
    date_val: Union[date, datetime, str],
    direction: int,
    probability: float,
    expected_return: Optional[float] = None,
    model_name: str = 'xgboost',
    model_version: Optional[str] = None,
    features_used: int = 0
) -> Prediction:
    """
    Insert or update a prediction.

    Args:
        session: Database session.
        symbol_id: Foreign key to symbols.
        date_val: Prediction date.
        direction: Predicted direction.
        probability: Probability of up movement.
        expected_return: Expected return.
        model_name: Name of the model.
        model_version: Version of the model.
        features_used: Number of features used.

    Returns:
        Prediction object.
    """
    if isinstance(date_val, str):
        date_val = pd.to_datetime(date_val).date()
    elif isinstance(date_val, datetime):
        date_val = date_val.date()

    prediction = session.query(Prediction).filter(
        and_(
            Prediction.symbol_id == symbol_id,
            Prediction.date == date_val,
            Prediction.model_name == model_name
        )
    ).first()

    if prediction is None:
        prediction = Prediction(
            symbol_id=symbol_id, date=date_val,
            direction=direction, probability=probability,
            expected_return=expected_return, model_name=model_name,
            model_version=model_version, features_used=features_used
        )
        session.add(prediction)
    else:
        prediction.direction = direction
        prediction.probability = probability
        prediction.expected_return = expected_return
        prediction.model_version = model_version
        prediction.features_used = features_used

    session.flush()
    return prediction


# =============================================================================
# Aggregation Queries
# =============================================================================

def get_sentiment_summary(
    session: Session,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None
) -> pd.DataFrame:
    """
    Get aggregated sentiment summary across all symbols.

    Args:
        session: Database session.
        start_date: Start date filter.
        end_date: End date filter.

    Returns:
        DataFrame with summary statistics per symbol.
    """
    query = session.query(
        Symbol.ticker,
        func.avg(DailySentiment.sentiment_score).label('avg_sentiment'),
        func.avg(DailySentiment.sentiment_confidence).label('avg_confidence'),
        func.sum(DailySentiment.article_count).label('total_articles'),
        func.avg(DailySentiment.bullish_ratio).label('avg_bullish_ratio'),
        func.avg(DailySentiment.bearish_ratio).label('avg_bearish_ratio'),
        func.count(DailySentiment.id).label('days_count')
    ).join(Symbol).group_by(Symbol.ticker)

    if start_date:
        query = query.filter(DailySentiment.date >= start_date)
    if end_date:
        query = query.filter(DailySentiment.date <= end_date)

    results = query.all()

    if not results:
        return pd.DataFrame(columns=[
            'ticker', 'avg_sentiment', 'avg_confidence', 'total_articles',
            'avg_bullish_ratio', 'avg_bearish_ratio', 'days_count'
        ])

    data = [{
        'ticker': r.ticker,
        'avg_sentiment': r.avg_sentiment,
        'avg_confidence': r.avg_confidence,
        'total_articles': r.total_articles,
        'avg_bullish_ratio': r.avg_bullish_ratio,
        'avg_bearish_ratio': r.avg_bearish_ratio,
        'days_count': r.days_count
    } for r in results]

    return pd.DataFrame(data)


def get_latest_sentiment_by_symbol(session: Session) -> pd.DataFrame:
    """
    Get the latest daily sentiment for each symbol.

    Args:
        session: Database session.

    Returns:
        DataFrame with latest sentiment per symbol.
    """
    # Subquery to get max date per symbol
    subq = session.query(
        DailySentiment.symbol_id,
        func.max(DailySentiment.date).label('max_date')
    ).group_by(DailySentiment.symbol_id).subquery()

    query = session.query(
        Symbol.ticker,
        DailySentiment.date,
        DailySentiment.sentiment_score,
        DailySentiment.sentiment_confidence,
        DailySentiment.article_count,
        DailySentiment.bullish_ratio,
        DailySentiment.bearish_ratio
    ).join(Symbol).join(
        subq,
        and_(
            DailySentiment.symbol_id == subq.c.symbol_id,
            DailySentiment.date == subq.c.max_date
        )
    )

    results = query.all()

    if not results:
        return pd.DataFrame(columns=[
            'ticker', 'date', 'sentiment_score', 'sentiment_confidence',
            'article_count', 'bullish_ratio', 'bearish_ratio'
        ])

    data = [{
        'ticker': r.ticker,
        'date': r.date,
        'sentiment_score': r.sentiment_score,
        'sentiment_confidence': r.sentiment_confidence,
        'article_count': r.article_count,
        'bullish_ratio': r.bullish_ratio,
        'bearish_ratio': r.bearish_ratio
    } for r in results]

    return pd.DataFrame(data)
