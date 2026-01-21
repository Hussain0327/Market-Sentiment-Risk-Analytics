"""
Sentiment API routes.

Provides endpoints for sentiment analysis data and metrics.
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from api.schemas import (
    SentimentResponse,
    SentimentHistoryResponse,
    SentimentDataPoint,
    SymbolsResponse,
)
from dashboard.data_loader import DashboardDataLoader

router = APIRouter()

# Shared data loader instance
_data_loader: Optional[DashboardDataLoader] = None


def get_data_loader() -> DashboardDataLoader:
    """Get or create the data loader instance."""
    global _data_loader
    if _data_loader is None:
        _data_loader = DashboardDataLoader()
    return _data_loader


@router.get("/{symbol}", response_model=SentimentResponse)
def get_sentiment(symbol: str):
    """
    Get current sentiment metrics for a symbol.

    Returns the latest sentiment score, confidence, and signal.
    """
    symbol = symbol.upper()
    loader = get_data_loader()

    # Load daily sentiment for the symbol
    sentiment_df = loader.load_daily_sentiment(symbol)

    if sentiment_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No sentiment data found for symbol: {symbol}"
        )

    # Get the latest row
    latest = sentiment_df.iloc[-1]

    # Load signals to get signal info
    signals_df = loader.load_sentiment_signals(symbol)
    if not signals_df.empty:
        latest_signal = signals_df.iloc[-1]
        signal = latest_signal.get("signal", "neutral")
        signal_strength = latest_signal.get("signal_strength", 0.0)
    else:
        # Determine signal from sentiment score
        score = latest["sentiment_score"]
        if score > 0.3:
            signal = "bullish"
        elif score < -0.3:
            signal = "bearish"
        else:
            signal = "neutral"
        signal_strength = abs(score)

    return SentimentResponse(
        symbol=symbol,
        latest_sentiment=float(latest["sentiment_score"]),
        sentiment_confidence=float(latest.get("sentiment_confidence", 0.5)),
        article_count=int(latest.get("article_count", 0)),
        bullish_ratio=float(latest.get("bullish_ratio", 0.0)),
        bearish_ratio=float(latest.get("bearish_ratio", 0.0)),
        signal=signal,
        signal_strength=float(signal_strength),
        last_updated=latest["date"],
    )


@router.get("/{symbol}/history", response_model=SentimentHistoryResponse)
def get_sentiment_history(
    symbol: str,
    days: int = Query(default=30, ge=1, le=365, description="Number of days of history")
):
    """
    Get historical sentiment data for a symbol.

    Returns daily sentiment scores for the specified number of days.
    """
    symbol = symbol.upper()
    loader = get_data_loader()

    sentiment_df = loader.load_daily_sentiment(symbol)

    if sentiment_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No sentiment data found for symbol: {symbol}"
        )

    # Limit to requested days
    sentiment_df = sentiment_df.tail(days)

    data_points = []
    for _, row in sentiment_df.iterrows():
        data_points.append(SentimentDataPoint(
            date=row["date"],
            sentiment_score=float(row["sentiment_score"]),
            sentiment_confidence=float(row.get("sentiment_confidence", 0.5)),
            article_count=int(row.get("article_count", 0)),
            bullish_ratio=float(row.get("bullish_ratio", 0.0)),
            bearish_ratio=float(row.get("bearish_ratio", 0.0)),
        ))

    return SentimentHistoryResponse(
        symbol=symbol,
        data=data_points,
        count=len(data_points),
    )


@router.get("/", response_model=SymbolsResponse)
def list_sentiment_symbols():
    """
    List all symbols with sentiment data available.
    """
    loader = get_data_loader()
    sentiment_df = loader.load_daily_sentiment()

    if sentiment_df.empty:
        return SymbolsResponse(symbols=[], count=0)

    symbols = sorted(sentiment_df["symbol"].unique().tolist())
    return SymbolsResponse(symbols=symbols, count=len(symbols))
