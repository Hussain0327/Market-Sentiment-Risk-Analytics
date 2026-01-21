"""
Trading Signals API routes.

Provides endpoints for trading signals based on sentiment analysis.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from api.schemas import SignalsResponse, SignalDataPoint, SymbolsResponse
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


@router.get("/{symbol}", response_model=SignalsResponse)
def get_signals(
    symbol: str,
    days: int = Query(default=30, ge=1, le=365, description="Days of signal history")
):
    """
    Get trading signals for a symbol.

    Returns the latest signal and signal history.
    """
    symbol = symbol.upper()
    loader = get_data_loader()

    signals_df = loader.load_sentiment_signals(symbol)

    if signals_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No signal data found for symbol: {symbol}"
        )

    # Limit to requested days
    signals_df = signals_df.tail(days)

    # Get latest signal info
    latest = signals_df.iloc[-1]
    latest_signal = latest.get("signal", "neutral")
    latest_strength = float(latest.get("signal_strength", 0.0))

    # Count signals by type
    signal_counts = signals_df["signal"].value_counts().to_dict()
    bullish_count = signal_counts.get("bullish", 0)
    bearish_count = signal_counts.get("bearish", 0)
    neutral_count = signal_counts.get("neutral", 0)

    # Build history
    history = []
    for _, row in signals_df.iterrows():
        history.append(SignalDataPoint(
            date=row["date"],
            sentiment_score=float(row.get("sentiment_score", 0.0)),
            signal=row.get("signal", "neutral"),
            signal_strength=float(row.get("signal_strength", 0.0)),
        ))

    return SignalsResponse(
        symbol=symbol,
        latest_signal=latest_signal,
        latest_strength=latest_strength,
        bullish_count=bullish_count,
        bearish_count=bearish_count,
        neutral_count=neutral_count,
        history=history,
    )


@router.get("/", response_model=SymbolsResponse)
def list_signal_symbols():
    """
    List all symbols with trading signal data available.
    """
    loader = get_data_loader()
    signals_df = loader.load_sentiment_signals()

    if signals_df.empty:
        return SymbolsResponse(symbols=[], count=0)

    symbols = sorted(signals_df["symbol"].unique().tolist())
    return SymbolsResponse(symbols=symbols, count=len(symbols))


@router.get("/summary/all")
def get_all_signals_summary():
    """
    Get summary of signals for all available symbols.

    Returns the latest signal and counts for each symbol.
    """
    loader = get_data_loader()
    summary_df = loader.get_signal_summary()

    if summary_df.empty:
        return {"symbols": []}

    results = []
    for _, row in summary_df.iterrows():
        results.append({
            "symbol": row["symbol"],
            "latest_signal": row.get("latest_signal", "neutral"),
            "latest_strength": float(row.get("latest_strength", 0.0)),
            "bullish_count": int(row.get("bullish", 0)),
            "bearish_count": int(row.get("bearish", 0)),
            "neutral_count": int(row.get("neutral", 0)),
        })

    return {"symbols": results}
