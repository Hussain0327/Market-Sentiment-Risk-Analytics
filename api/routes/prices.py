"""
Price Data API routes.

Provides endpoints for price data and market information.
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from api.schemas import (
    PriceResponse,
    PriceHistoryResponse,
    PriceDataPoint,
    SymbolsResponse,
)
from dashboard.data_loader import DashboardDataLoader
from src.data.price_client import PriceClient
from src.data.watchlist import Watchlist

router = APIRouter()

# Shared instances
_data_loader: Optional[DashboardDataLoader] = None
_price_client: Optional[PriceClient] = None
_watchlist: Optional[Watchlist] = None


def get_data_loader() -> DashboardDataLoader:
    """Get or create the data loader instance."""
    global _data_loader
    if _data_loader is None:
        _data_loader = DashboardDataLoader()
    return _data_loader


def get_price_client() -> PriceClient:
    """Get or create the price client instance."""
    global _price_client
    if _price_client is None:
        _price_client = PriceClient()
    return _price_client


def get_watchlist() -> Watchlist:
    """Get or create the watchlist instance."""
    global _watchlist
    if _watchlist is None:
        _watchlist = Watchlist()
    return _watchlist


@router.get("/symbols", response_model=SymbolsResponse)
def list_symbols():
    """
    List all available symbols with price data.

    Returns symbols from the default watchlist and any symbols
    with stored price data.
    """
    loader = get_data_loader()
    watchlist = get_watchlist()

    # Get symbols from both sources
    data_symbols = set(loader.get_available_symbols())
    watchlist_symbols = set(watchlist.get_symbols())

    all_symbols = sorted(data_symbols | watchlist_symbols)
    return SymbolsResponse(symbols=all_symbols, count=len(all_symbols))


@router.get("/{symbol}", response_model=PriceResponse)
def get_price(symbol: str):
    """
    Get latest price data for a symbol.

    Returns the most recent OHLCV data along with market metadata.
    """
    symbol = symbol.upper()
    price_client = get_price_client()

    try:
        quote = price_client.get_latest_quote(symbol)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching price data: {str(e)}"
        )

    if not quote.get("close"):
        raise HTTPException(
            status_code=404,
            detail=f"No price data available for symbol: {symbol}"
        )

    # Parse the date if it exists
    date_val = None
    if quote.get("date"):
        try:
            date_val = datetime.strptime(quote["date"], "%Y-%m-%d")
        except (ValueError, TypeError):
            date_val = None

    return PriceResponse(
        symbol=symbol,
        name=quote.get("name"),
        currency=quote.get("currency", "USD"),
        exchange=quote.get("exchange"),
        open=quote.get("open"),
        high=quote.get("high"),
        low=quote.get("low"),
        close=quote.get("close"),
        volume=int(quote.get("volume", 0)) if quote.get("volume") else None,
        market_cap=quote.get("market_cap"),
        pe_ratio=quote.get("pe_ratio"),
        fifty_two_week_high=quote.get("fifty_two_week_high"),
        fifty_two_week_low=quote.get("fifty_two_week_low"),
        date=date_val,
    )


@router.get("/{symbol}/history", response_model=PriceHistoryResponse)
def get_price_history(
    symbol: str,
    period: str = Query(
        default="1y",
        description="Price history period (1mo, 3mo, 6mo, 1y, 2y, 5y)"
    ),
    interval: str = Query(
        default="1d",
        description="Data interval (1d, 1wk, 1mo)"
    )
):
    """
    Get historical price data for a symbol.

    Returns OHLCV data for the specified period and interval.
    """
    symbol = symbol.upper()
    price_client = get_price_client()

    # Validate period
    valid_periods = ["1mo", "3mo", "6mo", "1y", "2y", "5y", "ytd", "max"]
    if period not in valid_periods:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid period. Must be one of: {', '.join(valid_periods)}"
        )

    # Validate interval
    valid_intervals = ["1d", "1wk", "1mo"]
    if interval not in valid_intervals:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid interval. Must be one of: {', '.join(valid_intervals)}"
        )

    try:
        prices_df = price_client.get_historical_prices(
            symbol, period=period, interval=interval
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching historical prices: {str(e)}"
        )

    if prices_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No price history found for symbol: {symbol}"
        )

    # Build response data
    data_points = []
    for idx, row in prices_df.iterrows():
        try:
            date_val = idx if hasattr(idx, 'strftime') else datetime.now()
            data_points.append(PriceDataPoint(
                date=date_val,
                open=float(row.get("Open", 0)),
                high=float(row.get("High", 0)),
                low=float(row.get("Low", 0)),
                close=float(row.get("Close", 0)),
                volume=int(row.get("Volume", 0)),
            ))
        except (ValueError, TypeError):
            continue

    return PriceHistoryResponse(
        symbol=symbol,
        data=data_points,
        count=len(data_points),
    )


@router.get("/{symbol}/validate")
def validate_symbol(symbol: str):
    """
    Validate if a symbol exists and has data available.
    """
    symbol = symbol.upper()
    watchlist = get_watchlist()

    result = watchlist.validate_symbol(symbol)
    return result
