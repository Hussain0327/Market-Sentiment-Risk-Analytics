"""
Price data client using yfinance for fetching historical stock prices.
"""

from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf


class PriceClient:
    """Client for fetching price data from yfinance."""

    # Valid periods for yfinance
    VALID_PERIODS = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]

    # Valid intervals for yfinance
    VALID_INTERVALS = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]

    def __init__(self):
        """Initialize the price client."""
        pass

    def get_historical_prices(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for a symbol.

        Args:
            symbol: Stock symbol (e.g., "AAPL").
            period: Data period. Options: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max.
            interval: Data interval. Options: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo.
            start: Start date (YYYY-MM-DD). If provided, period is ignored.
            end: End date (YYYY-MM-DD). Defaults to today.

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume, Adj Close.
            Index is DatetimeIndex.
        """
        ticker = yf.Ticker(symbol)

        if start:
            df = ticker.history(start=start, end=end, interval=interval)
        else:
            df = ticker.history(period=period, interval=interval)

        if df.empty:
            return pd.DataFrame()

        # Ensure consistent column names
        df.columns = [col.title().replace(" ", "_") for col in df.columns]

        # Add symbol column
        df["Symbol"] = symbol.upper()

        return df

    def get_multiple_symbols(
        self,
        symbols: list[str],
        period: str = "1y",
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple symbols.

        Args:
            symbols: List of stock symbols.
            period: Data period.
            interval: Data interval.
            start: Start date (YYYY-MM-DD).
            end: End date (YYYY-MM-DD).

        Returns:
            Dictionary mapping symbols to DataFrames.
        """
        results = {}
        for symbol in symbols:
            try:
                df = self.get_historical_prices(
                    symbol, period=period, interval=interval, start=start, end=end
                )
                if not df.empty:
                    results[symbol] = df
            except Exception:
                # Skip symbols that fail
                continue
        return results

    def get_combined_prices(
        self,
        symbols: list[str],
        period: str = "1y",
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
        column: str = "Close"
    ) -> pd.DataFrame:
        """
        Fetch and combine a specific column for multiple symbols.

        Args:
            symbols: List of stock symbols.
            period: Data period.
            interval: Data interval.
            start: Start date.
            end: End date.
            column: Column to extract (default: Close).

        Returns:
            DataFrame with symbols as columns and dates as index.
        """
        data = self.get_multiple_symbols(symbols, period, interval, start, end)
        if not data:
            return pd.DataFrame()

        combined = pd.DataFrame()
        for symbol, df in data.items():
            if column in df.columns:
                combined[symbol] = df[column]

        return combined

    def get_latest_quote(self, symbol: str) -> dict:
        """
        Get the latest quote for a symbol.

        Args:
            symbol: Stock symbol.

        Returns:
            Dictionary with latest price data and metadata.
        """
        ticker = yf.Ticker(symbol)
        info = ticker.info

        # Get the most recent price data
        hist = ticker.history(period="1d")

        quote = {
            "symbol": symbol.upper(),
            "name": info.get("shortName", info.get("longName", symbol)),
            "currency": info.get("currency", "USD"),
            "exchange": info.get("exchange", ""),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "dividend_yield": info.get("dividendYield"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
            "avg_volume": info.get("averageVolume"),
        }

        if not hist.empty:
            latest = hist.iloc[-1]
            quote.update({
                "open": latest.get("Open"),
                "high": latest.get("High"),
                "low": latest.get("Low"),
                "close": latest.get("Close"),
                "volume": latest.get("Volume"),
                "date": hist.index[-1].strftime("%Y-%m-%d"),
            })

        return quote

    def calculate_returns(
        self,
        prices: pd.DataFrame,
        method: str = "simple",
        periods: int = 1,
        column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Calculate returns from price data.

        Args:
            prices: DataFrame with price data.
            method: "simple" for arithmetic returns, "log" for logarithmic returns.
            periods: Number of periods for return calculation (1 = daily if daily data).
            column: Specific column to calculate returns for. If None, calculates for all numeric columns.

        Returns:
            DataFrame with returns.
        """
        if column:
            if column not in prices.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame")
            data = prices[[column]]
        else:
            # Select only numeric columns
            data = prices.select_dtypes(include=[np.number])

        if method == "log":
            returns = np.log(data / data.shift(periods))
        else:
            returns = data.pct_change(periods=periods)

        return returns.dropna()

    def calculate_daily_returns(
        self,
        symbol: str,
        period: str = "1y",
        method: str = "simple"
    ) -> pd.Series:
        """
        Calculate daily returns for a symbol.

        Args:
            symbol: Stock symbol.
            period: Data period.
            method: Return calculation method.

        Returns:
            Series with daily returns.
        """
        prices = self.get_historical_prices(symbol, period=period)
        if prices.empty:
            return pd.Series()

        returns = self.calculate_returns(prices, method=method, column="Close")
        return returns["Close"]

    def calculate_rolling_volatility(
        self,
        prices: pd.DataFrame,
        window: int = 21,
        column: str = "Close",
        annualize: bool = True,
        trading_days: int = 252
    ) -> pd.Series:
        """
        Calculate rolling volatility.

        Args:
            prices: DataFrame with price data.
            window: Rolling window size (default: 21 for monthly).
            column: Price column to use.
            annualize: Whether to annualize the volatility.
            trading_days: Number of trading days per year.

        Returns:
            Series with rolling volatility.
        """
        if column not in prices.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")

        returns = self.calculate_returns(prices, column=column)
        vol = returns[column].rolling(window=window).std()

        if annualize:
            vol = vol * np.sqrt(trading_days)

        return vol

    def fill_missing_data(
        self,
        prices: pd.DataFrame,
        method: str = "ffill",
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fill missing data in price DataFrame.

        Args:
            prices: DataFrame with potential missing data.
            method: Fill method - "ffill" (forward fill), "bfill" (backward fill), "interpolate".
            limit: Maximum number of consecutive NaN values to fill.

        Returns:
            DataFrame with missing values filled.
        """
        if method == "interpolate":
            return prices.interpolate(limit=limit)
        elif method == "ffill":
            return prices.ffill(limit=limit)
        elif method == "bfill":
            return prices.bfill(limit=limit)
        else:
            raise ValueError(f"Unknown fill method: {method}")

    def get_sector_info(self, symbol: str) -> dict:
        """
        Get sector and industry information for a symbol.

        Args:
            symbol: Stock symbol.

        Returns:
            Dictionary with sector and industry info.
        """
        ticker = yf.Ticker(symbol)
        info = ticker.info

        return {
            "symbol": symbol.upper(),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "country": info.get("country", ""),
            "website": info.get("website", ""),
            "full_time_employees": info.get("fullTimeEmployees"),
        }

    def validate_symbol(self, symbol: str) -> bool:
        """
        Check if a symbol is valid and has data available.

        Args:
            symbol: Stock symbol to validate.

        Returns:
            True if symbol is valid, False otherwise.
        """
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d")
            return not hist.empty
        except Exception:
            return False

    def validate_symbols(self, symbols: list[str]) -> dict[str, bool]:
        """
        Validate multiple symbols.

        Args:
            symbols: List of stock symbols.

        Returns:
            Dictionary mapping symbols to their validity status.
        """
        return {symbol: self.validate_symbol(symbol) for symbol in symbols}
