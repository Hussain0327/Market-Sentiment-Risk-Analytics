"""
Dashboard API Client.

HTTP client for communicating with the FastAPI backend.
Provides methods for all API endpoints with proper error handling.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd


class APIClientError(Exception):
    """Exception raised for API client errors."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class APIClient:
    """
    HTTP client for the Market Sentiment API.

    Example:
        >>> client = APIClient()
        >>> sentiment = client.get_sentiment("AAPL")
        >>> print(f"Sentiment: {sentiment['latest_sentiment']}")
    """

    DEFAULT_BASE_URL = "http://localhost:8000"
    DEFAULT_TIMEOUT = 30.0

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        """
        Initialize the API client.

        Args:
            base_url: Base URL of the API server.
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.Client] = None

    @property
    def client(self) -> httpx.Client:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._client

    def close(self):
        """Close the HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Make an HTTP request to the API.

        Args:
            method: HTTP method (GET, POST, etc.).
            endpoint: API endpoint path.
            params: Query parameters.
            json: JSON body for POST/PUT requests.

        Returns:
            Response data as dictionary.

        Raises:
            APIClientError: If request fails.
        """
        try:
            response = self.client.request(
                method=method,
                url=endpoint,
                params=params,
                json=json,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            error_detail = ""
            try:
                error_data = e.response.json()
                error_detail = error_data.get("detail", str(e))
            except Exception:
                error_detail = str(e)
            raise APIClientError(error_detail, e.response.status_code)
        except httpx.RequestError as e:
            raise APIClientError(f"Request failed: {str(e)}")

    def _get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a GET request."""
        return self._request("GET", endpoint, params=params)

    # Health & Utility endpoints
    def health_check(self) -> Dict[str, str]:
        """
        Check API health status.

        Returns:
            Health status response.
        """
        return self._get("/api/health")

    def is_healthy(self) -> bool:
        """
        Check if the API is healthy.

        Returns:
            True if API is healthy, False otherwise.
        """
        try:
            result = self.health_check()
            return result.get("status") == "healthy"
        except APIClientError:
            return False

    # Symbol endpoints
    def get_symbols(self) -> List[str]:
        """
        Get list of available symbols.

        Returns:
            List of stock symbols.
        """
        result = self._get("/api/prices/symbols")
        return result.get("symbols", [])

    # Sentiment endpoints
    def get_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Get current sentiment for a symbol.

        Args:
            symbol: Stock symbol.

        Returns:
            Sentiment data dictionary.
        """
        return self._get(f"/api/sentiment/{symbol}")

    def get_sentiment_history(
        self, symbol: str, days: int = 30
    ) -> Dict[str, Any]:
        """
        Get sentiment history for a symbol.

        Args:
            symbol: Stock symbol.
            days: Number of days of history.

        Returns:
            Sentiment history data.
        """
        return self._get(f"/api/sentiment/{symbol}/history", params={"days": days})

    def get_sentiment_history_df(
        self, symbol: str, days: int = 30
    ) -> pd.DataFrame:
        """
        Get sentiment history as a DataFrame.

        Args:
            symbol: Stock symbol.
            days: Number of days of history.

        Returns:
            DataFrame with sentiment history.
        """
        data = self.get_sentiment_history(symbol, days)
        if not data.get("data"):
            return pd.DataFrame()

        df = pd.DataFrame(data["data"])
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df

    def list_sentiment_symbols(self) -> List[str]:
        """
        Get list of symbols with sentiment data.

        Returns:
            List of symbols.
        """
        result = self._get("/api/sentiment/")
        return result.get("symbols", [])

    # Risk endpoints
    def get_risk_metrics(self, symbol: str, period: str = "1y") -> Dict[str, Any]:
        """
        Get risk metrics for a symbol.

        Args:
            symbol: Stock symbol.
            period: Price history period.

        Returns:
            Risk metrics data.
        """
        return self._get(f"/api/risk/{symbol}", params={"period": period})

    def get_full_risk_report(self, symbol: str) -> Dict[str, Any]:
        """
        Get detailed risk report for a symbol.

        Args:
            symbol: Stock symbol.

        Returns:
            Full risk report data.
        """
        return self._get(f"/api/risk/{symbol}/full")

    # Signals endpoints
    def get_signals(self, symbol: str, days: int = 30) -> Dict[str, Any]:
        """
        Get trading signals for a symbol.

        Args:
            symbol: Stock symbol.
            days: Days of signal history.

        Returns:
            Signals data.
        """
        return self._get(f"/api/signals/{symbol}", params={"days": days})

    def get_signals_df(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """
        Get trading signals as a DataFrame.

        Args:
            symbol: Stock symbol.
            days: Days of signal history.

        Returns:
            DataFrame with signal history.
        """
        data = self.get_signals(symbol, days)
        if not data.get("history"):
            return pd.DataFrame()

        df = pd.DataFrame(data["history"])
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df

    def get_all_signals_summary(self) -> Dict[str, Any]:
        """
        Get signals summary for all symbols.

        Returns:
            Summary data for all symbols.
        """
        return self._get("/api/signals/summary/all")

    def list_signal_symbols(self) -> List[str]:
        """
        Get list of symbols with signal data.

        Returns:
            List of symbols.
        """
        result = self._get("/api/signals/")
        return result.get("symbols", [])

    # Price endpoints
    def get_price(self, symbol: str) -> Dict[str, Any]:
        """
        Get latest price for a symbol.

        Args:
            symbol: Stock symbol.

        Returns:
            Latest price data.
        """
        return self._get(f"/api/prices/{symbol}")

    def get_price_history(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d"
    ) -> Dict[str, Any]:
        """
        Get price history for a symbol.

        Args:
            symbol: Stock symbol.
            period: Data period (1mo, 3mo, 6mo, 1y, 2y, 5y).
            interval: Data interval (1d, 1wk, 1mo).

        Returns:
            Price history data.
        """
        return self._get(
            f"/api/prices/{symbol}/history",
            params={"period": period, "interval": interval}
        )

    def get_price_history_df(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Get price history as a DataFrame.

        Args:
            symbol: Stock symbol.
            period: Data period.
            interval: Data interval.

        Returns:
            DataFrame with OHLCV data.
        """
        data = self.get_price_history(symbol, period, interval)
        if not data.get("data"):
            return pd.DataFrame()

        df = pd.DataFrame(data["data"])
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df

    def validate_symbol(self, symbol: str) -> Dict[str, Any]:
        """
        Validate a stock symbol.

        Args:
            symbol: Stock symbol to validate.

        Returns:
            Validation result.
        """
        return self._get(f"/api/prices/{symbol}/validate")


# Convenience function for creating a client
def create_client(
    base_url: str = APIClient.DEFAULT_BASE_URL,
    timeout: float = APIClient.DEFAULT_TIMEOUT,
) -> APIClient:
    """
    Create an API client instance.

    Args:
        base_url: Base URL of the API server.
        timeout: Request timeout in seconds.

    Returns:
        APIClient instance.
    """
    return APIClient(base_url=base_url, timeout=timeout)
