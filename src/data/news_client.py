"""
Finnhub News API client for fetching market news and sentiment data.
"""

import os
import time
import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()


class FinnhubNewsClient:
    """Client for fetching news from Finnhub API."""

    BASE_URL = "https://finnhub.io/api/v1"
    RATE_LIMIT_CALLS = 60  # Free tier: 60 calls per minute
    RATE_LIMIT_WINDOW = 60  # seconds

    def __init__(self, api_key: Optional[str] = None, cache_dir: Optional[str] = None):
        """
        Initialize the Finnhub news client.

        Args:
            api_key: Finnhub API key. If None, reads from FINNHUB_API_KEY env var.
            cache_dir: Directory to cache responses. Defaults to data/raw/news_cache.
        """
        self.api_key = api_key or os.getenv("FINNHUB_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Finnhub API key required. Set FINNHUB_API_KEY env var or pass api_key."
            )

        # Setup cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path(__file__).parent.parent.parent / "data" / "raw" / "news_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Rate limiting tracking
        self._call_times: list[float] = []

    def _rate_limit(self) -> None:
        """Enforce rate limiting by waiting if necessary."""
        now = time.time()
        # Remove calls outside the window
        self._call_times = [t for t in self._call_times if now - t < self.RATE_LIMIT_WINDOW]

        if len(self._call_times) >= self.RATE_LIMIT_CALLS:
            # Wait until oldest call expires
            sleep_time = self.RATE_LIMIT_WINDOW - (now - self._call_times[0]) + 0.1
            if sleep_time > 0:
                time.sleep(sleep_time)
            self._call_times = self._call_times[1:]

        self._call_times.append(time.time())

    def _get_cache_key(self, endpoint: str, params: dict) -> str:
        """Generate a cache key from endpoint and parameters."""
        param_str = json.dumps(params, sort_keys=True)
        hash_input = f"{endpoint}:{param_str}"
        return hashlib.md5(hash_input.encode()).hexdigest()

    def _get_cached(self, cache_key: str, max_age_hours: int = 24) -> Optional[dict]:
        """
        Get cached response if it exists and is not expired.

        Args:
            cache_key: Cache key to look up.
            max_age_hours: Maximum age of cache in hours.

        Returns:
            Cached data or None if not found/expired.
        """
        cache_file = self.cache_dir / f"{cache_key}.json"
        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "r") as f:
                cached = json.load(f)

            cached_time = datetime.fromisoformat(cached["timestamp"])
            if datetime.now() - cached_time > timedelta(hours=max_age_hours):
                return None

            return cached["data"]
        except (json.JSONDecodeError, KeyError):
            return None

    def _set_cache(self, cache_key: str, data: dict) -> None:
        """Save response to cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        cached = {
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        with open(cache_file, "w") as f:
            json.dump(cached, f)

    def _make_request(
        self,
        endpoint: str,
        params: Optional[dict] = None,
        use_cache: bool = True,
        cache_max_age_hours: int = 24,
        max_retries: int = 3
    ) -> dict:
        """
        Make an API request with rate limiting, caching, and retries.

        Args:
            endpoint: API endpoint (e.g., "/company-news").
            params: Query parameters.
            use_cache: Whether to use caching.
            cache_max_age_hours: Maximum cache age in hours.
            max_retries: Maximum number of retries on failure.

        Returns:
            API response as dict.

        Raises:
            requests.HTTPError: If request fails after retries.
        """
        params = params or {}
        params["token"] = self.api_key

        # Check cache first
        if use_cache:
            cache_key = self._get_cache_key(endpoint, {k: v for k, v in params.items() if k != "token"})
            cached = self._get_cached(cache_key, cache_max_age_hours)
            if cached is not None:
                return cached

        url = f"{self.BASE_URL}{endpoint}"

        for attempt in range(max_retries):
            try:
                self._rate_limit()
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

                # Cache successful response
                if use_cache:
                    self._set_cache(cache_key, data)

                return data

            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise
                wait_time = 2 ** attempt  # Exponential backoff
                time.sleep(wait_time)

        return {}

    def get_company_news(
        self,
        symbol: str,
        from_date: str,
        to_date: str,
        use_cache: bool = True
    ) -> list[dict]:
        """
        Fetch company-specific news.

        Args:
            symbol: Stock symbol (e.g., "AAPL").
            from_date: Start date in YYYY-MM-DD format.
            to_date: End date in YYYY-MM-DD format.
            use_cache: Whether to use response caching.

        Returns:
            List of news articles with keys: category, datetime, headline, id,
            image, related, source, summary, url.
        """
        params = {
            "symbol": symbol.upper(),
            "from": from_date,
            "to": to_date
        }
        return self._make_request("/company-news", params, use_cache=use_cache)

    def get_company_news_df(
        self,
        symbol: str,
        from_date: str,
        to_date: str,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch company news as a DataFrame.

        Args:
            symbol: Stock symbol.
            from_date: Start date in YYYY-MM-DD format.
            to_date: End date in YYYY-MM-DD format.
            use_cache: Whether to use response caching.

        Returns:
            DataFrame with news articles.
        """
        news = self.get_company_news(symbol, from_date, to_date, use_cache)
        if not news:
            return pd.DataFrame()

        df = pd.DataFrame(news)
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], unit="s")
        df["symbol"] = symbol.upper()
        return df

    def get_market_news(
        self,
        category: str = "general",
        min_id: int = 0,
        use_cache: bool = True
    ) -> list[dict]:
        """
        Fetch general market news.

        Args:
            category: News category. Options: general, forex, crypto, merger.
            min_id: Use this to get only news newer than this ID.
            use_cache: Whether to use response caching.

        Returns:
            List of news articles.
        """
        params = {"category": category}
        if min_id > 0:
            params["minId"] = min_id
        return self._make_request("/news", params, use_cache=use_cache)

    def get_market_news_df(
        self,
        category: str = "general",
        min_id: int = 0,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch market news as a DataFrame.

        Args:
            category: News category.
            min_id: Minimum news ID.
            use_cache: Whether to use response caching.

        Returns:
            DataFrame with news articles.
        """
        news = self.get_market_news(category, min_id, use_cache)
        if not news:
            return pd.DataFrame()

        df = pd.DataFrame(news)
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], unit="s")
        return df

    def get_news_sentiment(self, symbol: str, use_cache: bool = True) -> dict:
        """
        Fetch Finnhub's pre-computed news sentiment for a symbol.

        Args:
            symbol: Stock symbol.
            use_cache: Whether to use response caching.

        Returns:
            Sentiment data with buzz, sentiment scores, and sector info.
        """
        params = {"symbol": symbol.upper()}
        return self._make_request("/news-sentiment", params, use_cache=use_cache)

    def get_news_sentiment_df(self, symbol: str, use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch news sentiment as a DataFrame.

        Args:
            symbol: Stock symbol.
            use_cache: Whether to use response caching.

        Returns:
            DataFrame with sentiment data.
        """
        sentiment = self.get_news_sentiment(symbol, use_cache)
        if not sentiment:
            return pd.DataFrame()

        # Flatten nested structure
        flat_data = {
            "symbol": sentiment.get("symbol", symbol.upper()),
            "articles_in_last_week": sentiment.get("buzz", {}).get("articlesInLastWeek", 0),
            "buzz_score": sentiment.get("buzz", {}).get("buzz", 0),
            "weekly_average": sentiment.get("buzz", {}).get("weeklyAverage", 0),
            "bearish_percent": sentiment.get("sentiment", {}).get("bearishPercent", 0),
            "bullish_percent": sentiment.get("sentiment", {}).get("bullishPercent", 0),
            "company_news_score": sentiment.get("companyNewsScore", 0),
            "sector_average_bullish": sentiment.get("sectorAverageBullishPercent", 0),
            "sector_average_news_score": sentiment.get("sectorAverageNewsScore", 0),
        }
        return pd.DataFrame([flat_data])

    def get_multiple_company_news(
        self,
        symbols: list[str],
        from_date: str,
        to_date: str,
        use_cache: bool = True
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch news for multiple symbols.

        Args:
            symbols: List of stock symbols.
            from_date: Start date in YYYY-MM-DD format.
            to_date: End date in YYYY-MM-DD format.
            use_cache: Whether to use response caching.

        Returns:
            Dictionary mapping symbols to DataFrames.
        """
        results = {}
        for symbol in symbols:
            results[symbol] = self.get_company_news_df(symbol, from_date, to_date, use_cache)
        return results

    def clear_cache(self, older_than_hours: Optional[int] = None) -> int:
        """
        Clear cached responses.

        Args:
            older_than_hours: Only clear cache older than this. If None, clear all.

        Returns:
            Number of cache files deleted.
        """
        deleted = 0
        for cache_file in self.cache_dir.glob("*.json"):
            if older_than_hours is not None:
                try:
                    with open(cache_file, "r") as f:
                        cached = json.load(f)
                    cached_time = datetime.fromisoformat(cached["timestamp"])
                    if datetime.now() - cached_time <= timedelta(hours=older_than_hours):
                        continue
                except (json.JSONDecodeError, KeyError):
                    pass

            cache_file.unlink()
            deleted += 1

        return deleted
