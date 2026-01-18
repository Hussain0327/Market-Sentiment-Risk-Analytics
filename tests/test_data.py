"""
Unit tests for data collection module.
"""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.news_client import FinnhubNewsClient
from src.data.price_client import PriceClient
from src.data.watchlist import Watchlist


class TestFinnhubNewsClient:
    """Tests for FinnhubNewsClient."""

    @pytest.fixture
    def mock_api_key(self):
        """Provide a mock API key."""
        return "test_api_key_12345"

    @pytest.fixture
    def client(self, mock_api_key, tmp_path):
        """Create a client with a temporary cache directory."""
        return FinnhubNewsClient(api_key=mock_api_key, cache_dir=str(tmp_path / "cache"))

    @pytest.fixture
    def sample_news_response(self):
        """Sample news API response."""
        return [
            {
                "category": "company",
                "datetime": 1704067200,  # 2024-01-01 00:00:00 UTC
                "headline": "Apple announces new product",
                "id": 12345,
                "image": "https://example.com/image.jpg",
                "related": "AAPL",
                "source": "Reuters",
                "summary": "Apple Inc. has announced a new product line.",
                "url": "https://example.com/article"
            },
            {
                "category": "company",
                "datetime": 1704153600,  # 2024-01-02 00:00:00 UTC
                "headline": "Apple stock rises",
                "id": 12346,
                "image": "https://example.com/image2.jpg",
                "related": "AAPL",
                "source": "Bloomberg",
                "summary": "Apple shares rose 2% today.",
                "url": "https://example.com/article2"
            }
        ]

    @pytest.fixture
    def sample_sentiment_response(self):
        """Sample sentiment API response."""
        return {
            "buzz": {
                "articlesInLastWeek": 50,
                "buzz": 1.5,
                "weeklyAverage": 30
            },
            "companyNewsScore": 0.75,
            "sectorAverageBullishPercent": 0.6,
            "sectorAverageNewsScore": 0.5,
            "sentiment": {
                "bearishPercent": 0.2,
                "bullishPercent": 0.8
            },
            "symbol": "AAPL"
        }

    def test_init_with_api_key(self, mock_api_key, tmp_path):
        """Test initialization with explicit API key."""
        client = FinnhubNewsClient(api_key=mock_api_key, cache_dir=str(tmp_path))
        assert client.api_key == mock_api_key

    def test_init_without_api_key_raises(self, tmp_path):
        """Test initialization without API key raises error."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="Finnhub API key required"):
                FinnhubNewsClient(cache_dir=str(tmp_path))

    def test_cache_directory_created(self, client):
        """Test that cache directory is created."""
        assert client.cache_dir.exists()

    @patch("requests.get")
    def test_get_company_news(self, mock_get, client, sample_news_response):
        """Test fetching company news."""
        mock_get.return_value.json.return_value = sample_news_response
        mock_get.return_value.raise_for_status = MagicMock()

        news = client.get_company_news("AAPL", "2024-01-01", "2024-01-15")

        assert len(news) == 2
        assert news[0]["headline"] == "Apple announces new product"
        mock_get.assert_called_once()

    @patch("requests.get")
    def test_get_company_news_df(self, mock_get, client, sample_news_response):
        """Test fetching company news as DataFrame."""
        mock_get.return_value.json.return_value = sample_news_response
        mock_get.return_value.raise_for_status = MagicMock()

        df = client.get_company_news_df("AAPL", "2024-01-01", "2024-01-15")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "symbol" in df.columns
        assert df["symbol"].iloc[0] == "AAPL"
        assert pd.api.types.is_datetime64_any_dtype(df["datetime"])

    @patch("requests.get")
    def test_get_market_news(self, mock_get, client, sample_news_response):
        """Test fetching market news."""
        mock_get.return_value.json.return_value = sample_news_response
        mock_get.return_value.raise_for_status = MagicMock()

        news = client.get_market_news(category="general")

        assert len(news) == 2

    @patch("requests.get")
    def test_get_news_sentiment(self, mock_get, client, sample_sentiment_response):
        """Test fetching news sentiment."""
        mock_get.return_value.json.return_value = sample_sentiment_response
        mock_get.return_value.raise_for_status = MagicMock()

        sentiment = client.get_news_sentiment("AAPL")

        assert sentiment["symbol"] == "AAPL"
        assert "buzz" in sentiment
        assert "sentiment" in sentiment

    @patch("requests.get")
    def test_get_news_sentiment_df(self, mock_get, client, sample_sentiment_response):
        """Test fetching news sentiment as DataFrame."""
        mock_get.return_value.json.return_value = sample_sentiment_response
        mock_get.return_value.raise_for_status = MagicMock()

        df = client.get_news_sentiment_df("AAPL")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df["bullish_percent"].iloc[0] == 0.8
        assert df["buzz_score"].iloc[0] == 1.5

    @patch("requests.get")
    def test_caching_works(self, mock_get, client, sample_news_response):
        """Test that caching prevents duplicate API calls."""
        mock_get.return_value.json.return_value = sample_news_response
        mock_get.return_value.raise_for_status = MagicMock()

        # First call - should hit API
        client.get_company_news("AAPL", "2024-01-01", "2024-01-15")
        assert mock_get.call_count == 1

        # Second call - should use cache
        client.get_company_news("AAPL", "2024-01-01", "2024-01-15")
        assert mock_get.call_count == 1

    @patch("requests.get")
    def test_cache_bypass(self, mock_get, client, sample_news_response):
        """Test that cache can be bypassed."""
        mock_get.return_value.json.return_value = sample_news_response
        mock_get.return_value.raise_for_status = MagicMock()

        client.get_company_news("AAPL", "2024-01-01", "2024-01-15", use_cache=False)
        client.get_company_news("AAPL", "2024-01-01", "2024-01-15", use_cache=False)

        assert mock_get.call_count == 2

    def test_clear_cache(self, client):
        """Test clearing the cache."""
        # Create a fake cache file
        cache_file = client.cache_dir / "test_cache.json"
        cache_file.write_text('{"timestamp": "2020-01-01T00:00:00", "data": {}}')

        deleted = client.clear_cache()
        assert deleted == 1
        assert not cache_file.exists()


class TestPriceClient:
    """Tests for PriceClient."""

    @pytest.fixture
    def client(self):
        """Create a price client."""
        return PriceClient()

    def test_init(self, client):
        """Test initialization."""
        assert client is not None

    def test_get_historical_prices(self, client):
        """Test fetching historical prices (real API call)."""
        df = client.get_historical_prices("AAPL", period="5d")

        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert "Open" in df.columns
        assert "Close" in df.columns
        assert "Volume" in df.columns
        assert "Symbol" in df.columns

    def test_get_historical_prices_with_dates(self, client):
        """Test fetching historical prices with specific dates."""
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")

        df = client.get_historical_prices("AAPL", start=start_date, end=end_date)

        assert isinstance(df, pd.DataFrame)
        # May be empty on weekends/holidays

    def test_get_multiple_symbols(self, client):
        """Test fetching data for multiple symbols."""
        symbols = ["AAPL", "MSFT"]
        data = client.get_multiple_symbols(symbols, period="5d")

        assert isinstance(data, dict)
        assert len(data) >= 1  # At least one should work

    def test_get_combined_prices(self, client):
        """Test getting combined prices for multiple symbols."""
        symbols = ["AAPL", "MSFT"]
        df = client.get_combined_prices(symbols, period="5d")

        assert isinstance(df, pd.DataFrame)
        # Columns should be symbol names

    def test_get_latest_quote(self, client):
        """Test getting latest quote."""
        quote = client.get_latest_quote("AAPL")

        assert isinstance(quote, dict)
        assert quote["symbol"] == "AAPL"
        assert "name" in quote

    def test_calculate_returns_simple(self, client):
        """Test calculating simple returns."""
        df = client.get_historical_prices("AAPL", period="1mo")
        returns = client.calculate_returns(df, method="simple", column="Close")

        assert isinstance(returns, pd.DataFrame)
        assert "Close" in returns.columns

    def test_calculate_returns_log(self, client):
        """Test calculating log returns."""
        df = client.get_historical_prices("AAPL", period="1mo")
        returns = client.calculate_returns(df, method="log", column="Close")

        assert isinstance(returns, pd.DataFrame)

    def test_calculate_daily_returns(self, client):
        """Test calculating daily returns."""
        returns = client.calculate_daily_returns("AAPL", period="1mo")

        assert isinstance(returns, pd.Series)

    def test_calculate_rolling_volatility(self, client):
        """Test calculating rolling volatility."""
        df = client.get_historical_prices("AAPL", period="3mo")
        vol = client.calculate_rolling_volatility(df, window=10)

        assert isinstance(vol, pd.Series)

    def test_fill_missing_data(self, client):
        """Test filling missing data."""
        # Create a DataFrame with missing data
        df = pd.DataFrame({
            "Close": [100, None, 102, None, 104],
            "Volume": [1000, 1100, None, 1300, 1400]
        })

        filled = client.fill_missing_data(df, method="ffill")
        assert not filled.isna().any().any()

    def test_validate_symbol_valid(self, client):
        """Test validating a valid symbol."""
        result = client.validate_symbol("AAPL")
        assert result is True

    def test_validate_symbol_invalid(self, client):
        """Test validating an invalid symbol."""
        result = client.validate_symbol("INVALID_SYMBOL_XYZ123")
        assert result is False

    def test_validate_symbols(self, client):
        """Test validating multiple symbols."""
        symbols = ["AAPL", "INVALID_XYZ"]
        results = client.validate_symbols(symbols)

        assert results["AAPL"] is True
        assert results["INVALID_XYZ"] is False

    def test_get_sector_info(self, client):
        """Test getting sector info."""
        info = client.get_sector_info("AAPL")

        assert isinstance(info, dict)
        assert info["symbol"] == "AAPL"
        assert "sector" in info


class TestWatchlist:
    """Tests for Watchlist."""

    @pytest.fixture
    def watchlist(self):
        """Create a watchlist with some symbols."""
        return Watchlist(symbols=["AAPL", "GOOGL", "MSFT"])

    @pytest.fixture
    def empty_watchlist(self):
        """Create an empty watchlist."""
        return Watchlist(symbols=[])

    def test_init_with_symbols(self, watchlist):
        """Test initialization with symbols."""
        assert len(watchlist) == 3
        assert "AAPL" in watchlist

    def test_init_without_symbols(self):
        """Test initialization without symbols uses defaults."""
        wl = Watchlist()
        assert len(wl) > 0
        assert "AAPL" in wl

    def test_init_with_empty_list(self, empty_watchlist):
        """Test initialization with empty list."""
        assert len(empty_watchlist) == 0

    def test_add_symbol(self, watchlist):
        """Test adding a symbol."""
        result = watchlist.add_symbol("NVDA")
        assert result is True
        assert "NVDA" in watchlist
        assert len(watchlist) == 4

    def test_add_symbol_duplicate(self, watchlist):
        """Test adding a duplicate symbol."""
        result = watchlist.add_symbol("AAPL")
        assert result is False
        assert len(watchlist) == 3

    def test_add_symbol_case_insensitive(self, watchlist):
        """Test that symbols are case-insensitive."""
        result = watchlist.add_symbol("aapl")
        assert result is False  # Already exists

    def test_add_symbols(self, watchlist):
        """Test adding multiple symbols."""
        results = watchlist.add_symbols(["NVDA", "TSLA", "AAPL"])
        assert results["NVDA"] is True
        assert results["TSLA"] is True
        assert results["AAPL"] is False

    def test_remove_symbol(self, watchlist):
        """Test removing a symbol."""
        result = watchlist.remove_symbol("AAPL")
        assert result is True
        assert "AAPL" not in watchlist
        assert len(watchlist) == 2

    def test_remove_symbol_not_present(self, watchlist):
        """Test removing a symbol not in the list."""
        result = watchlist.remove_symbol("NVDA")
        assert result is False
        assert len(watchlist) == 3

    def test_get_symbols(self, watchlist):
        """Test getting all symbols."""
        symbols = watchlist.get_symbols()
        assert isinstance(symbols, list)
        assert len(symbols) == 3
        assert symbols == sorted(symbols)  # Should be sorted

    def test_contains(self, watchlist):
        """Test contains method."""
        assert watchlist.contains("AAPL") is True
        assert watchlist.contains("NVDA") is False

    def test_in_operator(self, watchlist):
        """Test 'in' operator."""
        assert "AAPL" in watchlist
        assert "NVDA" not in watchlist

    def test_clear(self, watchlist):
        """Test clearing the watchlist."""
        watchlist.clear()
        assert len(watchlist) == 0

    def test_iter(self, watchlist):
        """Test iteration."""
        symbols = list(watchlist)
        assert len(symbols) == 3
        assert symbols == sorted(symbols)

    def test_save_and_load(self, watchlist, tmp_path):
        """Test saving and loading watchlist."""
        filepath = tmp_path / "watchlist.json"

        watchlist.save(str(filepath))
        assert filepath.exists()

        new_watchlist = Watchlist(symbols=[])
        new_watchlist.load(str(filepath))

        assert len(new_watchlist) == len(watchlist)
        assert new_watchlist.get_symbols() == watchlist.get_symbols()

    def test_from_file(self, watchlist, tmp_path):
        """Test creating watchlist from file."""
        filepath = tmp_path / "watchlist.json"
        watchlist.save(str(filepath))

        loaded = Watchlist.from_file(str(filepath))
        assert loaded.get_symbols() == watchlist.get_symbols()

    def test_to_dict(self, watchlist):
        """Test converting to dictionary."""
        d = watchlist.to_dict()
        assert "symbols" in d
        assert "count" in d
        assert d["count"] == 3

    def test_merge(self):
        """Test merging watchlists."""
        wl1 = Watchlist(symbols=["AAPL", "MSFT"])
        wl2 = Watchlist(symbols=["GOOGL", "MSFT"])

        wl1.merge(wl2)

        assert len(wl1) == 3
        assert "GOOGL" in wl1
        assert "AAPL" in wl1

    def test_difference(self):
        """Test finding difference between watchlists."""
        wl1 = Watchlist(symbols=["AAPL", "MSFT", "NVDA"])
        wl2 = Watchlist(symbols=["MSFT", "GOOGL"])

        diff = wl1.difference(wl2)
        assert sorted(diff) == ["AAPL", "NVDA"]

    def test_intersection(self):
        """Test finding intersection of watchlists."""
        wl1 = Watchlist(symbols=["AAPL", "MSFT", "NVDA"])
        wl2 = Watchlist(symbols=["MSFT", "NVDA", "GOOGL"])

        common = wl1.intersection(wl2)
        assert sorted(common) == ["MSFT", "NVDA"]

    def test_repr(self, watchlist):
        """Test string representation."""
        r = repr(watchlist)
        assert "Watchlist" in r
        assert "AAPL" in r

    def test_str(self, watchlist):
        """Test human-readable string."""
        s = str(watchlist)
        assert "3 symbols" in s

    def test_validate_symbol(self, watchlist):
        """Test validating a single symbol."""
        result = watchlist.validate_symbol("AAPL")

        assert isinstance(result, dict)
        assert result["symbol"] == "AAPL"
        assert result["valid"] is True

    def test_validate_symbol_invalid(self, watchlist):
        """Test validating an invalid symbol."""
        result = watchlist.validate_symbol("INVALID_XYZ_123")

        assert result["valid"] is False


class TestIntegration:
    """Integration tests for data module."""

    def test_imports_work(self):
        """Test that all imports work correctly."""
        from src.data import FinnhubNewsClient, PriceClient, Watchlist

        assert FinnhubNewsClient is not None
        assert PriceClient is not None
        assert Watchlist is not None

    def test_price_client_with_watchlist(self):
        """Test using price client with watchlist."""
        watchlist = Watchlist(symbols=["AAPL", "MSFT"])
        client = PriceClient()

        # Validate watchlist symbols
        valid = client.validate_symbols(watchlist.get_symbols())
        assert all(valid.values())

        # Get prices for watchlist
        data = client.get_multiple_symbols(watchlist.get_symbols(), period="5d")
        assert len(data) == 2
