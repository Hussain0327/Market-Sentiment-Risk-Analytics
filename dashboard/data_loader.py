"""
Dashboard Data Loader.

Provides a unified data access layer for the dashboard.
Supports multiple data sources:
- CSV files (default)
- Database storage
- API backend (for production/separated deployment)
- Remote storage (GitHub Releases for cloud deployment)
"""

import os
import tempfile
import tarfile
import logging
from pathlib import Path
from typing import Optional, List, Union

import pandas as pd
import requests


logger = logging.getLogger(__name__)

# Define paths directly to avoid config dependency issues
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Environment variable for API mode
USE_API_ENV = "DASHBOARD_USE_API"
API_URL_ENV = "DASHBOARD_API_URL"
REMOTE_DATA_ENV = "REMOTE_DATA"


class RemoteDataLoader:
    """Handles downloading and caching data from GitHub Releases."""

    DEFAULT_REPO = "Hussain0327/Market-Sentiment-Risk-Analytics"
    DATA_ARCHIVE_NAME = "data.tar.gz"

    def __init__(
        self,
        repo: Optional[str] = None,
        github_token: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ):
        self.repo = repo or os.environ.get("GITHUB_REPO", self.DEFAULT_REPO)
        self.github_token = github_token or os.environ.get("GITHUB_TOKEN")

        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path(tempfile.gettempdir()) / "market_sentiment_data"

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._raw_dir = self.cache_dir / "raw"
        self._processed_dir = self.cache_dir / "processed"
        self._downloaded = False

    def get_raw_dir(self) -> Path:
        """Get path to raw data directory, downloading if needed."""
        self._ensure_downloaded()
        return self._raw_dir

    def get_processed_dir(self) -> Path:
        """Get path to processed data directory, downloading if needed."""
        self._ensure_downloaded()
        return self._processed_dir

    def _ensure_downloaded(self) -> None:
        """Download data if not already cached."""
        if self._downloaded:
            return

        # Check if we have cached data
        if self._raw_dir.exists() or self._processed_dir.exists():
            self._downloaded = True
            return

        # Try to download from GitHub
        try:
            self._download_data()
            self._downloaded = True
        except Exception as e:
            logger.warning(f"Failed to download remote data: {e}")

    def _download_data(self) -> None:
        """Download data archive from GitHub Releases."""
        url = f"https://api.github.com/repos/{self.repo}/releases/latest"
        headers = {"Accept": "application/vnd.github.v3+json"}
        if self.github_token:
            headers["Authorization"] = f"token {self.github_token}"

        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 404:
            logger.info("No releases found")
            return

        response.raise_for_status()
        release = response.json()

        # Find data asset
        asset = None
        for a in release.get("assets", []):
            if a["name"] == self.DATA_ARCHIVE_NAME:
                asset = a
                break

        if not asset:
            logger.info("No data archive in release")
            return

        # Download and extract
        download_url = asset["url"]
        headers["Accept"] = "application/octet-stream"

        response = requests.get(download_url, headers=headers, stream=True, timeout=60)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".tar.gz") as tmp:
            for chunk in response.iter_content(chunk_size=8192):
                tmp.write(chunk)
            tmp_path = tmp.name

        # Extract
        self._raw_dir.mkdir(parents=True, exist_ok=True)
        self._processed_dir.mkdir(parents=True, exist_ok=True)

        with tarfile.open(tmp_path, "r:gz") as tar:
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    continue
                tar.extract(member, path=self.cache_dir)

        os.unlink(tmp_path)
        logger.info("Downloaded remote data successfully")


class DashboardDataLoader:
    """
    Data loader for the dashboard with CSV/DB/API abstraction.

    Supports four data source modes:
    1. CSV files (default) - loads from local data directory
    2. Database - uses SQLite/PostgreSQL for structured queries
    3. API - fetches data from FastAPI backend over HTTP
    4. Remote - downloads from GitHub Releases for cloud deployment

    Example:
        >>> # Local mode (default)
        >>> loader = DashboardDataLoader()
        >>> prices = loader.load_prices('AAPL')

        >>> # API mode
        >>> loader = DashboardDataLoader(use_api=True)
        >>> sentiment = loader.load_daily_sentiment('AAPL')
    """

    def __init__(
        self,
        use_db: bool = False,
        db_path: Optional[str] = None,
        use_api: Optional[bool] = None,
        api_url: Optional[str] = None,
        use_remote: Optional[bool] = None,
    ):
        """
        Initialize the data loader.

        Args:
            use_db: If True, use database instead of CSV files.
            db_path: Path to database file (only used if use_db=True).
            use_api: If True, use API backend. If None, checks environment variable.
            api_url: API server URL. If None, checks environment variable or uses default.
            use_remote: If True, download data from GitHub Releases. Auto-detected in cloud.
        """
        # Determine API mode from environment or parameter
        if use_api is None:
            use_api = os.environ.get(USE_API_ENV, "").lower() in ("1", "true", "yes")

        # Determine remote mode from environment or parameter
        if use_remote is None:
            use_remote = (
                os.environ.get(REMOTE_DATA_ENV, "").lower() in ("1", "true", "yes") or
                os.environ.get("STREAMLIT_SHARING") == "1" or
                os.environ.get("MODEL_STORAGE") in ("github", "s3")
            )

        self.use_api = use_api
        self.use_db = use_db and not use_api  # DB mode disabled when using API
        self.use_remote = use_remote and not use_api  # Remote disabled when using API
        self.data_dir = DATA_DIR
        self.raw_dir = RAW_DATA_DIR
        self.processed_dir = PROCESSED_DATA_DIR

        # Initialize remote loader if in remote mode
        self._remote_loader: Optional[RemoteDataLoader] = None
        if self.use_remote:
            self._remote_loader = RemoteDataLoader()

        # Initialize API client if in API mode
        if self.use_api:
            from dashboard.api_client import APIClient
            if api_url is None:
                api_url = os.environ.get(API_URL_ENV, "http://localhost:8000")
            self._api_client = APIClient(base_url=api_url)
            self.db = None
        elif self.use_db:
            from src.db import DatabaseManager
            self.db = DatabaseManager(db_path)
            self.db.init_db()
            self._api_client = None
        else:
            self.db = None
            self._api_client = None

    def _get_raw_dir(self) -> Path:
        """Get the raw data directory, using remote if configured."""
        if self._remote_loader:
            try:
                return self._remote_loader.get_raw_dir()
            except Exception:
                pass
        return self.raw_dir

    def _get_processed_dir(self) -> Path:
        """Get the processed data directory, using remote if configured."""
        if self._remote_loader:
            try:
                return self._remote_loader.get_processed_dir()
            except Exception:
                pass
        return self.processed_dir

    def load_prices(self, symbol: str) -> pd.DataFrame:
        """
        Load price data for a symbol.

        Args:
            symbol: Stock symbol (e.g., 'AAPL').

        Returns:
            DataFrame with OHLCV price data.
        """
        # API mode
        if self.use_api and self._api_client is not None:
            try:
                df = self._api_client.get_price_history_df(symbol)
                if not df.empty:
                    df = df.reset_index()
                    # Standardize column names
                    df.columns = [c.title() for c in df.columns]
                return df
            except Exception:
                return pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])

        # Database mode
        if self.use_db and self.db is not None:
            from src.db import get_symbol_by_ticker, get_prices_df
            with self.db.session() as session:
                sym = get_symbol_by_ticker(session, symbol)
                if sym is None:
                    return pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume'])
                return get_prices_df(session, sym.id)

        # Load from CSV (with remote fallback)
        raw_dir = self._get_raw_dir()
        csv_path = raw_dir / f"{symbol}_prices.csv"
        if not csv_path.exists():
            return pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])

        df = pd.read_csv(csv_path)
        # Handle mixed timezone datetime parsing
        df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)
        return df.sort_values('Date').reset_index(drop=True)

    def load_all_prices(self) -> pd.DataFrame:
        """
        Load price data for all available symbols.

        Returns:
            DataFrame with OHLCV data and Symbol column.
        """
        symbols = self.get_available_symbols()
        dfs = []

        for symbol in symbols:
            df = self.load_prices(symbol)
            if not df.empty:
                if 'Symbol' not in df.columns:
                    df['Symbol'] = symbol
                dfs.append(df)

        if not dfs:
            return pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Symbol'])

        return pd.concat(dfs, ignore_index=True)

    def load_daily_sentiment(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """
        Load daily aggregated sentiment data.

        Args:
            symbol: Optional symbol filter.

        Returns:
            DataFrame with daily sentiment scores.
        """
        # API mode
        if self.use_api and self._api_client is not None:
            try:
                if symbol:
                    df = self._api_client.get_sentiment_history_df(symbol)
                    if not df.empty:
                        df['symbol'] = symbol
                    return df
                else:
                    # Get all symbols and combine
                    symbols = self._api_client.list_sentiment_symbols()
                    dfs = []
                    for sym in symbols:
                        try:
                            sym_df = self._api_client.get_sentiment_history_df(sym)
                            if not sym_df.empty:
                                sym_df['symbol'] = sym
                                dfs.append(sym_df)
                        except Exception:
                            continue
                    if dfs:
                        return pd.concat(dfs, ignore_index=True)
                    return pd.DataFrame()
            except Exception:
                return pd.DataFrame(columns=[
                    'symbol', 'date', 'sentiment_score', 'sentiment_confidence',
                    'article_count', 'bullish_ratio', 'bearish_ratio'
                ])

        # Database mode
        if self.use_db and self.db is not None:
            from src.db import get_symbol_by_ticker, get_daily_sentiment_df
            with self.db.session() as session:
                if symbol:
                    sym = get_symbol_by_ticker(session, symbol)
                    if sym is None:
                        return pd.DataFrame()
                    return get_daily_sentiment_df(session, sym.id)
                # Get all sentiments - would need a different query
                return pd.DataFrame()

        # Load from CSV (with remote fallback)
        processed_dir = self._get_processed_dir()
        csv_path = processed_dir / "daily_sentiment.csv"
        if not csv_path.exists():
            return pd.DataFrame(columns=[
                'symbol', 'date', 'sentiment_score', 'sentiment_confidence',
                'article_count', 'bullish_ratio', 'bearish_ratio'
            ])

        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])

        if symbol:
            df = df[df['symbol'] == symbol]

        return df.sort_values(['symbol', 'date']).reset_index(drop=True)

    def load_news_sentiment(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """
        Load article-level sentiment data.

        Args:
            symbol: Optional symbol filter.

        Returns:
            DataFrame with per-article sentiment.
        """
        # News sentiment is always from CSV (with remote fallback)
        processed_dir = self._get_processed_dir()
        csv_path = processed_dir / "news_sentiment.csv"
        if not csv_path.exists():
            return pd.DataFrame(columns=[
                'datetime', 'headline', 'symbol', 'sentiment_score',
                'sentiment_confidence', 'sentiment_signal'
            ])

        df = pd.read_csv(csv_path)
        df['datetime'] = pd.to_datetime(df['datetime'])

        if symbol:
            df = df[df['symbol'] == symbol]

        return df.sort_values('datetime', ascending=False).reset_index(drop=True)

    def load_sentiment_signals(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """
        Load sentiment trading signals.

        Args:
            symbol: Optional symbol filter.

        Returns:
            DataFrame with sentiment-based signals.
        """
        # API mode
        if self.use_api and self._api_client is not None:
            try:
                if symbol:
                    df = self._api_client.get_signals_df(symbol)
                    if not df.empty:
                        df['symbol'] = symbol
                    return df
                else:
                    # Get all symbols and combine
                    symbols = self._api_client.list_signal_symbols()
                    dfs = []
                    for sym in symbols:
                        try:
                            sym_df = self._api_client.get_signals_df(sym)
                            if not sym_df.empty:
                                sym_df['symbol'] = sym
                                dfs.append(sym_df)
                        except Exception:
                            continue
                    if dfs:
                        return pd.concat(dfs, ignore_index=True)
                    return pd.DataFrame()
            except Exception:
                return pd.DataFrame(columns=[
                    'symbol', 'date', 'sentiment_score', 'signal', 'signal_strength'
                ])

        # CSV mode (with remote fallback)
        processed_dir = self._get_processed_dir()
        csv_path = processed_dir / "sentiment_signals.csv"
        if not csv_path.exists():
            return pd.DataFrame(columns=[
                'symbol', 'date', 'sentiment_score', 'signal', 'signal_strength'
            ])

        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])

        if symbol:
            df = df[df['symbol'] == symbol]

        return df.sort_values(['symbol', 'date']).reset_index(drop=True)

    def get_available_symbols(self) -> List[str]:
        """
        Get list of available symbols.

        Returns:
            List of stock symbols with data.
        """
        # API mode
        if self.use_api and self._api_client is not None:
            try:
                return self._api_client.get_symbols()
            except Exception:
                return []

        # Database mode
        if self.use_db and self.db is not None:
            from src.db import get_all_symbols
            with self.db.session() as session:
                symbols = get_all_symbols(session)
                return [s.ticker for s in symbols]

        # Detect from CSV files (with remote fallback)
        symbols = set()
        raw_dir = self._get_raw_dir()
        processed_dir = self._get_processed_dir()

        # Check raw price files
        if raw_dir.exists():
            for f in raw_dir.glob("*_prices.csv"):
                symbol = f.stem.replace("_prices", "")
                symbols.add(symbol)

        # Check daily sentiment
        daily_sent = processed_dir / "daily_sentiment.csv"
        if daily_sent.exists():
            df = pd.read_csv(daily_sent)
            if 'symbol' in df.columns:
                symbols.update(df['symbol'].unique())

        return sorted(list(symbols))

    def get_latest_prices(self) -> pd.DataFrame:
        """
        Get latest price for each symbol.

        Returns:
            DataFrame with latest prices per symbol.
        """
        all_prices = self.load_all_prices()

        if all_prices.empty:
            return pd.DataFrame(columns=['Symbol', 'Date', 'Close', 'Volume'])

        # Rename date column if needed
        date_col = 'Date' if 'Date' in all_prices.columns else 'date'

        # Get latest row per symbol
        idx = all_prices.groupby('Symbol')[date_col].idxmax()
        latest = all_prices.loc[idx].reset_index(drop=True)

        return latest

    def get_sentiment_overview(self) -> pd.DataFrame:
        """
        Get sentiment overview for dashboard.

        Returns:
            DataFrame with symbol, latest sentiment, and stats.
        """
        sentiment = self.load_daily_sentiment()

        if sentiment.empty:
            return pd.DataFrame(columns=[
                'symbol', 'latest_date', 'sentiment_score',
                'article_count', 'bullish_ratio', 'bearish_ratio'
            ])

        # Get latest row per symbol
        idx = sentiment.groupby('symbol')['date'].idxmax()
        latest = sentiment.loc[idx].copy()

        # Add average stats
        avg_stats = sentiment.groupby('symbol').agg({
            'sentiment_score': 'mean',
            'article_count': 'sum'
        }).rename(columns={
            'sentiment_score': 'avg_sentiment',
            'article_count': 'total_articles'
        })

        latest = latest.merge(avg_stats, left_on='symbol', right_index=True)

        return latest.reset_index(drop=True)

    def get_signal_summary(self) -> pd.DataFrame:
        """
        Get summary of trading signals.

        Returns:
            DataFrame with signal counts and latest signals.
        """
        signals = self.load_sentiment_signals()

        if signals.empty:
            return pd.DataFrame(columns=['symbol', 'bullish_count', 'bearish_count', 'latest_signal'])

        # Count signals per symbol
        signal_counts = signals.groupby(['symbol', 'signal']).size().unstack(fill_value=0)

        # Get latest signal per symbol
        idx = signals.groupby('symbol')['date'].idxmax()
        latest = signals.loc[idx][['symbol', 'signal', 'signal_strength']].rename(
            columns={'signal': 'latest_signal', 'signal_strength': 'latest_strength'}
        )

        # Merge counts with latest
        summary = latest.merge(
            signal_counts.reset_index(),
            on='symbol',
            how='left'
        )

        return summary.reset_index(drop=True)
