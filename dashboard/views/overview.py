"""
Dashboard Overview Page.

Main dashboard view with:
- KPI metrics row
- Watchlist and sentiment chart
- Latest signals table
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd

from dashboard.data_loader import DashboardDataLoader
from dashboard.components.charts import (
    sentiment_time_series,
    candlestick_chart
)
from dashboard.components.tables import (
    kpi_row,
    watchlist_table,
    signals_table
)


def render(loader: DashboardDataLoader, selected_symbol: str) -> None:
    """
    Render the overview page.

    Args:
        loader: Data loader instance.
        selected_symbol: Currently selected symbol.
    """
    st.header("Market Overview")

    # Load data
    daily_sentiment = loader.load_daily_sentiment()
    signals = loader.load_sentiment_signals()
    prices = loader.load_prices(selected_symbol)

    # Calculate KPIs
    _render_kpis(daily_sentiment, signals)

    st.divider()

    # Two-column layout: Watchlist + Sentiment Chart
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Watchlist")
        _render_watchlist(loader)

    with col2:
        st.subheader(f"Sentiment: {selected_symbol}")
        _render_sentiment_chart(daily_sentiment, selected_symbol)

    st.divider()

    # Price chart
    st.subheader(f"Price: {selected_symbol}")
    _render_price_chart(prices, selected_symbol)

    st.divider()

    # Latest signals
    st.subheader("Latest Signals")
    _render_signals(signals)


def _render_kpis(sentiment_df: pd.DataFrame, signals_df: pd.DataFrame) -> None:
    """Render KPI metrics row."""
    # Calculate metrics
    if not sentiment_df.empty:
        avg_sentiment = sentiment_df['sentiment_score'].mean()
        total_articles = sentiment_df['article_count'].sum()
        symbols_count = sentiment_df['symbol'].nunique()

        # Calculate bullish percentage
        if 'bullish_ratio' in sentiment_df.columns:
            bullish_pct = sentiment_df['bullish_ratio'].mean()
        else:
            bullish_pct = (sentiment_df['sentiment_score'] > 0.1).mean()
    else:
        avg_sentiment = 0
        total_articles = 0
        symbols_count = 0
        bullish_pct = 0

    # Format metrics
    metrics = [
        {
            'label': 'Avg Sentiment',
            'value': f"{avg_sentiment:.3f}",
            'delta': None
        },
        {
            'label': 'Total Articles',
            'value': f"{int(total_articles):,}",
            'delta': None
        },
        {
            'label': 'Symbols Tracked',
            'value': str(symbols_count),
            'delta': None
        },
        {
            'label': 'Bullish %',
            'value': f"{bullish_pct:.1%}",
            'delta': None
        }
    ]

    kpi_row(metrics)


def _render_watchlist(loader: DashboardDataLoader) -> None:
    """Render watchlist table."""
    overview = loader.get_sentiment_overview()

    if overview.empty:
        # Fall back to symbols with latest prices
        symbols = loader.get_available_symbols()
        if symbols:
            latest_prices = loader.get_latest_prices()
            if not latest_prices.empty:
                watchlist_table(
                    latest_prices,
                    symbol_col='Symbol',
                    price_col='Close',
                    sentiment_col='sentiment_score'
                )
            else:
                st.info("No price data available")
        else:
            st.info("No symbols available")
    else:
        watchlist_table(
            overview,
            symbol_col='symbol',
            price_col='Close',
            sentiment_col='sentiment_score'
        )


def _render_sentiment_chart(df: pd.DataFrame, symbol: str) -> None:
    """Render sentiment time series chart."""
    symbol_data = df[df['symbol'] == symbol] if not df.empty else pd.DataFrame()

    if symbol_data.empty:
        st.info(f"No sentiment data for {symbol}")
        return

    fig = sentiment_time_series(
        symbol_data,
        date_col='date',
        score_col='sentiment_score',
        title='',
        show_zones=True,
        height=350
    )

    st.plotly_chart(fig, width='stretch')


def _render_price_chart(df: pd.DataFrame, symbol: str) -> None:
    """Render price candlestick chart."""
    if df.empty:
        st.info(f"No price data for {symbol}")
        return

    fig = candlestick_chart(
        df,
        date_col='Date',
        title='',
        height=400
    )

    st.plotly_chart(fig, width='stretch')


def _render_signals(df: pd.DataFrame) -> None:
    """Render latest signals table."""
    if df.empty:
        st.info("No signals available")
        return

    # Sort by date descending and show latest
    df_sorted = df.sort_values('date', ascending=False)

    signals_table(
        df_sorted,
        symbol_col='symbol',
        date_col='date',
        signal_col='signal',
        strength_col='signal_strength',
        max_rows=10
    )
