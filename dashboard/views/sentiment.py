"""
Dashboard Sentiment Analysis Page.

Detailed sentiment analysis with:
- Tab 1: Time series + gauge + summary stats
- Tab 2: Cross-symbol heatmap
- Tab 3: Article volume + recent articles
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
    sentiment_heatmap,
    bullish_bearish_gauge,
    article_volume_chart
)
from dashboard.components.tables import (
    kpi_row,
    sentiment_summary_table,
    news_articles_table
)


def render(loader: DashboardDataLoader, selected_symbol: str) -> None:
    """
    Render the sentiment analysis page.

    Args:
        loader: Data loader instance.
        selected_symbol: Currently selected symbol.
    """
    st.header("Sentiment Analysis")

    # Load data
    daily_sentiment = loader.load_daily_sentiment()
    news_sentiment = loader.load_news_sentiment()

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["Time Series", "Cross-Symbol", "News Volume"])

    with tab1:
        _render_time_series_tab(daily_sentiment, selected_symbol)

    with tab2:
        _render_heatmap_tab(daily_sentiment)

    with tab3:
        _render_news_tab(daily_sentiment, news_sentiment, selected_symbol)


def _render_time_series_tab(df: pd.DataFrame, symbol: str) -> None:
    """Render time series analysis tab."""
    symbol_data = df[df['symbol'] == symbol] if not df.empty else pd.DataFrame()

    if symbol_data.empty:
        st.info(f"No sentiment data available for {symbol}")
        return

    # KPIs for selected symbol
    latest = symbol_data.iloc[-1] if not symbol_data.empty else None

    if latest is not None:
        metrics = [
            {
                'label': 'Current Sentiment',
                'value': f"{latest['sentiment_score']:.3f}"
            },
            {
                'label': 'Confidence',
                'value': f"{latest['sentiment_confidence']:.1%}"
            },
            {
                'label': 'Articles Today',
                'value': str(int(latest['article_count']))
            },
            {
                'label': 'Bullish Ratio',
                'value': f"{latest['bullish_ratio']:.1%}"
            }
        ]
        kpi_row(metrics)

    st.divider()

    # Two columns: Chart + Gauge
    col1, col2 = st.columns([3, 1])

    with col1:
        fig = sentiment_time_series(
            symbol_data,
            date_col='date',
            score_col='sentiment_score',
            title=f'{symbol} Sentiment Over Time',
            show_zones=True,
            height=400
        )
        st.plotly_chart(fig, width='stretch')

    with col2:
        if latest is not None:
            fig = bullish_bearish_gauge(
                float(latest['sentiment_score']),
                title='Current',
                height=200
            )
            st.plotly_chart(fig, width='stretch')

            # Summary stats
            st.markdown("**Summary Stats**")
            st.write(f"Avg: {symbol_data['sentiment_score'].mean():.3f}")
            st.write(f"Std: {symbol_data['sentiment_score'].std():.3f}")
            st.write(f"Min: {symbol_data['sentiment_score'].min():.3f}")
            st.write(f"Max: {symbol_data['sentiment_score'].max():.3f}")


def _render_heatmap_tab(df: pd.DataFrame) -> None:
    """Render cross-symbol heatmap tab."""
    st.subheader("Cross-Symbol Sentiment Comparison")

    if df.empty:
        st.info("No sentiment data available")
        return

    # Limit to last N days for readability
    n_days = st.slider("Days to Display", min_value=5, max_value=30, value=14)

    df['date'] = pd.to_datetime(df['date'])
    cutoff = df['date'].max() - pd.Timedelta(days=n_days)
    filtered = df[df['date'] >= cutoff]

    fig = sentiment_heatmap(
        filtered,
        date_col='date',
        symbol_col='symbol',
        score_col='sentiment_score',
        title='Sentiment Heatmap',
        height=450
    )

    st.plotly_chart(fig, width='stretch')

    st.divider()

    # Summary table
    st.subheader("Symbol Summary")
    sentiment_summary_table(
        df.groupby('symbol').agg({
            'sentiment_score': 'mean',
            'sentiment_confidence': 'mean',
            'article_count': 'sum',
            'bullish_ratio': 'mean',
            'bearish_ratio': 'mean'
        }).reset_index(),
        symbol_col='symbol'
    )


def _render_news_tab(
    daily_df: pd.DataFrame,
    news_df: pd.DataFrame,
    symbol: str
) -> None:
    """Render news volume and articles tab."""
    st.subheader("News Volume")

    symbol_daily = daily_df[daily_df['symbol'] == symbol] if not daily_df.empty else pd.DataFrame()

    if not symbol_daily.empty:
        fig = article_volume_chart(
            symbol_daily,
            date_col='date',
            count_col='article_count',
            title=f'{symbol} Daily Article Count',
            height=300
        )
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("No volume data available")

    st.divider()

    # Recent articles
    st.subheader("Recent Articles")

    symbol_news = news_df[news_df['symbol'] == symbol] if not news_df.empty else pd.DataFrame()

    if not symbol_news.empty:
        # Sort by datetime descending
        symbol_news = symbol_news.sort_values('datetime', ascending=False)

        news_articles_table(
            symbol_news,
            datetime_col='datetime',
            headline_col='headline',
            sentiment_col='sentiment_score',
            signal_col='sentiment_signal',
            max_rows=25
        )
    else:
        st.info(f"No news articles available for {symbol}")
