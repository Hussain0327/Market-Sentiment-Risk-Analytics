"""
Reusable table components for the dashboard.

Provides formatted data tables and KPI cards using Streamlit.
"""

from typing import Optional, Dict, Any, List

import pandas as pd
import streamlit as st


# =============================================================================
# Color Helpers
# =============================================================================

def _sentiment_color(value: float) -> str:
    """Get color based on sentiment value."""
    if value > 0.2:
        return "green"
    elif value < -0.2:
        return "red"
    return "gray"


def _signal_color(signal: str) -> str:
    """Get color based on signal direction."""
    signal = str(signal).lower()
    if signal in ['bullish', '1', 'long']:
        return "green"
    elif signal in ['bearish', '-1', 'short']:
        return "red"
    return "gray"


def _format_percent(value: float, decimals: int = 2) -> str:
    """Format value as percentage."""
    return f"{value * 100:.{decimals}f}%"


def _format_number(value: float, decimals: int = 2) -> str:
    """Format number with specified decimals."""
    if pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}"


# =============================================================================
# KPI Cards
# =============================================================================

def kpi_card(
    label: str,
    value: Any,
    delta: Optional[float] = None,
    delta_color: str = "normal"
) -> None:
    """
    Display a KPI metric card.

    Args:
        label: Metric label.
        value: Metric value.
        delta: Optional change value.
        delta_color: Color for delta ("normal", "inverse", or "off").
    """
    st.metric(
        label=label,
        value=value,
        delta=delta,
        delta_color=delta_color
    )


def kpi_row(metrics: List[Dict[str, Any]]) -> None:
    """
    Display a row of KPI cards.

    Args:
        metrics: List of dictionaries with 'label', 'value', and optionally 'delta'.
    """
    cols = st.columns(len(metrics))

    for col, metric in zip(cols, metrics):
        with col:
            kpi_card(
                label=metric['label'],
                value=metric['value'],
                delta=metric.get('delta'),
                delta_color=metric.get('delta_color', 'normal')
            )


# =============================================================================
# Watchlist Table
# =============================================================================

def watchlist_table(
    df: pd.DataFrame,
    symbol_col: str = 'symbol',
    price_col: str = 'Close',
    sentiment_col: str = 'sentiment_score',
    signal_col: Optional[str] = None
) -> None:
    """
    Display watchlist table with symbol, price, sentiment, and signal.

    Args:
        df: DataFrame with watchlist data.
        symbol_col: Name of symbol column.
        price_col: Name of price column.
        sentiment_col: Name of sentiment column.
        signal_col: Name of signal column (optional).
    """
    if df.empty:
        st.info("No watchlist data available")
        return

    # Create display dataframe
    display_df = df.copy()

    # Format columns
    if price_col in display_df.columns:
        display_df[price_col] = display_df[price_col].apply(
            lambda x: f"${x:.2f}" if pd.notna(x) else "N/A"
        )

    if sentiment_col in display_df.columns:
        display_df['Sentiment'] = display_df[sentiment_col].apply(
            lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
        )

    # Select columns to display
    cols_to_show = [symbol_col]
    if price_col in display_df.columns:
        cols_to_show.append(price_col)
    if 'Sentiment' in display_df.columns:
        cols_to_show.append('Sentiment')
    if signal_col and signal_col in display_df.columns:
        cols_to_show.append(signal_col)

    display_df = display_df[cols_to_show]

    # Rename columns for display
    display_df.columns = [c.replace('_', ' ').title() for c in display_df.columns]

    st.dataframe(display_df, width='stretch', hide_index=True)


# =============================================================================
# Signals Table
# =============================================================================

def signals_table(
    df: pd.DataFrame,
    symbol_col: str = 'symbol',
    date_col: str = 'date',
    signal_col: str = 'signal',
    strength_col: str = 'signal_strength',
    max_rows: int = 10
) -> None:
    """
    Display formatted trading signals table.

    Args:
        df: DataFrame with signal data.
        symbol_col: Name of symbol column.
        date_col: Name of date column.
        signal_col: Name of signal column.
        strength_col: Name of strength column.
        max_rows: Maximum number of rows to display.
    """
    if df.empty:
        st.info("No signals available")
        return

    display_df = df.head(max_rows).copy()

    # Format date
    if date_col in display_df.columns:
        display_df[date_col] = pd.to_datetime(display_df[date_col]).dt.strftime('%Y-%m-%d')

    # Format strength
    if strength_col in display_df.columns:
        display_df[strength_col] = display_df[strength_col].apply(
            lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
        )

    # Select columns
    cols_to_show = []
    for col in [symbol_col, date_col, signal_col, strength_col]:
        if col in display_df.columns:
            cols_to_show.append(col)

    display_df = display_df[cols_to_show]
    display_df.columns = [c.replace('_', ' ').title() for c in display_df.columns]

    st.dataframe(display_df, width='stretch', hide_index=True)


# =============================================================================
# Risk Summary Table
# =============================================================================

def risk_summary_table(
    var_95: float,
    var_99: float,
    cvar_95: Optional[float] = None,
    cvar_99: Optional[float] = None,
    volatility_21d: Optional[float] = None,
    volatility_63d: Optional[float] = None,
    max_drawdown: Optional[float] = None,
    current_drawdown: Optional[float] = None
) -> None:
    """
    Display risk metrics summary table.

    Args:
        var_95: 95% VaR.
        var_99: 99% VaR.
        cvar_95: 95% CVaR (optional).
        cvar_99: 99% CVaR (optional).
        volatility_21d: 21-day volatility (optional).
        volatility_63d: 63-day volatility (optional).
        max_drawdown: Maximum drawdown (optional).
        current_drawdown: Current drawdown (optional).
    """
    data = {
        'Metric': [],
        'Value': []
    }

    # VaR metrics
    data['Metric'].append('VaR 95%')
    data['Value'].append(_format_percent(abs(var_95)))

    data['Metric'].append('VaR 99%')
    data['Value'].append(_format_percent(abs(var_99)))

    if cvar_95 is not None:
        data['Metric'].append('CVaR 95%')
        data['Value'].append(_format_percent(abs(cvar_95)))

    if cvar_99 is not None:
        data['Metric'].append('CVaR 99%')
        data['Value'].append(_format_percent(abs(cvar_99)))

    # Volatility
    if volatility_21d is not None:
        data['Metric'].append('Volatility (21d)')
        data['Value'].append(_format_percent(volatility_21d))

    if volatility_63d is not None:
        data['Metric'].append('Volatility (63d)')
        data['Value'].append(_format_percent(volatility_63d))

    # Drawdown
    if max_drawdown is not None:
        data['Metric'].append('Max Drawdown')
        data['Value'].append(_format_percent(abs(max_drawdown)))

    if current_drawdown is not None:
        data['Metric'].append('Current Drawdown')
        data['Value'].append(_format_percent(abs(current_drawdown)))

    df = pd.DataFrame(data)
    st.dataframe(df, width='stretch', hide_index=True)


# =============================================================================
# Sentiment Summary Table
# =============================================================================

def sentiment_summary_table(
    df: pd.DataFrame,
    symbol_col: str = 'symbol',
    score_col: str = 'sentiment_score',
    confidence_col: str = 'sentiment_confidence',
    articles_col: str = 'article_count',
    bullish_col: str = 'bullish_ratio',
    bearish_col: str = 'bearish_ratio'
) -> None:
    """
    Display sentiment summary statistics table.

    Args:
        df: DataFrame with sentiment summary data.
        symbol_col: Symbol column name.
        score_col: Sentiment score column name.
        confidence_col: Confidence column name.
        articles_col: Article count column name.
        bullish_col: Bullish ratio column name.
        bearish_col: Bearish ratio column name.
    """
    if df.empty:
        st.info("No sentiment data available")
        return

    display_df = df.copy()

    # Format columns
    if score_col in display_df.columns:
        display_df[score_col] = display_df[score_col].apply(
            lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
        )

    if confidence_col in display_df.columns:
        display_df[confidence_col] = display_df[confidence_col].apply(
            lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
        )

    if bullish_col in display_df.columns:
        display_df[bullish_col] = display_df[bullish_col].apply(
            lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
        )

    if bearish_col in display_df.columns:
        display_df[bearish_col] = display_df[bearish_col].apply(
            lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
        )

    # Select and rename columns
    cols_to_show = []
    col_names = []

    if symbol_col in display_df.columns:
        cols_to_show.append(symbol_col)
        col_names.append('Symbol')

    if score_col in display_df.columns:
        cols_to_show.append(score_col)
        col_names.append('Sentiment')

    if confidence_col in display_df.columns:
        cols_to_show.append(confidence_col)
        col_names.append('Confidence')

    if articles_col in display_df.columns:
        cols_to_show.append(articles_col)
        col_names.append('Articles')

    if bullish_col in display_df.columns:
        cols_to_show.append(bullish_col)
        col_names.append('Bullish %')

    if bearish_col in display_df.columns:
        cols_to_show.append(bearish_col)
        col_names.append('Bearish %')

    display_df = display_df[cols_to_show]
    display_df.columns = col_names

    st.dataframe(display_df, width='stretch', hide_index=True)


# =============================================================================
# News Articles Table
# =============================================================================

def news_articles_table(
    df: pd.DataFrame,
    datetime_col: str = 'datetime',
    headline_col: str = 'headline',
    sentiment_col: str = 'sentiment_score',
    signal_col: str = 'sentiment_signal',
    max_rows: int = 20
) -> None:
    """
    Display news articles with sentiment.

    Args:
        df: DataFrame with news data.
        datetime_col: DateTime column name.
        headline_col: Headline column name.
        sentiment_col: Sentiment score column name.
        signal_col: Signal column name.
        max_rows: Maximum rows to display.
    """
    if df.empty:
        st.info("No news articles available")
        return

    display_df = df.head(max_rows).copy()

    # Format datetime
    if datetime_col in display_df.columns:
        display_df[datetime_col] = pd.to_datetime(display_df[datetime_col]).dt.strftime('%m/%d %H:%M')

    # Format sentiment
    if sentiment_col in display_df.columns:
        display_df[sentiment_col] = display_df[sentiment_col].apply(
            lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
        )

    # Truncate headlines
    if headline_col in display_df.columns:
        display_df[headline_col] = display_df[headline_col].apply(
            lambda x: (x[:80] + '...') if len(str(x)) > 80 else x
        )

    # Select columns
    cols_to_show = []
    for col in [datetime_col, headline_col, sentiment_col, signal_col]:
        if col in display_df.columns:
            cols_to_show.append(col)

    display_df = display_df[cols_to_show]
    display_df.columns = ['Time', 'Headline', 'Score', 'Signal'][:len(cols_to_show)]

    st.dataframe(display_df, width='stretch', hide_index=True)
