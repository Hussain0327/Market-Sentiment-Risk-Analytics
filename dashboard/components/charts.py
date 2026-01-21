"""
Reusable Plotly chart components for the dashboard.

Provides chart functions for:
- Sentiment analysis visualization
- Risk metrics display
- Price and signal charts
"""

from typing import Optional, List, Dict, Any

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# =============================================================================
# Color Schemes
# =============================================================================

COLORS = {
    'bullish': '#00C853',      # Green
    'bearish': '#FF1744',      # Red
    'neutral': '#9E9E9E',      # Gray
    'primary': '#2196F3',      # Blue
    'secondary': '#673AB7',    # Purple
    'warning': '#FF9800',      # Orange
    'background': '#FAFAFA',
    'text': '#212121',
}


# =============================================================================
# Sentiment Charts
# =============================================================================

def sentiment_time_series(
    df: pd.DataFrame,
    date_col: str = 'date',
    score_col: str = 'sentiment_score',
    title: str = 'Sentiment Over Time',
    show_zones: bool = True,
    height: int = 400
) -> go.Figure:
    """
    Create sentiment time series chart with bullish/bearish zones.

    Args:
        df: DataFrame with date and sentiment columns.
        date_col: Name of date column.
        score_col: Name of sentiment score column.
        title: Chart title.
        show_zones: Whether to show bullish/bearish colored zones.
        height: Chart height in pixels.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()

    if df.empty:
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(title=title, height=height)
        return fig

    # Add background zones
    if show_zones:
        fig.add_hrect(
            y0=0, y1=1,
            fillcolor=COLORS['bullish'], opacity=0.1,
            layer="below", line_width=0
        )
        fig.add_hrect(
            y0=-1, y1=0,
            fillcolor=COLORS['bearish'], opacity=0.1,
            layer="below", line_width=0
        )

    # Add sentiment line
    fig.add_trace(go.Scatter(
        x=df[date_col],
        y=df[score_col],
        mode='lines+markers',
        name='Sentiment',
        line=dict(color=COLORS['primary'], width=2),
        marker=dict(size=6)
    ))

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Sentiment Score",
        yaxis=dict(range=[-1.1, 1.1]),
        height=height,
        hovermode='x unified',
        template='plotly_white'
    )

    return fig


def sentiment_heatmap(
    df: pd.DataFrame,
    date_col: str = 'date',
    symbol_col: str = 'symbol',
    score_col: str = 'sentiment_score',
    title: str = 'Sentiment Heatmap',
    height: int = 400
) -> go.Figure:
    """
    Create symbol x date sentiment heatmap.

    Args:
        df: DataFrame with date, symbol, and sentiment columns.
        date_col: Name of date column.
        symbol_col: Name of symbol column.
        score_col: Name of sentiment score column.
        title: Chart title.
        height: Chart height.

    Returns:
        Plotly Figure object.
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(title=title, height=height)
        return fig

    # Pivot data for heatmap
    pivot = df.pivot_table(
        index=symbol_col,
        columns=date_col,
        values=score_col,
        aggfunc='mean'
    )

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale=[
            [0, COLORS['bearish']],
            [0.5, 'white'],
            [1, COLORS['bullish']]
        ],
        zmid=0,
        colorbar=dict(title='Sentiment')
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Symbol",
        height=height,
        template='plotly_white'
    )

    return fig


def bullish_bearish_gauge(
    value: float,
    title: str = 'Sentiment',
    height: int = 250
) -> go.Figure:
    """
    Create a gauge indicator for sentiment.

    Args:
        value: Sentiment value [-1, 1].
        title: Gauge title.
        height: Chart height.

    Returns:
        Plotly Figure object.
    """
    # Determine color based on value
    if value > 0.2:
        color = COLORS['bullish']
    elif value < -0.2:
        color = COLORS['bearish']
    else:
        color = COLORS['neutral']

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        gauge={
            'axis': {'range': [-1, 1], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [-1, -0.2], 'color': 'rgba(255,23,68,0.3)'},
                {'range': [-0.2, 0.2], 'color': 'rgba(158,158,158,0.3)'},
                {'range': [0.2, 1], 'color': 'rgba(0,200,83,0.3)'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))

    fig.update_layout(height=height, template='plotly_white')
    return fig


def article_volume_chart(
    df: pd.DataFrame,
    date_col: str = 'date',
    count_col: str = 'article_count',
    title: str = 'News Volume',
    height: int = 300
) -> go.Figure:
    """
    Create bar chart of article counts over time.

    Args:
        df: DataFrame with date and count columns.
        date_col: Name of date column.
        count_col: Name of article count column.
        title: Chart title.
        height: Chart height.

    Returns:
        Plotly Figure object.
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(title=title, height=height)
        return fig

    fig = go.Figure(go.Bar(
        x=df[date_col],
        y=df[count_col],
        marker_color=COLORS['primary'],
        name='Articles'
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Article Count",
        height=height,
        template='plotly_white'
    )

    return fig


# =============================================================================
# Risk Charts
# =============================================================================

def var_bar_chart(
    var_95: float,
    var_99: float,
    cvar_95: Optional[float] = None,
    cvar_99: Optional[float] = None,
    title: str = 'Value at Risk',
    height: int = 350
) -> go.Figure:
    """
    Create bar chart comparing VaR metrics.

    Args:
        var_95: 95% VaR value.
        var_99: 99% VaR value.
        cvar_95: 95% CVaR (optional).
        cvar_99: 99% CVaR (optional).
        title: Chart title.
        height: Chart height.

    Returns:
        Plotly Figure object.
    """
    categories = ['VaR 95%', 'VaR 99%']
    values = [abs(var_95), abs(var_99)]
    colors = [COLORS['warning'], COLORS['bearish']]

    if cvar_95 is not None:
        categories.append('CVaR 95%')
        values.append(abs(cvar_95))
        colors.append(COLORS['warning'])

    if cvar_99 is not None:
        categories.append('CVaR 99%')
        values.append(abs(cvar_99))
        colors.append(COLORS['bearish'])

    fig = go.Figure(go.Bar(
        x=categories,
        y=values,
        marker_color=colors,
        text=[f'{v:.2%}' for v in values],
        textposition='outside'
    ))

    fig.update_layout(
        title=title,
        yaxis_title="Loss Percentage",
        yaxis=dict(tickformat='.1%'),
        height=height,
        template='plotly_white'
    )

    return fig


def volatility_term_structure(
    volatilities: Dict[str, float],
    title: str = 'Volatility Term Structure',
    height: int = 350
) -> go.Figure:
    """
    Create line chart showing volatility across different windows.

    Args:
        volatilities: Dictionary mapping window name to volatility.
        title: Chart title.
        height: Chart height.

    Returns:
        Plotly Figure object.
    """
    windows = list(volatilities.keys())
    values = list(volatilities.values())

    fig = go.Figure(go.Scatter(
        x=windows,
        y=values,
        mode='lines+markers',
        line=dict(color=COLORS['primary'], width=3),
        marker=dict(size=10),
        text=[f'{v:.1%}' for v in values],
        textposition='top center'
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Window",
        yaxis_title="Annualized Volatility",
        yaxis=dict(tickformat='.1%'),
        height=height,
        template='plotly_white'
    )

    return fig


def drawdown_underwater_chart(
    prices: pd.Series,
    title: str = 'Underwater Chart (Drawdown)',
    height: int = 350
) -> go.Figure:
    """
    Create underwater (drawdown) chart from price series.

    Args:
        prices: Price series.
        title: Chart title.
        height: Chart height.

    Returns:
        Plotly Figure object.
    """
    if prices.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(title=title, height=height)
        return fig

    # Calculate drawdown
    cummax = prices.cummax()
    drawdown = (prices - cummax) / cummax

    fig = go.Figure()

    # Fill area
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values,
        fill='tozeroy',
        fillcolor='rgba(255,23,68,0.3)',
        line=dict(color=COLORS['bearish'], width=1),
        name='Drawdown'
    ))

    # Zero line
    fig.add_hline(y=0, line_dash="solid", line_color="gray")

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Drawdown",
        yaxis=dict(tickformat='.1%'),
        height=height,
        template='plotly_white'
    )

    return fig


# =============================================================================
# Signal Charts
# =============================================================================

def feature_importance_chart(
    importance: Dict[str, float],
    top_n: int = 15,
    title: str = 'Feature Importance',
    height: int = 400
) -> go.Figure:
    """
    Create horizontal bar chart of feature importance.

    Args:
        importance: Dictionary mapping feature names to importance values.
        top_n: Number of top features to show.
        title: Chart title.
        height: Chart height.

    Returns:
        Plotly Figure object.
    """
    # Sort by importance
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features = [f[0] for f in sorted_features]
    values = [f[1] for f in sorted_features]

    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation='h',
        marker_color=COLORS['primary']
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Importance",
        yaxis=dict(autorange="reversed"),
        height=height,
        template='plotly_white'
    )

    return fig


def signal_strength_chart(
    df: pd.DataFrame,
    date_col: str = 'date',
    strength_col: str = 'signal_strength',
    direction_col: str = 'signal',
    title: str = 'Signal Strength Over Time',
    height: int = 400
) -> go.Figure:
    """
    Create scatter chart of signal strength with direction coloring.

    Args:
        df: DataFrame with signal data.
        date_col: Name of date column.
        strength_col: Name of signal strength column.
        direction_col: Name of direction/signal column.
        title: Chart title.
        height: Chart height.

    Returns:
        Plotly Figure object.
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(title=title, height=height)
        return fig

    # Map directions to colors
    color_map = {
        'bullish': COLORS['bullish'],
        'bearish': COLORS['bearish'],
        'neutral': COLORS['neutral'],
        1: COLORS['bullish'],
        -1: COLORS['bearish'],
        0: COLORS['neutral'],
    }

    colors = df[direction_col].map(lambda x: color_map.get(x, COLORS['neutral']))

    fig = go.Figure(go.Scatter(
        x=df[date_col],
        y=df[strength_col],
        mode='markers',
        marker=dict(
            size=10,
            color=colors,
            opacity=0.7
        ),
        text=df[direction_col],
        hovertemplate='Date: %{x}<br>Strength: %{y:.2f}<br>Signal: %{text}<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Signal Strength",
        height=height,
        template='plotly_white'
    )

    return fig


# =============================================================================
# Price Charts
# =============================================================================

def candlestick_chart(
    df: pd.DataFrame,
    date_col: str = 'Date',
    open_col: str = 'Open',
    high_col: str = 'High',
    low_col: str = 'Low',
    close_col: str = 'Close',
    volume_col: Optional[str] = 'Volume',
    title: str = 'Price Chart',
    height: int = 500
) -> go.Figure:
    """
    Create OHLC candlestick chart with optional volume.

    Args:
        df: DataFrame with OHLCV data.
        date_col: Name of date column.
        open_col: Name of open price column.
        high_col: Name of high price column.
        low_col: Name of low price column.
        close_col: Name of close price column.
        volume_col: Name of volume column (optional).
        title: Chart title.
        height: Chart height.

    Returns:
        Plotly Figure object.
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(title=title, height=height)
        return fig

    if volume_col and volume_col in df.columns:
        # Create subplots with volume
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3]
        )

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df[date_col],
            open=df[open_col],
            high=df[high_col],
            low=df[low_col],
            close=df[close_col],
            name='OHLC'
        ), row=1, col=1)

        # Volume bars
        colors = ['green' if c >= o else 'red'
                  for o, c in zip(df[open_col], df[close_col])]
        fig.add_trace(go.Bar(
            x=df[date_col],
            y=df[volume_col],
            marker_color=colors,
            name='Volume',
            opacity=0.5
        ), row=2, col=1)

        fig.update_layout(
            title=title,
            height=height,
            xaxis_rangeslider_visible=False,
            template='plotly_white'
        )
    else:
        # Just candlestick
        fig = go.Figure(go.Candlestick(
            x=df[date_col],
            open=df[open_col],
            high=df[high_col],
            low=df[low_col],
            close=df[close_col],
            name='OHLC'
        ))

        fig.update_layout(
            title=title,
            height=height,
            xaxis_rangeslider_visible=False,
            template='plotly_white'
        )

    return fig


def price_with_signals(
    price_df: pd.DataFrame,
    signal_df: pd.DataFrame,
    price_date_col: str = 'Date',
    price_col: str = 'Close',
    signal_date_col: str = 'date',
    signal_col: str = 'signal',
    title: str = 'Price with Signals',
    height: int = 450
) -> go.Figure:
    """
    Create price chart with buy/sell signal markers.

    Args:
        price_df: DataFrame with price data.
        signal_df: DataFrame with signal data.
        price_date_col: Date column in price DataFrame.
        price_col: Price column name.
        signal_date_col: Date column in signal DataFrame.
        signal_col: Signal column name.
        title: Chart title.
        height: Chart height.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()

    # Price line
    fig.add_trace(go.Scatter(
        x=price_df[price_date_col],
        y=price_df[price_col],
        mode='lines',
        name='Price',
        line=dict(color=COLORS['primary'], width=2)
    ))

    if not signal_df.empty:
        # Merge signals with prices to get y-values
        signal_df = signal_df.copy()
        signal_df[signal_date_col] = pd.to_datetime(signal_df[signal_date_col])

        # Bullish signals
        bullish = signal_df[signal_df[signal_col].isin(['bullish', 1])]
        if not bullish.empty:
            # Find corresponding prices
            merged = bullish.merge(
                price_df[[price_date_col, price_col]].rename(columns={price_date_col: signal_date_col}),
                on=signal_date_col,
                how='left'
            )
            fig.add_trace(go.Scatter(
                x=merged[signal_date_col],
                y=merged[price_col],
                mode='markers',
                name='Bullish',
                marker=dict(
                    symbol='triangle-up',
                    size=12,
                    color=COLORS['bullish']
                )
            ))

        # Bearish signals
        bearish = signal_df[signal_df[signal_col].isin(['bearish', -1])]
        if not bearish.empty:
            merged = bearish.merge(
                price_df[[price_date_col, price_col]].rename(columns={price_date_col: signal_date_col}),
                on=signal_date_col,
                how='left'
            )
            fig.add_trace(go.Scatter(
                x=merged[signal_date_col],
                y=merged[price_col],
                mode='markers',
                name='Bearish',
                marker=dict(
                    symbol='triangle-down',
                    size=12,
                    color=COLORS['bearish']
                )
            ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        height=height,
        template='plotly_white',
        hovermode='x unified'
    )

    return fig
