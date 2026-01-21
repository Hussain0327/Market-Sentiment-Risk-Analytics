"""
Dashboard reusable components.

Provides:
- Chart components (Plotly visualizations)
- Table components
- Common UI elements
"""

from .charts import (
    sentiment_time_series,
    sentiment_heatmap,
    bullish_bearish_gauge,
    article_volume_chart,
    var_bar_chart,
    volatility_term_structure,
    drawdown_underwater_chart,
    feature_importance_chart,
    signal_strength_chart,
    candlestick_chart,
    price_with_signals,
    COLORS,
)

from .tables import (
    kpi_card,
    kpi_row,
    watchlist_table,
    signals_table,
    risk_summary_table,
    sentiment_summary_table,
    news_articles_table,
)

__all__ = [
    # Charts
    'sentiment_time_series',
    'sentiment_heatmap',
    'bullish_bearish_gauge',
    'article_volume_chart',
    'var_bar_chart',
    'volatility_term_structure',
    'drawdown_underwater_chart',
    'feature_importance_chart',
    'signal_strength_chart',
    'candlestick_chart',
    'price_with_signals',
    'COLORS',
    # Tables
    'kpi_card',
    'kpi_row',
    'watchlist_table',
    'signals_table',
    'risk_summary_table',
    'sentiment_summary_table',
    'news_articles_table',
]
