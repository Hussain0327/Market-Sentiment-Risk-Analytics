"""
Dashboard Trading Signals Page.

Signal analysis with:
- KPI row: Signal counts and latest
- Signal strength scatter chart
- Recent signals table
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
    signal_strength_chart,
    price_with_signals
)
from dashboard.components.tables import (
    kpi_row,
    signals_table
)


def render(loader: DashboardDataLoader, selected_symbol: str) -> None:
    """
    Render the trading signals page.

    Args:
        loader: Data loader instance.
        selected_symbol: Currently selected symbol.
    """
    st.header("Trading Signals")

    # Load data
    signals_df = loader.load_sentiment_signals()
    prices_df = loader.load_prices(selected_symbol)

    # Filter for selected symbol
    symbol_signals = signals_df[signals_df['symbol'] == selected_symbol] if not signals_df.empty else pd.DataFrame()

    # KPI row
    _render_kpis(signals_df, symbol_signals, selected_symbol)

    st.divider()

    # Signal overview for selected symbol
    st.subheader(f"Signals: {selected_symbol}")

    if symbol_signals.empty:
        st.info(f"No signals available for {selected_symbol}")
    else:
        # Two columns: Chart + Stats
        col1, col2 = st.columns([3, 1])

        with col1:
            # Price with signals overlay
            if not prices_df.empty:
                # Convert Date column for matching
                prices_df_copy = prices_df.copy()
                prices_df_copy['Date'] = pd.to_datetime(prices_df_copy['Date']).dt.tz_localize(None)

                fig = price_with_signals(
                    prices_df_copy,
                    symbol_signals,
                    price_date_col='Date',
                    price_col='Close',
                    signal_date_col='date',
                    signal_col='signal',
                    title='Price with Signals',
                    height=400
                )
                st.plotly_chart(fig, width='stretch')
            else:
                # Just show signal strength chart
                fig = signal_strength_chart(
                    symbol_signals,
                    date_col='date',
                    strength_col='signal_strength',
                    direction_col='signal',
                    title='Signal Strength',
                    height=400
                )
                st.plotly_chart(fig, width='stretch')

        with col2:
            _render_signal_stats(symbol_signals)

    st.divider()

    # Signal strength chart for all signals
    st.subheader("Signal Strength Over Time")

    if not symbol_signals.empty:
        fig = signal_strength_chart(
            symbol_signals,
            date_col='date',
            strength_col='signal_strength',
            direction_col='signal',
            title='',
            height=350
        )
        st.plotly_chart(fig, width='stretch')

    st.divider()

    # Recent signals table - all symbols
    st.subheader("All Recent Signals")

    if not signals_df.empty:
        signals_sorted = signals_df.sort_values('date', ascending=False)
        signals_table(
            signals_sorted,
            symbol_col='symbol',
            date_col='date',
            signal_col='signal',
            strength_col='signal_strength',
            max_rows=20
        )
    else:
        st.info("No signals available")


def _render_kpis(all_signals: pd.DataFrame, symbol_signals: pd.DataFrame, symbol: str) -> None:
    """Render KPI metrics row."""
    # Count signals across all symbols
    if not all_signals.empty:
        total_bullish = (all_signals['signal'] == 'bullish').sum()
        total_bearish = (all_signals['signal'] == 'bearish').sum()
    else:
        total_bullish = 0
        total_bearish = 0

    # Symbol-specific stats
    if not symbol_signals.empty:
        avg_strength = symbol_signals['signal_strength'].mean()
        latest_signal = symbol_signals.sort_values('date', ascending=False).iloc[0]['signal']
    else:
        avg_strength = 0
        latest_signal = 'N/A'

    metrics = [
        {
            'label': 'Bullish Signals',
            'value': str(total_bullish)
        },
        {
            'label': 'Bearish Signals',
            'value': str(total_bearish)
        },
        {
            'label': f'{symbol} Avg Strength',
            'value': f"{avg_strength:.2f}"
        },
        {
            'label': f'{symbol} Latest',
            'value': str(latest_signal).title()
        }
    ]

    kpi_row(metrics)


def _render_signal_stats(df: pd.DataFrame) -> None:
    """Render signal statistics sidebar."""
    st.markdown("**Signal Statistics**")

    if df.empty:
        st.write("No data")
        return

    # Count by signal type
    signal_counts = df['signal'].value_counts()

    bullish = signal_counts.get('bullish', 0)
    bearish = signal_counts.get('bearish', 0)
    neutral = signal_counts.get('neutral', 0)
    total = len(df)

    st.write(f"Total: {total}")
    st.write(f"Bullish: {bullish} ({bullish/total:.0%})")
    st.write(f"Bearish: {bearish} ({bearish/total:.0%})")
    st.write(f"Neutral: {neutral} ({neutral/total:.0%})")

    st.markdown("---")
    st.markdown("**Strength Stats**")
    st.write(f"Mean: {df['signal_strength'].mean():.3f}")
    st.write(f"Max: {df['signal_strength'].max():.3f}")
    st.write(f"Min: {df['signal_strength'].min():.3f}")
