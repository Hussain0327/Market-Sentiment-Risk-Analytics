"""
Dashboard Risk Analysis Page.

Risk metrics display with:
- VaR bar chart
- Volatility term structure
- Drawdown underwater chart
- Risk summary table
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import numpy as np

from dashboard.data_loader import DashboardDataLoader
from dashboard.components.charts import (
    var_bar_chart,
    volatility_term_structure,
    drawdown_underwater_chart
)
from dashboard.components.tables import (
    kpi_row,
    risk_summary_table
)


def render(loader: DashboardDataLoader, selected_symbol: str) -> None:
    """
    Render the risk analysis page.

    Args:
        loader: Data loader instance.
        selected_symbol: Currently selected symbol.
    """
    st.header("Risk Analysis")

    # Load price data
    prices_df = loader.load_prices(selected_symbol)

    if prices_df.empty:
        st.warning(f"No price data available for {selected_symbol}")
        return

    # Calculate risk metrics using existing risk module
    risk_metrics = _calculate_risk_metrics(prices_df)

    if risk_metrics is None:
        st.warning("Unable to calculate risk metrics. Insufficient data.")
        return

    # KPI row
    _render_kpis(risk_metrics)

    st.divider()

    # Top row: VaR chart + Volatility term structure
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Value at Risk")
        fig = var_bar_chart(
            risk_metrics['var_95'],
            risk_metrics['var_99'],
            risk_metrics.get('cvar_95'),
            risk_metrics.get('cvar_99'),
            title='',
            height=350
        )
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.subheader("Volatility Term Structure")
        vol_data = {
            '21d': risk_metrics.get('volatility_21d', 0),
            '63d': risk_metrics.get('volatility_63d', 0),
        }
        if 'volatility_126d' in risk_metrics:
            vol_data['126d'] = risk_metrics['volatility_126d']
        if 'volatility_252d' in risk_metrics:
            vol_data['252d'] = risk_metrics['volatility_252d']

        fig = volatility_term_structure(vol_data, title='', height=350)
        st.plotly_chart(fig, width='stretch')

    st.divider()

    # Bottom row: Drawdown chart + Risk summary table
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Drawdown Analysis")
        # Create price series with date index
        price_series = pd.Series(
            prices_df['Close'].values,
            index=pd.to_datetime(prices_df['Date'])
        )
        fig = drawdown_underwater_chart(price_series, title='', height=350)
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.subheader("Risk Summary")
        risk_summary_table(
            var_95=risk_metrics['var_95'],
            var_99=risk_metrics['var_99'],
            cvar_95=risk_metrics.get('cvar_95'),
            cvar_99=risk_metrics.get('cvar_99'),
            volatility_21d=risk_metrics.get('volatility_21d'),
            volatility_63d=risk_metrics.get('volatility_63d'),
            max_drawdown=risk_metrics.get('max_drawdown'),
            current_drawdown=risk_metrics.get('current_drawdown')
        )


def _calculate_risk_metrics(prices_df: pd.DataFrame) -> dict:
    """
    Calculate risk metrics from price data.

    Uses the existing risk module for calculations.

    Args:
        prices_df: DataFrame with price data.

    Returns:
        Dictionary of risk metrics.
    """
    try:
        from src.risk import RiskReport

        prices = pd.Series(prices_df['Close'].values)

        if len(prices) < 30:
            return None

        report = RiskReport()
        metrics = report.generate_report(prices, symbol='temp')

        return {
            'var_95': metrics.var_95,
            'var_99': metrics.var_99,
            'cvar_95': metrics.cvar_95,
            'cvar_99': metrics.cvar_99,
            'volatility_21d': metrics.volatility_21d,
            'volatility_63d': metrics.volatility_63d,
            'max_drawdown': metrics.max_drawdown,
            'current_drawdown': metrics.current_drawdown,
            'volatility_regime': metrics.volatility_regime
        }

    except ImportError:
        # Fallback to basic calculations if risk module not available
        return _basic_risk_metrics(prices_df)
    except Exception as e:
        st.error(f"Error calculating risk metrics: {e}")
        return _basic_risk_metrics(prices_df)


def _basic_risk_metrics(prices_df: pd.DataFrame) -> dict:
    """
    Calculate basic risk metrics without risk module.

    Args:
        prices_df: DataFrame with price data.

    Returns:
        Dictionary of basic risk metrics.
    """
    prices = prices_df['Close'].values
    returns = np.diff(prices) / prices[:-1]

    if len(returns) < 20:
        return None

    # Basic VaR
    var_95 = np.percentile(returns, 5)
    var_99 = np.percentile(returns, 1)

    # CVaR (Expected Shortfall)
    cvar_95 = returns[returns <= var_95].mean()
    cvar_99 = returns[returns <= var_99].mean()

    # Volatility
    vol_21d = np.std(returns[-21:]) * np.sqrt(252) if len(returns) >= 21 else np.std(returns) * np.sqrt(252)
    vol_63d = np.std(returns[-63:]) * np.sqrt(252) if len(returns) >= 63 else vol_21d

    # Drawdown
    cummax = np.maximum.accumulate(prices)
    drawdown = (prices - cummax) / cummax
    max_drawdown = drawdown.min()
    current_drawdown = drawdown[-1]

    return {
        'var_95': var_95,
        'var_99': var_99,
        'cvar_95': cvar_95,
        'cvar_99': cvar_99,
        'volatility_21d': vol_21d,
        'volatility_63d': vol_63d,
        'max_drawdown': max_drawdown,
        'current_drawdown': current_drawdown,
        'volatility_regime': 'unknown'
    }


def _render_kpis(metrics: dict) -> None:
    """Render KPI metrics row."""
    kpis = [
        {
            'label': 'VaR 95%',
            'value': f"{abs(metrics['var_95']):.2%}"
        },
        {
            'label': 'Volatility (21d)',
            'value': f"{metrics.get('volatility_21d', 0):.1%}"
        },
        {
            'label': 'Max Drawdown',
            'value': f"{abs(metrics.get('max_drawdown', 0)):.1%}"
        },
        {
            'label': 'Vol Regime',
            'value': metrics.get('volatility_regime', 'N/A').title()
        }
    ]

    kpi_row(kpis)
