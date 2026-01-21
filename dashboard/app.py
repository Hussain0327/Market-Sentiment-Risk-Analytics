"""
Market Sentiment & Risk Analytics Dashboard.

Main entry point for the Streamlit dashboard application.

Run with: streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st

from dashboard.data_loader import DashboardDataLoader
from dashboard.views import overview, sentiment, risk, signals, predictions


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Market Sentiment Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# Initialize Data Loader
# =============================================================================

@st.cache_resource
def get_data_loader():
    """Get cached data loader instance."""
    return DashboardDataLoader(use_db=False)


# =============================================================================
# Sidebar Navigation
# =============================================================================

def render_sidebar(loader: DashboardDataLoader):
    """
    Render sidebar with navigation and symbol selector.

    Args:
        loader: Data loader instance.

    Returns:
        Tuple of (selected_page, selected_symbol).
    """
    st.sidebar.title("📊 Market Analytics")
    st.sidebar.markdown("---")

    # Navigation
    st.sidebar.subheader("Navigation")
    pages = {
        "Overview": "📈",
        "Sentiment": "😊",
        "Risk": "⚠️",
        "Signals": "🎯",
        "Predictions": "🤖"
    }

    selected_page = st.sidebar.radio(
        "Select Page",
        options=list(pages.keys()),
        format_func=lambda x: f"{pages[x]} {x}"
    )

    st.sidebar.markdown("---")

    # Symbol Selector
    st.sidebar.subheader("Symbol")
    symbols = loader.get_available_symbols()

    if symbols:
        selected_symbol = st.sidebar.selectbox(
            "Select Symbol",
            options=symbols,
            index=0
        )
    else:
        selected_symbol = "AAPL"
        st.sidebar.warning("No symbols found. Using default.")

    st.sidebar.markdown("---")

    # Info section
    st.sidebar.markdown("### About")
    st.sidebar.markdown(
        """
        Market Sentiment & Risk Analytics Dashboard.

        **Features:**
        - Real-time sentiment analysis
        - Risk metrics (VaR, volatility)
        - Trading signal generation
        - Multi-symbol comparison
        """
    )

    return selected_page, selected_symbol


# =============================================================================
# Main Application
# =============================================================================

def main():
    """Main application entry point."""
    # Initialize loader
    loader = get_data_loader()

    # Render sidebar and get selections
    selected_page, selected_symbol = render_sidebar(loader)

    # Route to selected page
    if selected_page == "Overview":
        overview.render(loader, selected_symbol)
    elif selected_page == "Sentiment":
        sentiment.render(loader, selected_symbol)
    elif selected_page == "Risk":
        risk.render(loader, selected_symbol)
    elif selected_page == "Signals":
        signals.render(loader, selected_symbol)
    elif selected_page == "Predictions":
        predictions.render(loader, selected_symbol)


if __name__ == "__main__":
    main()
