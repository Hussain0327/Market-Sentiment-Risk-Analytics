"""
Dashboard package for Streamlit web application.

Contains:
- Main application entry point
- Page components
- Visualization components
"""

# Lazy imports to avoid dependency issues
__all__ = [
    'DashboardDataLoader',
]

def __getattr__(name):
    if name == 'DashboardDataLoader':
        from .data_loader import DashboardDataLoader
        return DashboardDataLoader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
