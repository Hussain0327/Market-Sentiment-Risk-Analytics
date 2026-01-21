"""
Dashboard views.

Contains:
- Overview view (main dashboard)
- Sentiment analysis view
- Risk metrics view
- Signals view
- ML Predictions view
"""

from . import overview
from . import sentiment
from . import risk
from . import signals
from . import predictions

__all__ = [
    'overview',
    'sentiment',
    'risk',
    'signals',
    'predictions',
]
