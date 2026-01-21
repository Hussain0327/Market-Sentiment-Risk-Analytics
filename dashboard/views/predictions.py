"""
Dashboard ML Predictions Page.

ML-based prediction visualization with:
- Latest prediction display with confidence
- Prediction history chart
- ML vs Sentiment signal comparison
- Model information display
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import json
from datetime import datetime
from typing import Optional, Dict, List, Any

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard.data_loader import DashboardDataLoader
from dashboard.components.charts import COLORS
from dashboard.components.tables import kpi_row
from dashboard.remote_loader import (
    is_remote_mode,
    get_remote_loader,
)


# =============================================================================
# Model Loading Utilities
# =============================================================================

MODELS_DIR = project_root / "data" / "models"


def get_models_dir():
    """Get the models directory, using remote loader if in cloud mode."""
    if is_remote_mode():
        try:
            loader = get_remote_loader()
            return loader.get_models_dir()
        except Exception as e:
            st.warning(f"Remote model loading failed: {e}. Using local fallback.")

    return MODELS_DIR


def get_trained_symbols() -> List[str]:
    """Get list of symbols with trained models."""
    if is_remote_mode():
        try:
            loader = get_remote_loader()
            return loader.get_trained_symbols()
        except Exception:
            pass

    models_dir = get_models_dir()
    if not models_dir.exists():
        return []

    symbols = []
    for path in models_dir.iterdir():
        if path.is_dir():
            metadata = path / "metadata.json"
            if metadata.exists():
                symbols.append(path.name)

    return sorted(symbols)


def load_model_metadata(symbol: str) -> Optional[Dict]:
    """Load metadata for a trained model."""
    if is_remote_mode():
        try:
            loader = get_remote_loader()
            return loader.load_model_metadata(symbol)
        except Exception:
            pass

    models_dir = get_models_dir()
    metadata_path = models_dir / symbol / "metadata.json"
    if not metadata_path.exists():
        return None

    with open(metadata_path) as f:
        return json.load(f)


def load_models(symbol: str):
    """Load classifier and regressor models for a symbol."""
    from src.ml import DirectionClassifier, ReturnRegressor

    models_dir = get_models_dir()
    model_dir = models_dir / symbol
    classifier_path = model_dir / "model_classifier.joblib"
    regressor_path = model_dir / "model_regressor.joblib"

    if not classifier_path.exists() or not regressor_path.exists():
        return None, None

    try:
        classifier = DirectionClassifier.load(classifier_path)
        regressor = ReturnRegressor.load(regressor_path)
        return classifier, regressor
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None


def build_features_for_prediction(
    loader: DashboardDataLoader,
    symbol: str,
    min_samples: int = 10
) -> Optional[pd.DataFrame]:
    """
    Build features for making predictions.

    If sentiment data causes too few samples, falls back to price/risk features only.
    """
    from src.features import FeatureBuilder

    prices = loader.load_prices(symbol)
    sentiment = loader.load_daily_sentiment(symbol)

    if prices.empty:
        return None

    builder = FeatureBuilder()

    # Try with sentiment first if available
    if not sentiment.empty:
        features = builder.build_features(
            prices=prices,
            sentiment=sentiment,
            symbol=symbol,
            include_volume=True,
            include_garch=False,
            include_sentiment=True,
            include_risk=True
        )

        # Check if we got enough samples
        if not features.empty and len(features) >= min_samples:
            return features

    # Fall back to price/risk features only
    features = builder.build_features(
        prices=prices,
        sentiment=None,
        symbol=symbol,
        include_volume=True,
        include_garch=False,
        include_sentiment=False,
        include_risk=True
    )

    return features


def generate_predictions(
    features: pd.DataFrame,
    classifier,
    regressor,
    days: int = 30
) -> pd.DataFrame:
    """Generate predictions for recent dates."""
    import numpy as np

    # Use last N days of data
    if len(features) > days:
        features = features.tail(days)

    # Get feature columns (exclude non-numeric and symbol)
    feature_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    X = features[feature_cols]

    # Handle any NaN values
    X = X.fillna(0)

    try:
        # Get predictions
        directions = classifier.predict(X)
        probabilities = classifier.predict_proba(X)
        expected_returns = regressor.predict(X)

        # Build results DataFrame
        results = []
        for i, idx in enumerate(X.index):
            direction_val = int(directions[i])
            prob_up = float(probabilities[i, 1])

            # Determine direction string
            if prob_up > 0.55:
                direction_str = "bullish"
            elif prob_up < 0.45:
                direction_str = "bearish"
            else:
                direction_str = "neutral"

            # Confidence is distance from 0.5
            confidence = abs(prob_up - 0.5) * 2

            # Signal strength combines direction and confidence
            signal_strength = (2 * direction_val - 1) * confidence

            # Convert expected return to percentage
            exp_ret = float(expected_returns[i]) * 100

            results.append({
                'date': idx,
                'direction': direction_str,
                'confidence': confidence,
                'expected_return': exp_ret,
                'signal_strength': signal_strength,
                'prob_up': prob_up
            })

        return pd.DataFrame(results)

    except Exception as e:
        st.error(f"Error generating predictions: {e}")
        return pd.DataFrame()


# =============================================================================
# Chart Components
# =============================================================================

def prediction_confidence_gauge(
    direction: str,
    confidence: float,
    expected_return: float,
    height: int = 250
) -> go.Figure:
    """Create gauge showing prediction confidence."""
    # Determine color based on direction
    if direction == "bullish":
        color = COLORS['bullish']
        indicator_value = confidence
    elif direction == "bearish":
        color = COLORS['bearish']
        indicator_value = -confidence
    else:
        color = COLORS['neutral']
        indicator_value = 0

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=indicator_value,
        title={'text': f"Signal: {direction.upper()}<br><span style='font-size:0.7em'>Expected: {expected_return:+.2f}%</span>"},
        number={'suffix': "", 'valueformat': '.0%'},
        gauge={
            'axis': {'range': [-1, 1], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [-1, -0.3], 'color': 'rgba(255,23,68,0.3)'},
                {'range': [-0.3, 0.3], 'color': 'rgba(158,158,158,0.3)'},
                {'range': [0.3, 1], 'color': 'rgba(0,200,83,0.3)'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': indicator_value
            }
        }
    ))

    fig.update_layout(height=height, template='plotly_white')
    return fig


def prediction_history_chart(
    predictions_df: pd.DataFrame,
    height: int = 400
) -> go.Figure:
    """Create prediction history chart with confidence bands."""
    if predictions_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No predictions available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(height=height)
        return fig

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.6, 0.4],
        subplot_titles=("Signal Strength", "Confidence")
    )

    # Color code by direction
    colors = predictions_df['direction'].map({
        'bullish': COLORS['bullish'],
        'bearish': COLORS['bearish'],
        'neutral': COLORS['neutral']
    }).fillna(COLORS['neutral'])

    # Signal strength line with scatter
    fig.add_trace(go.Scatter(
        x=predictions_df['date'],
        y=predictions_df['signal_strength'],
        mode='lines+markers',
        marker=dict(size=8, color=colors),
        line=dict(color=COLORS['primary'], width=2),
        name='Signal Strength',
        hovertemplate='Date: %{x}<br>Strength: %{y:.2f}<extra></extra>'
    ), row=1, col=1)

    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)

    # Confidence bars
    fig.add_trace(go.Bar(
        x=predictions_df['date'],
        y=predictions_df['confidence'],
        marker_color=colors,
        name='Confidence',
        opacity=0.7,
        hovertemplate='Date: %{x}<br>Confidence: %{y:.1%}<extra></extra>'
    ), row=2, col=1)

    fig.update_layout(
        height=height,
        template='plotly_white',
        showlegend=False,
        hovermode='x unified'
    )

    fig.update_yaxes(title_text="Strength", row=1, col=1)
    fig.update_yaxes(title_text="Confidence", tickformat=".0%", row=2, col=1)

    return fig


def ml_vs_sentiment_chart(
    ml_predictions: pd.DataFrame,
    sentiment_signals: pd.DataFrame,
    height: int = 400
) -> go.Figure:
    """Create comparison chart of ML predictions vs sentiment signals."""
    fig = go.Figure()

    if ml_predictions.empty and sentiment_signals.empty:
        fig.add_annotation(
            text="No data available for comparison",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(height=height)
        return fig

    # ML Predictions
    if not ml_predictions.empty:
        fig.add_trace(go.Scatter(
            x=ml_predictions['date'],
            y=ml_predictions['signal_strength'],
            mode='lines+markers',
            name='ML Prediction',
            line=dict(color=COLORS['primary'], width=2),
            marker=dict(size=6)
        ))

    # Sentiment Signals
    if not sentiment_signals.empty:
        # Normalize sentiment signal strength to same scale
        sentiment_signals = sentiment_signals.copy()
        sentiment_signals['date'] = pd.to_datetime(sentiment_signals['date'])

        # Convert signal to numeric if needed
        if 'signal_strength' in sentiment_signals.columns:
            fig.add_trace(go.Scatter(
                x=sentiment_signals['date'],
                y=sentiment_signals['signal_strength'],
                mode='lines+markers',
                name='Sentiment Signal',
                line=dict(color=COLORS['secondary'], width=2, dash='dot'),
                marker=dict(size=6)
            ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title="ML Predictions vs Sentiment Signals",
        xaxis_title="Date",
        yaxis_title="Signal Strength",
        height=height,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


# =============================================================================
# Main View
# =============================================================================

def render(loader: DashboardDataLoader, selected_symbol: str) -> None:
    """
    Render the ML predictions page.

    Args:
        loader: Data loader instance.
        selected_symbol: Currently selected symbol.
    """
    st.header("ML Predictions")

    # Check if model exists for this symbol
    trained_symbols = get_trained_symbols()

    if selected_symbol not in trained_symbols:
        _render_no_model_state(selected_symbol, trained_symbols)
        return

    # Load model metadata
    metadata = load_model_metadata(selected_symbol)

    # Load models
    with st.spinner("Loading models..."):
        classifier, regressor = load_models(selected_symbol)

    if classifier is None or regressor is None:
        st.error(f"Failed to load models for {selected_symbol}")
        return

    # Build features and generate predictions
    with st.spinner("Generating predictions..."):
        features = build_features_for_prediction(loader, selected_symbol)

        if features is None or features.empty:
            st.warning(f"No recent data available for {selected_symbol}")
            return

        predictions_df = generate_predictions(features, classifier, regressor, days=60)

    if predictions_df.empty:
        st.warning("Could not generate predictions")
        return

    # Load sentiment signals for comparison
    sentiment_signals = loader.load_sentiment_signals(selected_symbol)

    # Render the page
    _render_kpis(predictions_df, selected_symbol)
    st.divider()

    # Two-column layout: Latest prediction + Model info
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Latest Prediction")
        _render_latest_prediction(predictions_df)

    with col2:
        st.subheader("Model Info")
        _render_model_info(metadata)

    st.divider()

    # Prediction history
    st.subheader("Prediction History")
    fig = prediction_history_chart(predictions_df)
    st.plotly_chart(fig, width='stretch')

    st.divider()

    # ML vs Sentiment comparison
    st.subheader("ML vs Sentiment Comparison")
    _render_comparison(predictions_df, sentiment_signals)


def _render_no_model_state(symbol: str, trained_symbols: List[str]) -> None:
    """Render state when no model exists for symbol."""
    st.warning(f"No trained model available for {symbol}")

    st.markdown(f"""
    ### Train a Model

    To train a model for **{symbol}**, run:

    ```bash
    python scripts/train_models.py --symbol {symbol}
    ```

    This will:
    1. Load price and sentiment data
    2. Build features using technical indicators and sentiment scores
    3. Run walk-forward validation to evaluate model performance
    4. Save trained models to `data/models/{symbol}/`
    """)

    if trained_symbols:
        st.markdown("---")
        st.markdown("### Available Models")
        st.write(f"Trained models exist for: **{', '.join(trained_symbols)}**")

        st.info("Select one of these symbols from the sidebar to view predictions.")
    else:
        st.markdown("---")
        st.markdown("### No Models Trained Yet")
        st.write("Train your first model by running:")
        st.code("python scripts/train_models.py", language="bash")


def _render_kpis(predictions_df: pd.DataFrame, symbol: str) -> None:
    """Render KPI metrics row."""
    if predictions_df.empty:
        return

    latest = predictions_df.iloc[-1]

    # Count predictions
    bullish_count = (predictions_df['direction'] == 'bullish').sum()
    bearish_count = (predictions_df['direction'] == 'bearish').sum()
    avg_confidence = predictions_df['confidence'].mean()

    metrics = [
        {
            'label': f'{symbol} Latest',
            'value': latest['direction'].title()
        },
        {
            'label': 'Confidence',
            'value': f"{latest['confidence']:.0%}"
        },
        {
            'label': 'Bullish (30d)',
            'value': str(bullish_count)
        },
        {
            'label': 'Bearish (30d)',
            'value': str(bearish_count)
        }
    ]

    kpi_row(metrics)


def _render_latest_prediction(predictions_df: pd.DataFrame) -> None:
    """Render latest prediction with gauge."""
    if predictions_df.empty:
        st.info("No predictions available")
        return

    latest = predictions_df.iloc[-1]

    fig = prediction_confidence_gauge(
        direction=latest['direction'],
        confidence=latest['confidence'],
        expected_return=latest['expected_return'],
        height=280
    )
    st.plotly_chart(fig, width='stretch')

    # Details below gauge
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Expected Return", f"{latest['expected_return']:+.2f}%")
    with col2:
        pred_date = latest['date']
        if isinstance(pred_date, pd.Timestamp):
            pred_date = pred_date.strftime('%Y-%m-%d')
        st.metric("Prediction Date", str(pred_date))


def _render_model_info(metadata: Optional[Dict]) -> None:
    """Render model metadata information."""
    if metadata is None:
        st.info("No model metadata available")
        return

    # Training date
    trained_at = metadata.get('trained_at', 'Unknown')
    if isinstance(trained_at, str) and trained_at != 'Unknown':
        try:
            dt = datetime.fromisoformat(trained_at)
            trained_at = dt.strftime('%Y-%m-%d %H:%M')
        except (ValueError, TypeError):
            pass

    st.write(f"**Trained:** {trained_at}")
    st.write(f"**Samples:** {metadata.get('n_samples', 'N/A')}")
    st.write(f"**Features:** {metadata.get('n_features', 'N/A')}")

    st.markdown("---")
    st.markdown("**Validation Metrics**")

    # Classifier metrics
    classifier_metrics = metadata.get('classifier_metrics', {})
    accuracy = classifier_metrics.get('accuracy_mean')
    if accuracy is not None:
        st.write(f"Accuracy: {accuracy:.1%}")

    # Regressor metrics
    regressor_metrics = metadata.get('regressor_metrics', {})
    dir_acc = regressor_metrics.get('direction_accuracy_mean')
    if dir_acc is not None:
        st.write(f"Direction Acc: {dir_acc:.1%}")

    r2 = regressor_metrics.get('r2_mean')
    if r2 is not None:
        st.write(f"R² Score: {r2:.3f}")


def _render_comparison(predictions_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> None:
    """Render ML vs Sentiment comparison."""
    # Chart
    fig = ml_vs_sentiment_chart(predictions_df, sentiment_df)
    st.plotly_chart(fig, width='stretch')

    # Agreement analysis
    if not predictions_df.empty and not sentiment_df.empty:
        _render_agreement_stats(predictions_df, sentiment_df)


def _render_agreement_stats(ml_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> None:
    """Render agreement statistics between ML and sentiment."""
    # Merge on date
    ml_df = ml_df.copy()
    sentiment_df = sentiment_df.copy()

    ml_df['date'] = pd.to_datetime(ml_df['date']).dt.normalize()
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.normalize()

    merged = ml_df.merge(
        sentiment_df[['date', 'signal']],
        on='date',
        how='inner',
        suffixes=('_ml', '_sent')
    )

    if merged.empty:
        st.info("No overlapping dates for comparison")
        return

    # Calculate agreement
    def signals_agree(ml_dir, sent_signal):
        """Check if ML and sentiment signals agree."""
        sent_signal = str(sent_signal).lower()
        if ml_dir == 'bullish' and sent_signal == 'bullish':
            return True
        if ml_dir == 'bearish' and sent_signal == 'bearish':
            return True
        if ml_dir == 'neutral' and sent_signal == 'neutral':
            return True
        return False

    merged['agrees'] = merged.apply(
        lambda row: signals_agree(row['direction'], row['signal']),
        axis=1
    )

    agreement_rate = merged['agrees'].mean()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Agreement Rate", f"{agreement_rate:.0%}")
    with col2:
        st.metric("Overlapping Days", len(merged))
    with col3:
        agreements = merged['agrees'].sum()
        st.metric("Agreements", f"{agreements}/{len(merged)}")

    # Recent comparison table
    st.markdown("**Recent Comparison**")
    recent = merged.tail(10)[['date', 'direction', 'signal', 'agrees']].copy()
    recent.columns = ['Date', 'ML Prediction', 'Sentiment Signal', 'Agrees']
    recent['Date'] = recent['Date'].dt.strftime('%Y-%m-%d')
    recent['Agrees'] = recent['Agrees'].map({True: 'Yes', False: 'No'})
    st.dataframe(recent, width='stretch', hide_index=True)
