"""
Pydantic response models for the API.
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service health status")
    version: str = Field(default="1.0.0", description="API version")


class ErrorResponse(BaseModel):
    """Error response."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")


class SymbolsResponse(BaseModel):
    """List of available symbols."""
    symbols: List[str] = Field(..., description="List of stock symbols")
    count: int = Field(..., description="Number of symbols")


class SentimentDataPoint(BaseModel):
    """Single sentiment data point."""
    date: datetime = Field(..., description="Date of sentiment measurement")
    sentiment_score: float = Field(..., description="Aggregated sentiment score (-1 to 1)")
    sentiment_confidence: float = Field(..., description="Confidence in the sentiment score")
    article_count: int = Field(..., description="Number of articles analyzed")
    bullish_ratio: float = Field(..., description="Ratio of bullish articles")
    bearish_ratio: float = Field(..., description="Ratio of bearish articles")


class SentimentResponse(BaseModel):
    """Current sentiment metrics for a symbol."""
    symbol: str = Field(..., description="Stock symbol")
    latest_sentiment: float = Field(..., description="Most recent sentiment score")
    sentiment_confidence: float = Field(..., description="Confidence in the score")
    article_count: int = Field(..., description="Number of articles analyzed")
    bullish_ratio: float = Field(..., description="Ratio of bullish sentiment")
    bearish_ratio: float = Field(..., description="Ratio of bearish sentiment")
    signal: str = Field(..., description="Trading signal (bullish/neutral/bearish)")
    signal_strength: float = Field(..., description="Signal strength (0-1)")
    last_updated: datetime = Field(..., description="When data was last updated")


class SentimentHistoryResponse(BaseModel):
    """Historical sentiment data for a symbol."""
    symbol: str = Field(..., description="Stock symbol")
    data: List[SentimentDataPoint] = Field(..., description="Historical sentiment data")
    count: int = Field(..., description="Number of data points")


class RiskMetricsResponse(BaseModel):
    """Risk metrics for a symbol."""
    symbol: str = Field(..., description="Stock symbol")
    var_95: float = Field(..., description="95% Value at Risk (daily)")
    var_99: float = Field(..., description="99% Value at Risk (daily)")
    cvar_95: float = Field(..., description="95% Conditional VaR (Expected Shortfall)")
    cvar_99: float = Field(..., description="99% Conditional VaR")
    volatility_21d: float = Field(..., description="21-day annualized volatility")
    volatility_63d: float = Field(..., description="63-day annualized volatility")
    garch_forecast: Optional[float] = Field(None, description="GARCH volatility forecast")
    max_drawdown: float = Field(..., description="Maximum historical drawdown")
    current_drawdown: float = Field(..., description="Current drawdown from peak")
    calmar_ratio: float = Field(..., description="Calmar ratio (return/max drawdown)")
    volatility_regime: str = Field(..., description="Current volatility regime")
    risk_score: float = Field(..., description="Composite risk score (0-10)")
    timestamp: datetime = Field(..., description="When metrics were calculated")


class SignalDataPoint(BaseModel):
    """Single trading signal data point."""
    date: datetime = Field(..., description="Date of the signal")
    sentiment_score: float = Field(..., description="Sentiment score")
    signal: str = Field(..., description="Trading signal (bullish/neutral/bearish)")
    signal_strength: float = Field(..., description="Signal strength (0-1)")


class SignalsResponse(BaseModel):
    """Trading signals for a symbol."""
    symbol: str = Field(..., description="Stock symbol")
    latest_signal: str = Field(..., description="Most recent trading signal")
    latest_strength: float = Field(..., description="Strength of the latest signal")
    bullish_count: int = Field(default=0, description="Count of bullish signals")
    bearish_count: int = Field(default=0, description="Count of bearish signals")
    neutral_count: int = Field(default=0, description="Count of neutral signals")
    history: List[SignalDataPoint] = Field(default_factory=list, description="Signal history")


class PriceDataPoint(BaseModel):
    """Single price data point (OHLCV)."""
    date: datetime = Field(..., description="Date")
    open: float = Field(..., description="Opening price")
    high: float = Field(..., description="High price")
    low: float = Field(..., description="Low price")
    close: float = Field(..., description="Closing price")
    volume: int = Field(..., description="Trading volume")


class PriceResponse(BaseModel):
    """Latest price data for a symbol."""
    symbol: str = Field(..., description="Stock symbol")
    name: Optional[str] = Field(None, description="Company name")
    currency: str = Field(default="USD", description="Currency")
    exchange: Optional[str] = Field(None, description="Stock exchange")
    open: Optional[float] = Field(None, description="Opening price")
    high: Optional[float] = Field(None, description="High price")
    low: Optional[float] = Field(None, description="Low price")
    close: Optional[float] = Field(None, description="Closing price")
    volume: Optional[int] = Field(None, description="Trading volume")
    market_cap: Optional[float] = Field(None, description="Market capitalization")
    pe_ratio: Optional[float] = Field(None, description="P/E ratio")
    fifty_two_week_high: Optional[float] = Field(None, description="52-week high")
    fifty_two_week_low: Optional[float] = Field(None, description="52-week low")
    date: Optional[datetime] = Field(None, description="Price date")


class PriceHistoryResponse(BaseModel):
    """Historical price data for a symbol."""
    symbol: str = Field(..., description="Stock symbol")
    data: List[PriceDataPoint] = Field(..., description="Historical price data")
    count: int = Field(..., description="Number of data points")


# =============================================================================
# ML Predictions Schemas
# =============================================================================

class MLPredictionDataPoint(BaseModel):
    """Single ML prediction data point."""
    date: datetime = Field(..., description="Date of the prediction")
    direction: str = Field(..., description="Predicted direction (bullish/bearish/neutral)")
    confidence: float = Field(..., description="Model confidence (0-1)")
    expected_return: Optional[float] = Field(None, description="Expected return (%)")
    signal_strength: float = Field(..., description="Combined signal strength (-1 to 1)")


class ModelInfo(BaseModel):
    """Information about a trained model."""
    trained_at: Optional[datetime] = Field(None, description="When the model was trained")
    accuracy: Optional[float] = Field(None, description="Validation accuracy (classifier)")
    r2_score: Optional[float] = Field(None, description="Validation R2 score (regressor)")
    direction_accuracy: Optional[float] = Field(None, description="Direction accuracy (regressor)")
    n_samples: Optional[int] = Field(None, description="Number of training samples")
    n_features: Optional[int] = Field(None, description="Number of features used")


class MLPredictionResponse(BaseModel):
    """ML prediction response for a symbol."""
    symbol: str = Field(..., description="Stock symbol")
    model_available: bool = Field(..., description="Whether a trained model exists")
    latest_prediction: Optional[MLPredictionDataPoint] = Field(
        None, description="Most recent prediction"
    )
    model_info: Optional[ModelInfo] = Field(None, description="Model metadata")
    history: List[MLPredictionDataPoint] = Field(
        default_factory=list, description="Prediction history"
    )
    message: Optional[str] = Field(
        None, description="Status message (e.g., instructions if model unavailable)"
    )


class MLModelsListResponse(BaseModel):
    """List of symbols with trained ML models."""
    symbols: List[str] = Field(..., description="Symbols with trained models")
    count: int = Field(..., description="Number of trained models")
