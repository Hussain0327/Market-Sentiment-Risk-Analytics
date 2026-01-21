"""
Market Sentiment API - FastAPI Application.

Provides REST endpoints for market sentiment and risk analytics,
serving as the backend for the Streamlit dashboard.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import sentiment, risk, signals, prices, predictions
from api.schemas import HealthResponse

app = FastAPI(
    title="Market Sentiment API",
    description="REST API for market sentiment and risk analytics",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS for Streamlit dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(sentiment.router, prefix="/api/sentiment", tags=["sentiment"])
app.include_router(risk.router, prefix="/api/risk", tags=["risk"])
app.include_router(signals.router, prefix="/api/signals", tags=["signals"])
app.include_router(prices.router, prefix="/api/prices", tags=["prices"])
app.include_router(predictions.router, prefix="/api/predictions", tags=["predictions"])


@app.get("/api/health", response_model=HealthResponse, tags=["health"])
def health_check():
    """
    Health check endpoint.

    Returns the service status and API version.
    """
    return HealthResponse(status="healthy", version="1.0.0")


@app.get("/", tags=["root"])
def root():
    """Root endpoint with API info."""
    return {
        "name": "Market Sentiment API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
    }
