"""
Risk API routes.

Provides endpoints for risk metrics (VaR, volatility, drawdown).
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from api.schemas import RiskMetricsResponse
from dashboard.data_loader import DashboardDataLoader
from src.risk import RiskReport

router = APIRouter()

# Shared instances
_data_loader: Optional[DashboardDataLoader] = None
_risk_report: Optional[RiskReport] = None


def get_data_loader() -> DashboardDataLoader:
    """Get or create the data loader instance."""
    global _data_loader
    if _data_loader is None:
        _data_loader = DashboardDataLoader()
    return _data_loader


def get_risk_report() -> RiskReport:
    """Get or create the risk report generator."""
    global _risk_report
    if _risk_report is None:
        _risk_report = RiskReport()
    return _risk_report


@router.get("/{symbol}", response_model=RiskMetricsResponse)
def get_risk_metrics(
    symbol: str,
    period: str = Query(
        default="1y",
        description="Price history period (1mo, 3mo, 6mo, 1y, 2y)"
    )
):
    """
    Get comprehensive risk metrics for a symbol.

    Calculates VaR, volatility, drawdown, and composite risk scores.
    """
    symbol = symbol.upper()
    loader = get_data_loader()
    risk_report = get_risk_report()

    # Load price data
    prices_df = loader.load_prices(symbol)

    if prices_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No price data found for symbol: {symbol}"
        )

    # Get the close prices as a Series
    close_col = "Close" if "Close" in prices_df.columns else "close"
    if close_col not in prices_df.columns:
        raise HTTPException(
            status_code=500,
            detail=f"Price data missing Close column for symbol: {symbol}"
        )

    prices = prices_df[close_col].dropna()

    if len(prices) < 30:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient price data for risk calculation (need at least 30 days)"
        )

    # Generate the risk report
    try:
        metrics = risk_report.generate_report(prices, symbol=symbol)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error calculating risk metrics: {str(e)}"
        )

    # Calculate composite risk score
    risk_score = risk_report._calculate_risk_score(metrics)

    return RiskMetricsResponse(
        symbol=symbol,
        var_95=metrics.var_95,
        var_99=metrics.var_99,
        cvar_95=metrics.cvar_95,
        cvar_99=metrics.cvar_99,
        volatility_21d=metrics.volatility_21d,
        volatility_63d=metrics.volatility_63d,
        garch_forecast=metrics.garch_forecast,
        max_drawdown=metrics.max_drawdown,
        current_drawdown=metrics.current_drawdown,
        calmar_ratio=metrics.calmar_ratio,
        volatility_regime=metrics.volatility_regime,
        risk_score=risk_score,
        timestamp=metrics.timestamp,
    )


@router.get("/{symbol}/full")
def get_full_risk_report(symbol: str):
    """
    Get detailed risk report with all analysis.

    Returns comprehensive risk analysis including detailed VaR,
    volatility term structure, and top drawdown periods.
    """
    symbol = symbol.upper()
    loader = get_data_loader()
    risk_report = get_risk_report()

    # Load price data
    prices_df = loader.load_prices(symbol)

    if prices_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No price data found for symbol: {symbol}"
        )

    close_col = "Close" if "Close" in prices_df.columns else "close"
    prices = prices_df[close_col].dropna()

    if len(prices) < 30:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient price data for risk calculation"
        )

    try:
        full_report = risk_report.generate_full_report(prices, symbol=symbol)
        return full_report
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating full risk report: {str(e)}"
        )
