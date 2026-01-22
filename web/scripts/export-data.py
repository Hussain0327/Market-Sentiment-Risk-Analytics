#!/usr/bin/env python3
"""Export CSV data to JSON for Next.js dashboard."""
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

# Paths
DATA_RAW = project_root / "data" / "raw"
DATA_PROCESSED = project_root / "data" / "processed"
DATA_MODELS = project_root / "data" / "models"
OUTPUT_DIR = Path(__file__).parent.parent / "public" / "data"

SYMBOLS = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"]


def clean_for_json(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return clean_for_json(obj.tolist())
    elif pd.isna(obj):
        return None
    return obj


def export_symbols():
    """Export list of symbols."""
    output = {
        "symbols": SYMBOLS,
        "exported_at": datetime.now().isoformat()
    }

    output_path = OUTPUT_DIR / "symbols.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Exported symbols to {output_path}")


def export_prices():
    """Export price data for each symbol."""
    prices_dir = OUTPUT_DIR / "prices"
    prices_dir.mkdir(parents=True, exist_ok=True)

    for symbol in SYMBOLS:
        price_file = DATA_RAW / f"{symbol}_prices.csv"
        if not price_file.exists():
            print(f"Warning: {price_file} not found")
            continue

        df = pd.read_csv(price_file)
        # Handle mixed timezone dates by extracting just the date portion
        df['Date'] = df['Date'].str.split(' ').str[0]

        # Get latest and history
        df_sorted = df.sort_values('Date', ascending=False)
        latest = df_sorted.iloc[0]

        # Calculate stats
        close_prices = df_sorted['Close'].values
        returns = np.diff(close_prices[::-1]) / close_prices[::-1][:-1] * 100

        output = {
            "symbol": symbol,
            "latest": {
                "date": latest['Date'],
                "open": round(latest['Open'], 2),
                "high": round(latest['High'], 2),
                "low": round(latest['Low'], 2),
                "close": round(latest['Close'], 2),
                "volume": int(latest['Volume']),
            },
            "stats": {
                "avg_volume": int(df['Volume'].mean()),
                "price_range_52w": {
                    "low": round(df['Low'].min(), 2),
                    "high": round(df['High'].max(), 2),
                },
                "total_return": round(
                    (close_prices[0] - close_prices[-1]) / close_prices[-1] * 100, 2
                ) if len(close_prices) > 1 else 0,
            },
            "history": [
                {
                    "date": row['Date'],
                    "open": round(row['Open'], 2),
                    "high": round(row['High'], 2),
                    "low": round(row['Low'], 2),
                    "close": round(row['Close'], 2),
                    "volume": int(row['Volume']),
                }
                for _, row in df_sorted.head(90).iterrows()  # Last 90 days
            ]
        }

        output_path = prices_dir / f"{symbol}.json"
        with open(output_path, "w") as f:
            json.dump(clean_for_json(output), f, indent=2)
        print(f"Exported prices for {symbol}")


def export_sentiment():
    """Export sentiment data for each symbol."""
    sentiment_dir = OUTPUT_DIR / "sentiment"
    sentiment_dir.mkdir(parents=True, exist_ok=True)

    # Load sentiment signals
    signals_file = DATA_PROCESSED / "sentiment_signals.csv"
    if not signals_file.exists():
        print(f"Warning: {signals_file} not found")
        return

    df = pd.read_csv(signals_file)
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

    for symbol in SYMBOLS:
        symbol_df = df[df['symbol'] == symbol].sort_values('date', ascending=False)

        if symbol_df.empty:
            print(f"Warning: No sentiment data for {symbol}")
            continue

        latest = symbol_df.iloc[0]

        output = {
            "symbol": symbol,
            "latest": {
                "date": latest['date'],
                "score": round(latest['sentiment_score'], 4),
                "confidence": round(latest['sentiment_confidence'], 4),
                "signal": latest['signal'],
                "signal_strength": round(latest['signal_strength'], 4),
                "article_count": int(latest['article_count']),
                "bullish_ratio": round(latest['bullish_ratio'], 4),
                "bearish_ratio": round(latest['bearish_ratio'], 4),
            },
            "history": [
                {
                    "date": row['date'],
                    "score": round(row['sentiment_score'], 4),
                    "confidence": round(row['sentiment_confidence'], 4),
                    "signal": row['signal'],
                    "signal_strength": round(row['signal_strength'], 4),
                    "article_count": int(row['article_count']),
                }
                for _, row in symbol_df.iterrows()
            ]
        }

        output_path = sentiment_dir / f"{symbol}.json"
        with open(output_path, "w") as f:
            json.dump(clean_for_json(output), f, indent=2)
        print(f"Exported sentiment for {symbol}")


def export_risk():
    """Export risk metrics for each symbol."""
    risk_dir = OUTPUT_DIR / "risk"
    risk_dir.mkdir(parents=True, exist_ok=True)

    for symbol in SYMBOLS:
        price_file = DATA_RAW / f"{symbol}_prices.csv"
        if not price_file.exists():
            continue

        df = pd.read_csv(price_file)
        # Handle mixed timezone dates by extracting just the date portion
        df['Date'] = df['Date'].str.split(' ').str[0]
        df = df.sort_values('Date')

        # Calculate returns
        df['returns'] = df['Close'].pct_change()
        returns = df['returns'].dropna()

        # Calculate risk metrics
        volatility_21d = returns.tail(21).std() * np.sqrt(252) * 100
        volatility_63d = returns.tail(63).std() * np.sqrt(252) * 100 if len(returns) >= 63 else volatility_21d

        # VaR (Historical)
        var_95 = np.percentile(returns, 5) * 100
        var_99 = np.percentile(returns, 1) * 100

        # Expected Shortfall
        es_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100
        es_99 = returns[returns <= np.percentile(returns, 1)].mean() * 100

        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        current_drawdown = drawdown.iloc[-1] * 100 if len(drawdown) > 0 else 0

        # Sharpe ratio (assuming risk-free rate of 5%)
        annual_return = returns.mean() * 252 * 100
        sharpe = (annual_return - 5) / (volatility_21d) if volatility_21d > 0 else 0

        # Create risk score (0-100)
        risk_score = min(100, max(0,
            50 + (volatility_21d - 20) * 2 + abs(var_95) * 2 + abs(max_drawdown)
        ))

        output = {
            "symbol": symbol,
            "metrics": {
                "volatility_21d": round(volatility_21d, 2),
                "volatility_63d": round(volatility_63d, 2),
                "var_95": round(var_95, 2),
                "var_99": round(var_99, 2),
                "es_95": round(es_95, 2),
                "es_99": round(es_99, 2),
                "max_drawdown": round(max_drawdown, 2),
                "current_drawdown": round(current_drawdown, 2),
                "sharpe_ratio": round(sharpe, 2),
                "risk_score": round(risk_score, 0),
            },
            "history": [
                {
                    "date": row['Date'],
                    "volatility": round(returns.tail(len(returns) - i).head(21).std() * np.sqrt(252) * 100, 2) if i + 21 <= len(returns) else None,
                    "drawdown": round(drawdown.iloc[-(i+1)] * 100, 2) if i < len(drawdown) else None,
                }
                for i, (_, row) in enumerate(df.tail(30).iloc[::-1].iterrows())
            ]
        }

        output_path = risk_dir / f"{symbol}.json"
        with open(output_path, "w") as f:
            json.dump(clean_for_json(output), f, indent=2)
        print(f"Exported risk for {symbol}")


def export_predictions():
    """Export model predictions/metadata for each symbol."""
    predictions_dir = OUTPUT_DIR / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    for symbol in SYMBOLS:
        model_dir = DATA_MODELS / symbol
        metadata_file = model_dir / "metadata.json"

        if not metadata_file.exists():
            print(f"Warning: No model metadata for {symbol}")
            continue

        with open(metadata_file) as f:
            metadata = json.load(f)

        output = {
            "symbol": symbol,
            "model": {
                "trained_at": metadata.get("trained_at"),
                "n_samples": metadata.get("n_samples"),
                "n_features": metadata.get("n_features"),
            },
            "classifier": {
                "auc": round(metadata.get("classifier_metrics", {}).get("auc_mean", 0), 4),
                "accuracy": round(metadata.get("classifier_metrics", {}).get("accuracy_mean", 0), 4),
                "precision": round(metadata.get("classifier_metrics", {}).get("precision_mean", 0), 4),
                "recall": round(metadata.get("classifier_metrics", {}).get("recall_mean", 0), 4),
                "f1": round(metadata.get("classifier_metrics", {}).get("f1_mean", 0), 4),
            },
            "regressor": {
                "r2": round(metadata.get("regressor_metrics", {}).get("r2_mean", 0), 4),
                "rmse": round(metadata.get("regressor_metrics", {}).get("rmse_mean", 0), 6),
                "mae": round(metadata.get("regressor_metrics", {}).get("mae_mean", 0), 6),
                "direction_accuracy": round(metadata.get("regressor_metrics", {}).get("direction_accuracy_mean", 0), 4),
            },
            "features": metadata.get("feature_names", [])[:20],  # Top 20 features
        }

        output_path = predictions_dir / f"{symbol}.json"
        with open(output_path, "w") as f:
            json.dump(clean_for_json(output), f, indent=2)
        print(f"Exported predictions for {symbol}")


def main():
    """Run all exports."""
    print("Exporting data for Next.js dashboard...")
    print(f"Output directory: {OUTPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    export_symbols()
    export_prices()
    export_sentiment()
    export_risk()
    export_predictions()

    print("\nExport complete!")


if __name__ == "__main__":
    main()
