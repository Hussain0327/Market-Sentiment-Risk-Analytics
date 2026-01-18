"""
Watchlist management for tracking stock symbols.
"""

import json
from pathlib import Path
from typing import Optional

import yfinance as yf


class Watchlist:
    """Manage list of tracked symbols."""

    # Default symbols for a new watchlist
    DEFAULT_SYMBOLS = ["AAPL", "GOOGL", "MSFT", "AMZN", "META", "NVDA", "TSLA"]

    def __init__(self, symbols: Optional[list[str]] = None):
        """
        Initialize the watchlist.

        Args:
            symbols: Initial list of symbols. If None, uses default symbols.
        """
        if symbols is not None:
            self._symbols = set(s.upper() for s in symbols)
        else:
            self._symbols = set(self.DEFAULT_SYMBOLS)

    def add_symbol(self, symbol: str) -> bool:
        """
        Add a symbol to the watchlist.

        Args:
            symbol: Stock symbol to add.

        Returns:
            True if symbol was added, False if already present.
        """
        symbol = symbol.upper()
        if symbol in self._symbols:
            return False
        self._symbols.add(symbol)
        return True

    def add_symbols(self, symbols: list[str]) -> dict[str, bool]:
        """
        Add multiple symbols to the watchlist.

        Args:
            symbols: List of stock symbols to add.

        Returns:
            Dictionary mapping symbols to whether they were added.
        """
        return {symbol: self.add_symbol(symbol) for symbol in symbols}

    def remove_symbol(self, symbol: str) -> bool:
        """
        Remove a symbol from the watchlist.

        Args:
            symbol: Stock symbol to remove.

        Returns:
            True if symbol was removed, False if not present.
        """
        symbol = symbol.upper()
        if symbol not in self._symbols:
            return False
        self._symbols.remove(symbol)
        return True

    def remove_symbols(self, symbols: list[str]) -> dict[str, bool]:
        """
        Remove multiple symbols from the watchlist.

        Args:
            symbols: List of stock symbols to remove.

        Returns:
            Dictionary mapping symbols to whether they were removed.
        """
        return {symbol: self.remove_symbol(symbol) for symbol in symbols}

    def get_symbols(self) -> list[str]:
        """
        Get all symbols in the watchlist.

        Returns:
            Sorted list of symbols.
        """
        return sorted(self._symbols)

    def contains(self, symbol: str) -> bool:
        """
        Check if a symbol is in the watchlist.

        Args:
            symbol: Stock symbol to check.

        Returns:
            True if symbol is in watchlist.
        """
        return symbol.upper() in self._symbols

    def clear(self) -> None:
        """Remove all symbols from the watchlist."""
        self._symbols.clear()

    def __len__(self) -> int:
        """Return the number of symbols in the watchlist."""
        return len(self._symbols)

    def __iter__(self):
        """Iterate over symbols in sorted order."""
        return iter(self.get_symbols())

    def __contains__(self, symbol: str) -> bool:
        """Check if symbol is in watchlist using 'in' operator."""
        return self.contains(symbol)

    def validate_symbol(self, symbol: str) -> dict:
        """
        Validate a single symbol and get basic info.

        Args:
            symbol: Stock symbol to validate.

        Returns:
            Dictionary with validation result and info.
        """
        symbol = symbol.upper()
        result = {
            "symbol": symbol,
            "valid": False,
            "name": None,
            "type": None,
            "exchange": None,
            "error": None
        }

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Check if we got valid data
            if info and info.get("regularMarketPrice") is not None:
                result["valid"] = True
                result["name"] = info.get("shortName") or info.get("longName")
                result["type"] = info.get("quoteType")
                result["exchange"] = info.get("exchange")
            else:
                # Try to get history as fallback
                hist = ticker.history(period="5d")
                if not hist.empty:
                    result["valid"] = True
                    result["name"] = info.get("shortName") or info.get("longName") or symbol
                else:
                    result["error"] = "No data available"

        except Exception as e:
            result["error"] = str(e)

        return result

    def validate_symbols(self) -> dict[str, dict]:
        """
        Validate all symbols in the watchlist.

        Returns:
            Dictionary mapping symbols to their validation results.
        """
        return {symbol: self.validate_symbol(symbol) for symbol in self._symbols}

    def get_valid_symbols(self) -> list[str]:
        """
        Get only valid symbols from the watchlist.

        Returns:
            List of validated symbols.
        """
        validations = self.validate_symbols()
        return sorted([s for s, v in validations.items() if v["valid"]])

    def save(self, filepath: str) -> None:
        """
        Save the watchlist to a JSON file.

        Args:
            filepath: Path to save the watchlist.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "symbols": self.get_symbols(),
            "count": len(self._symbols)
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str) -> None:
        """
        Load a watchlist from a JSON file.

        Args:
            filepath: Path to load the watchlist from.

        Raises:
            FileNotFoundError: If file doesn't exist.
            json.JSONDecodeError: If file is not valid JSON.
        """
        path = Path(filepath)
        with open(path, "r") as f:
            data = json.load(f)

        symbols = data.get("symbols", [])
        self._symbols = set(s.upper() for s in symbols)

    @classmethod
    def from_file(cls, filepath: str) -> "Watchlist":
        """
        Create a watchlist from a JSON file.

        Args:
            filepath: Path to the watchlist file.

        Returns:
            New Watchlist instance.
        """
        watchlist = cls(symbols=[])
        watchlist.load(filepath)
        return watchlist

    def to_dict(self) -> dict:
        """
        Convert watchlist to a dictionary.

        Returns:
            Dictionary representation of the watchlist.
        """
        return {
            "symbols": self.get_symbols(),
            "count": len(self._symbols)
        }

    def merge(self, other: "Watchlist") -> None:
        """
        Merge another watchlist into this one.

        Args:
            other: Another Watchlist to merge.
        """
        self._symbols.update(other._symbols)

    def difference(self, other: "Watchlist") -> list[str]:
        """
        Get symbols in this watchlist but not in another.

        Args:
            other: Another Watchlist to compare.

        Returns:
            List of symbols unique to this watchlist.
        """
        return sorted(self._symbols - other._symbols)

    def intersection(self, other: "Watchlist") -> list[str]:
        """
        Get symbols common to both watchlists.

        Args:
            other: Another Watchlist to compare.

        Returns:
            List of common symbols.
        """
        return sorted(self._symbols & other._symbols)

    def __repr__(self) -> str:
        """String representation of the watchlist."""
        return f"Watchlist({self.get_symbols()})"

    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"Watchlist with {len(self)} symbols: {', '.join(self.get_symbols())}"
