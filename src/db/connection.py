"""
Database connection management for Market Sentiment & Risk Analytics.

Provides:
- DatabaseManager: Manages SQLite connections and sessions
- Context managers for transaction handling
- Database initialization
"""

from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.engine import Engine

from .models import Base


# Enable foreign keys for SQLite
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Enable foreign key support for SQLite."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


class DatabaseManager:
    """
    Manages SQLite database connections and sessions.

    Handles database initialization, session creation, and
    provides context managers for transaction handling.

    Example:
        >>> db = DatabaseManager('data/market_sentiment.db')
        >>> db.init_db()
        >>> with db.session() as session:
        ...     symbols = session.query(Symbol).all()
    """

    DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data" / "market_sentiment.db"

    def __init__(self, db_path: Optional[str] = None, echo: bool = False):
        """
        Initialize the database manager.

        Args:
            db_path: Path to SQLite database file.
                    If None, uses default path (data/market_sentiment.db).
            echo: If True, echo SQL statements to stdout (for debugging).
        """
        if db_path is None:
            db_path = str(self.DEFAULT_DB_PATH)

        # Handle in-memory database
        if db_path == ":memory:":
            self.db_url = "sqlite:///:memory:"
        else:
            self.db_path = Path(db_path)
            self.db_url = f"sqlite:///{self.db_path}"

            # Ensure parent directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.engine = create_engine(
            self.db_url,
            echo=echo,
            connect_args={"check_same_thread": False}  # Allow multi-threading
        )
        self._SessionFactory = sessionmaker(bind=self.engine)

    def init_db(self) -> None:
        """
        Create all database tables.

        Creates tables defined in models.py if they don't exist.
        Safe to call multiple times.
        """
        Base.metadata.create_all(self.engine)

    def drop_all(self) -> None:
        """
        Drop all database tables.

        WARNING: This will delete all data. Use with caution.
        """
        Base.metadata.drop_all(self.engine)

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """
        Context manager for database sessions.

        Automatically handles commit on success and rollback on error.

        Yields:
            SQLAlchemy Session object.

        Example:
            >>> with db.session() as session:
            ...     symbol = Symbol(ticker='AAPL', name='Apple Inc.')
            ...     session.add(symbol)
            ...     # Commits automatically on exit
        """
        session = self._SessionFactory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_session(self) -> Session:
        """
        Get a new session (manual management required).

        The caller is responsible for committing, rolling back,
        and closing the session.

        Returns:
            New SQLAlchemy Session object.
        """
        return self._SessionFactory()


# Module-level convenience functions
_default_db: Optional[DatabaseManager] = None


def init_db(db_path: Optional[str] = None, echo: bool = False) -> DatabaseManager:
    """
    Initialize the default database.

    Args:
        db_path: Path to SQLite database file.
        echo: If True, echo SQL statements.

    Returns:
        DatabaseManager instance.

    Example:
        >>> db = init_db('data/market_sentiment.db')
        >>> with db.session() as s:
        ...     # Use session
    """
    global _default_db
    _default_db = DatabaseManager(db_path, echo=echo)
    _default_db.init_db()
    return _default_db


def get_db() -> DatabaseManager:
    """
    Get the default database manager.

    Raises:
        RuntimeError: If database has not been initialized.

    Returns:
        Default DatabaseManager instance.
    """
    if _default_db is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _default_db
