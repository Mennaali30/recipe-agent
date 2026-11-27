"""Engine and session setup. The database URL comes from RECIPE_AGENT_DB_URL.

SQLite (default):  sqlite:///matbakhak.db
PostgreSQL:        postgresql+psycopg://user:password@host:5432/dbname   (install `psycopg[binary]`)
"""
import os
from typing import Optional

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from .models import Base

DEFAULT_DB_URL = "sqlite:///matbakhak.db"


def get_database_url(url: Optional[str] = None) -> str:
    return url or os.getenv("RECIPE_AGENT_DB_URL") or DEFAULT_DB_URL


def make_engine(url: Optional[str] = None) -> Engine:
    url = get_database_url(url)
    kwargs = {}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        if url in ("sqlite://", "sqlite:///:memory:"):
            kwargs["poolclass"] = StaticPool  # one shared in-memory database
    engine = create_engine(url, future=True, **kwargs)
    if url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def _fk_on(dbapi_conn, _rec):
            dbapi_conn.execute("PRAGMA foreign_keys=ON")
    return engine


def init_db(engine: Engine) -> None:
    """Create the tables if they do not exist."""
    Base.metadata.create_all(engine)


def make_session_factory(engine: Engine):
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)
