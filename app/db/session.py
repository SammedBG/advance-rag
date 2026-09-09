from contextlib import asynccontextmanager
import logging
from pathlib import Path
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class DatabaseManager:
    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url or settings.effective_database_url
        self.engine: AsyncEngine | None = None
        self.session_factory: async_sessionmaker[AsyncSession] | None = None
        self._initialized = False

    def _setup_engine(self) -> None:
        # Ensure data folder exists for SQLite fallback
        Path("./data").mkdir(parents=True, exist_ok=True)

        url = self.database_url
        try:
            logger.info("Initializing async database engine with URL: %s", url)
            self.engine = create_async_engine(
                url,
                echo=False,
                future=True,
            )
            self.session_factory = async_sessionmaker(
                bind=self.engine,
                expire_on_commit=False,
                class_=AsyncSession,
            )
        except Exception as exc:
            logger.warning(
                "Failed to initialize database engine for '%s' (%s). Falling back to SQLite.",
                url,
                exc,
            )
            fallback_url = settings.sqlite_fallback_url
            self.engine = create_async_engine(
                fallback_url,
                echo=False,
                future=True,
            )
            self.session_factory = async_sessionmaker(
                bind=self.engine,
                expire_on_commit=False,
                class_=AsyncSession,
            )

    async def init_db(self) -> None:
        if not self.engine:
            self._setup_engine()

        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            self._initialized = True
            logger.info("Database schema initialized successfully.")
        except Exception as exc:
            logger.warning(
                "Could not connect to configured database (%s). Switching to local SQLite.",
                exc,
            )
            fallback_url = settings.sqlite_fallback_url
            self.engine = create_async_engine(
                fallback_url,
                echo=False,
                future=True,
            )
            self.session_factory = async_sessionmaker(
                bind=self.engine,
                expire_on_commit=False,
                class_=AsyncSession,
            )
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            self._initialized = True
            logger.info("Local SQLite database initialized successfully at '%s'.", fallback_url)

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        if not self.session_factory:
            self._setup_engine()

        async with self.session_factory() as sess:
            try:
                yield sess
                await sess.commit()
            except Exception:
                await sess.rollback()
                raise


db_manager = DatabaseManager()


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async with db_manager.session() as session:
        yield session
