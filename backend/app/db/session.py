"""
Async database session management using SQLModel and aiosqlite
Provides engine creation, session management, and database initialization
"""

import os
import logging
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, AsyncEngine, async_sessionmaker
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel

from ..config import settings

logger = logging.getLogger(__name__)

# Global engine instance
_engine: Optional[AsyncEngine] = None
_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


def get_database_url() -> str:
    """
    Get database URL from configuration
    
    Returns:
        Database URL string for SQLAlchemy
    """
    # Use configured database URL or default to SQLite
    if hasattr(settings, 'DATABASE_URL') and settings.DATABASE_URL:
        return settings.DATABASE_URL
    
    # Default to SQLite database in the project directory
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "gene_analysis.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    return f"sqlite+aiosqlite:///{db_path}"


def create_engine(database_url: Optional[str] = None, echo: bool = False) -> AsyncEngine:
    """
    Create async database engine
    
    Args:
        database_url: Database URL (uses default if None)
        echo: Whether to echo SQL queries (for debugging)
        
    Returns:
        AsyncEngine instance
    """
    url = database_url or get_database_url()
    
    # SQLite-specific configuration for development
    if url.startswith("sqlite"):
        engine = create_async_engine(
            url,
            echo=echo,
            future=True,
            # SQLite specific settings
            poolclass=StaticPool,
            connect_args={
                "check_same_thread": False,
                "timeout": 20,  # 20 second timeout
            },
        )
    else:
        # PostgreSQL or other database configuration
        engine = create_async_engine(
            url,
            echo=echo,
            future=True,
            # Connection pool settings for production
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            pool_recycle=3600,  # Recycle connections every hour
        )
    
    logger.info(f"Created database engine for: {url.split('://')[0]}://[DATABASE]")
    return engine


async def init_db(engine: Optional[AsyncEngine] = None) -> None:
    """
    Initialize database by creating all tables
    
    Args:
        engine: Database engine (uses global if None)
    """
    global _engine
    
    if engine is None:
        engine = _engine
        
    if engine is None:
        raise RuntimeError("No database engine available")
    
    try:
        # Import all models to ensure they're registered
        from .models import (
            UserSession, Analysis, AnalysisChunk, 
            AnalysisTemplate, SearchIndex, CostTracking,
            ContextAugmentation, AnalysisVersion, AdminMetrics
        )
        
        logger.info("Creating database tables...")
        
        # Create all tables
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
            
        logger.info("Database tables created successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


def get_engine() -> AsyncEngine:
    """
    Get the global database engine
    
    Returns:
        AsyncEngine instance
        
    Raises:
        RuntimeError: If engine is not initialized
    """
    global _engine
    
    if _engine is None:
        raise RuntimeError("Database engine not initialized. Call setup_database() first.")
    
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """
    Get the global session factory
    
    Returns:
        Session factory for creating database sessions
        
    Raises:
        RuntimeError: If session factory is not initialized
    """
    global _session_factory
    
    if _session_factory is None:
        raise RuntimeError("Session factory not initialized. Call setup_database() first.")
    
    return _session_factory


async def setup_database(database_url: Optional[str] = None, echo: bool = False, init_tables: bool = True) -> AsyncEngine:
    """
    Setup database engine and session factory
    
    Args:
        database_url: Database URL (uses default if None)
        echo: Whether to echo SQL queries
        init_tables: Whether to create tables on startup
        
    Returns:
        Configured AsyncEngine
    """
    global _engine, _session_factory
    
    # Create engine
    _engine = create_engine(database_url, echo)
    
    # Create session factory
    _session_factory = async_sessionmaker(
        bind=_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=True,
        autocommit=False,
    )
    
    # Initialize tables if requested
    if init_tables:
        await init_db(_engine)
    
    return _engine


async def close_database() -> None:
    """
    Close database engine and cleanup connections
    """
    global _engine, _session_factory
    
    if _engine:
        logger.info("Closing database engine...")
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("Database engine closed")


@asynccontextmanager
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager for database sessions with automatic cleanup
    
    Yields:
        AsyncSession instance
        
    Example:
        async with get_async_session() as session:
            # Use session for database operations
            result = await session.execute(select(UserSession))
            await session.commit()
    """
    session_factory = get_session_factory()
    
    async with session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_session() -> AsyncSession:
    """
    Get a new database session (for dependency injection)
    
    Note: This is primarily for FastAPI dependency injection.
    For manual session management, prefer get_async_session() context manager.
    
    Returns:
        AsyncSession instance
        
    Usage:
        @app.get("/users/")
        async def get_users(session: AsyncSession = Depends(get_session)):
            result = await session.execute(select(UserSession))
            return result.scalars().all()
    """
    session_factory = get_session_factory()
    return session_factory()


# Health check utilities

async def check_database_health() -> dict:
    """
    Check database connectivity and basic health
    
    Returns:
        Dictionary with health status information
    """
    try:
        async with get_async_session() as session:
            # Simple query to test connectivity
            from sqlalchemy import text
            result = await session.execute(text("SELECT 1 as health_check"))
            row = result.fetchone()
            
            if row and row[0] == 1:
                return {
                    "status": "healthy",
                    "database": "connected",
                    "message": "Database is accessible"
                }
            else:
                return {
                    "status": "unhealthy",
                    "database": "error",
                    "message": "Unexpected query result"
                }
                
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "status": "unhealthy",
            "database": "error",
            "message": str(e)
        }


async def get_database_stats() -> dict:
    """
    Get database statistics and table counts
    
    Returns:
        Dictionary with database statistics
    """
    try:
        async with get_async_session() as session:
            from .models import UserSession, Analysis, AnalysisChunk
            from sqlmodel import select, func
            
            # Get table counts
            user_count = await session.scalar(select(func.count(UserSession.id)))
            analysis_count = await session.scalar(select(func.count(Analysis.id)))
            chunk_count = await session.scalar(select(func.count(AnalysisChunk.id)))
            
            return {
                "user_sessions": user_count or 0,
                "analyses": analysis_count or 0,
                "analysis_chunks": chunk_count or 0,
                "database_url": get_database_url().split("://")[0] + "://[REDACTED]"
            }
            
    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        return {
            "error": str(e),
            "database_url": get_database_url().split("://")[0] + "://[REDACTED]"
        }


# Testing utilities

async def create_test_database(test_db_url: Optional[str] = None) -> AsyncEngine:
    """
    Create a test database engine for testing
    
    Args:
        test_db_url: Test database URL (defaults to in-memory SQLite)
        
    Returns:
        Test database engine
    """
    if test_db_url is None:
        # Use in-memory SQLite for testing
        test_db_url = "sqlite+aiosqlite:///:memory:"
    
    # Create test engine
    test_engine = create_engine(test_db_url, echo=False)
    
    # Initialize tables
    await init_db(test_engine)
    
    logger.info("Test database created and initialized")
    return test_engine


@asynccontextmanager
async def test_session(test_engine: AsyncEngine) -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager for test database sessions
    
    Args:
        test_engine: Test database engine
        
    Yields:
        AsyncSession for testing
    """
    # Create test session factory
    test_session_factory = async_sessionmaker(
        bind=test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=True,
        autocommit=False,
    )
    
    async with test_session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def cleanup_test_database(test_engine: AsyncEngine) -> None:
    """
    Clean up test database resources
    
    Args:
        test_engine: Test database engine to cleanup
    """
    await test_engine.dispose()
    logger.info("Test database cleaned up")