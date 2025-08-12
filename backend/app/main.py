"""
FastAPI application entry point for Gene-Disease Analysis API
Configures the application, middleware, and routes
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging.config
import asyncio
from contextlib import asynccontextmanager

from .config import settings
from .routers import session, analysis
from .storage.memory_store import storage
from .services.worker_pool import get_worker_pool, shutdown_worker_pool
from .db.session import setup_database, close_database, check_database_health, get_database_stats


# Configure logging
logging.config.dictConfig(settings.log_config)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager
    Handles startup and shutdown tasks
    """
    # Startup tasks
    logger.info("Starting Gene-Disease Analysis API")
    logger.info(f"Debug mode: {settings.DEBUG}")
    logger.info(f"Allowed origins: {settings.ALLOWED_ORIGINS}")
    
    # Initialize database
    logger.info("Initializing database...")
    try:
        await setup_database()
        db_health = await check_database_health()
        logger.info(f"Database initialized: {db_health['status']}")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise
    
    # Initialize worker pool
    logger.info("Initializing worker pool...")
    try:
        worker_pool = await get_worker_pool()
        logger.info(f"Worker pool started with {worker_pool.concurrency_limit} workers")
    except Exception as e:
        logger.error(f"Failed to initialize worker pool: {e}")
        raise
    
    # Start background cleanup task
    cleanup_task = asyncio.create_task(periodic_cleanup())
    
    yield
    
    # Shutdown tasks
    logger.info("Shutting down Gene-Disease Analysis API")
    
    # Stop worker pool
    logger.info("Shutting down worker pool...")
    try:
        await shutdown_worker_pool()
        logger.info("Worker pool shutdown completed")
    except Exception as e:
        logger.error(f"Error shutting down worker pool: {e}")
    
    # Close database
    logger.info("Closing database...")
    try:
        await close_database()
        logger.info("Database closed successfully")
    except Exception as e:
        logger.error(f"Error closing database: {e}")
    
    # Cancel cleanup task
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        logger.info("Cleanup task cancelled")


async def periodic_cleanup():
    """
    Background task for periodic session cleanup
    Removes expired sessions and their associated analyses
    """
    while True:
        try:
            # Wait 1 hour between cleanup cycles
            await asyncio.sleep(3600)
            
            # Clean up expired sessions
            cleaned_count = storage.cleanup_expired_sessions(settings.SESSION_CLEANUP_HOURS)
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} expired sessions")
                
        except asyncio.CancelledError:
            logger.info("Cleanup task cancelled")
            break
        except Exception as e:
            logger.error(f"Error in periodic cleanup: {str(e)}")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    debug=settings.DEBUG,
    lifespan=lifespan
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=settings.ALLOWED_METHODS,
    allow_headers=settings.ALLOWED_HEADERS,
)


# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring and load balancers
    
    Returns:
        dict: Application status and basic statistics
    """
    try:
        # Get worker pool status if available
        from .services.worker_pool import get_pool_status
        worker_status = await get_pool_status()
        worker_health = "healthy" if worker_status.get("running") else "unhealthy"
    except Exception:
        worker_health = "unknown"
        worker_status = {"error": "Worker pool status unavailable"}
    
    # Check database health
    try:
        db_health = await check_database_health()
        db_stats = await get_database_stats()
    except Exception as e:
        db_health = {"status": "unhealthy", "error": str(e)}
        db_stats = {"error": "Database stats unavailable"}
    
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "debug": settings.DEBUG,
        "total_sessions": storage.get_total_sessions(),
        "total_analyses": storage.get_total_analyses(),
        "database": {
            "status": db_health.get("status", "unknown"),
            "stats": db_stats
        },
        "worker_pool": {
            "status": worker_health,
            "details": worker_status
        }
    }


# Root endpoint with API information
@app.get("/")
async def root():
    """
    Root endpoint with API information and available endpoints
    
    Returns:
        dict: API information and navigation
    """
    return {
        "message": "Gene-Disease Analysis API",
        "version": settings.APP_VERSION,
        "documentation": "/docs",
        "health": "/health",
        "endpoints": {
            "session": {
                "create": "POST /session",
                "get": "GET /session/{session_id}",
                "delete": "DELETE /session/{session_id}"
            },
            "analysis": {
                "create": "POST /analysis",
                "get": "GET /analysis/{analysis_id}",
                "stream": "GET /analysis/stream/{analysis_id}",
                "history": "GET /analysis/history/{session_id}"
            },
            "nhs_scotland": {
                "info": "GET /nhs/",
                "datasets": "GET /nhs/datasets",
                "gene_search": "GET /nhs/gene/{gene_name}",
                "disease_search": "GET /nhs/disease/{disease_name}"
            },
            "llm_analysis": {
                "analyze": "POST /llm/analyze",
                "providers": "GET /llm/providers",
                "models": "GET /llm/models/{provider}",
                "cache_stats": "GET /llm/cache/stats"
            },
            "worker_management": {
                "info": "GET /workers/",
                "submit_job": "POST /workers/jobs",
                "job_status": "GET /workers/jobs/{job_id}",
                "cancel_job": "DELETE /workers/jobs/{job_id}",
                "pool_status": "GET /workers/pool",
                "statistics": "GET /workers/stats",
                "health": "GET /workers/health"
            },
            "persistent_analysis": {
                "create_session": "POST /api/v1/persistent/sessions",
                "get_session": "GET /api/v1/persistent/sessions/{session_id}",
                "create_analysis": "POST /api/v1/persistent/analyses",
                "get_analysis": "GET /api/v1/persistent/analyses/{analysis_id}",
                "analysis_history": "GET /api/v1/persistent/sessions/{session_id}/analyses",
                "search_analyses": "GET /api/v1/persistent/analyses/search",
                "session_stats": "GET /api/v1/persistent/sessions/{session_id}/stats",
                "global_stats": "GET /api/v1/persistent/stats/global",
                "health": "GET /api/v1/persistent/health"
            }
        }
    }


# Include routers
app.include_router(session.router)
app.include_router(analysis.router)

# Import NHS Scotland router
from .routers import nhs_scotland
app.include_router(nhs_scotland.router)

# Import LLM Analysis router
from .routers import llm_analysis
app.include_router(llm_analysis.router)

# Import Worker Management router
from .routers import worker_management
app.include_router(worker_management.router)

# Import Persistent Analysis router
from .routers import analysis_persistent
app.include_router(analysis_persistent.router)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler for unhandled errors
    Prevents sensitive information leakage in production
    
    Args:
        request: FastAPI request object
        exc: Exception that occurred
        
    Returns:
        JSONResponse: Error response with appropriate status code
    """
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    if settings.DEBUG:
        # In debug mode, return detailed error information
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error",
                "error": str(exc),
                "type": type(exc).__name__
            }
        )
    else:
        # In production, return generic error message
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )


def main():
    """
    Main entry point for Poetry script
    Used when running: poetry run start
    """
    import uvicorn
    
    logger.info(f"Starting server at http://{settings.HOST}:{settings.PORT}")
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )


if __name__ == "__main__":
    """
    Direct execution entry point for development
    In production, use proper ASGI server like uvicorn
    """
    main()