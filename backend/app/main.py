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
    
    # Start background cleanup task
    cleanup_task = asyncio.create_task(periodic_cleanup())
    
    yield
    
    # Shutdown tasks
    logger.info("Shutting down Gene-Disease Analysis API")
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
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "debug": settings.DEBUG,
        "total_sessions": storage.get_total_sessions(),
        "total_analyses": storage.get_total_analyses()
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
            }
        }
    }


# Include routers
app.include_router(session.router)
app.include_router(analysis.router)


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