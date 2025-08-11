"""
Configuration settings for the FastAPI application
Contains environment-specific settings and constants
"""

import os
from typing import List


class Settings:
    """
    Application configuration settings
    Designed for easy environment-based configuration
    """
    
    # Application Info
    APP_NAME: str = "Gene-Disease Analysis API"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "RESTful API for gene-disease relationship analysis"
    
    # Server Configuration
    HOST: str = os.getenv("HOST", "localhost")
    PORT: int = int(os.getenv("PORT", 8000))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # CORS Configuration
    # Allow frontend at localhost:3000 and other common development ports
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",  # React development server
        "http://localhost:3001",  # Alternative React port
        "http://127.0.0.1:3000",  # Alternative localhost format
        "http://127.0.0.1:3001",
    ]
    
    # Add production origins from environment if specified
    if os.getenv("ALLOWED_ORIGINS"):
        ALLOWED_ORIGINS.extend(
            origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "").split(",")
        )
    
    # CORS Methods and Headers
    ALLOWED_METHODS: List[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    ALLOWED_HEADERS: List[str] = ["*"]
    
    # API Configuration
    API_V1_PREFIX: str = "/api/v1"
    
    # Session Configuration
    SESSION_CLEANUP_HOURS: int = int(os.getenv("SESSION_CLEANUP_HOURS", 24))
    
    # Analysis Configuration
    MAX_CONCURRENT_ANALYSES: int = int(os.getenv("MAX_CONCURRENT_ANALYSES", 10))
    ANALYSIS_TIMEOUT_SECONDS: int = int(os.getenv("ANALYSIS_TIMEOUT_SECONDS", 300))
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Security Configuration
    # In production, these would come from secure environment variables
    ALLOWED_API_KEY_PREFIXES: List[str] = ["sk-", "sk-ant-", "anthropic-"]
    
    # Rate Limiting (for future implementation)
    REQUESTS_PER_MINUTE: int = int(os.getenv("REQUESTS_PER_MINUTE", 60))
    
    # Database Configuration (for future database integration)
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return os.getenv("ENVIRONMENT", "development").lower() == "production"
    
    @property
    def log_config(self) -> dict:
        """Logging configuration dictionary"""
        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s",
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default" if not self.DEBUG else "detailed",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {
                "level": self.LOG_LEVEL,
                "handlers": ["default"],
            },
        }


# Global settings instance
settings = Settings()