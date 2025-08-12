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
    
    # NHS Scotland Open Data API Configuration
    NHS_SCOTLAND_BASE_URL: str = "https://www.opendata.nhs.scot/api/3"
    NHS_SCOTLAND_TIMEOUT_SECONDS: int = int(os.getenv("NHS_TIMEOUT", 30))
    
    # Cache Configuration
    NHS_DATA_CACHE_TTL_SECONDS: int = int(os.getenv("NHS_CACHE_TTL", 3600))  # 1 hour default
    
    # LLM API Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_ORGANIZATION: str = os.getenv("OPENAI_ORGANIZATION", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    
    # LLM Settings
    DEFAULT_LLM_PROVIDER: str = os.getenv("DEFAULT_LLM_PROVIDER", "openai")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", 0.1))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", 2000))
    LLM_ENABLE_CACHING: bool = os.getenv("LLM_ENABLE_CACHING", "true").lower() == "true"
    LLM_CACHE_TTL: int = int(os.getenv("LLM_CACHE_TTL", 3600))  # 1 hour
    
    # Worker Pool Configuration
    LLM_CONCURRENCY: int = int(os.getenv("LLM_CONCURRENCY", 4))
    LLM_QUEUE_SIZE: int = int(os.getenv("LLM_QUEUE_SIZE", 100))
    LLM_JOB_TIMEOUT: float = float(os.getenv("LLM_JOB_TIMEOUT", 300.0))  # 5 minutes
    LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", 3))
    
    # Retry Configuration
    LLM_RETRY_BASE_DELAY: float = float(os.getenv("LLM_RETRY_BASE_DELAY", 1.0))
    LLM_RETRY_MAX_DELAY: float = float(os.getenv("LLM_RETRY_MAX_DELAY", 60.0))
    
    # Rate Limiting Configuration
    OPENAI_RATE_LIMIT_CAPACITY: int = int(os.getenv("OPENAI_RATE_LIMIT_CAPACITY", 60))
    OPENAI_RATE_LIMIT_REFILL: float = float(os.getenv("OPENAI_RATE_LIMIT_REFILL", 1.0))
    ANTHROPIC_RATE_LIMIT_CAPACITY: int = int(os.getenv("ANTHROPIC_RATE_LIMIT_CAPACITY", 50))
    ANTHROPIC_RATE_LIMIT_REFILL: float = float(os.getenv("ANTHROPIC_RATE_LIMIT_REFILL", 0.8))
    
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