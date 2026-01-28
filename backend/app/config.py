"""Application Configuration.

Centralized configuration management using Pydantic settings with
environment variable support and validation.
"""

import json
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support.
    
    All settings can be overridden via environment variables.
    Example: DATABASE_URL, DEBUG, SECRET_KEY, etc.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    APP_NAME: str = "ML Data Pipeline"
    VERSION: str = "0.1.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Security
    SECRET_KEY: str = Field(
        default="changeme-in-production",
        description="Secret key for JWT and sessions",
    )
    ALLOWED_HOSTS: List[str] = Field(
        default=["localhost", "127.0.0.1"],
        description="Allowed hosts for CORS",
    )
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000"],
        description="Allowed CORS origins",
    )

    # Database
    DATABASE_URL: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/ml_pipeline",
        description="PostgreSQL database connection string",
    )
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_ECHO: bool = False  # SQL query logging

    # Redis
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection string",
    )

    # JWT
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_HOURS: int = 24
    REFRESH_TOKEN_EXPIRATION_DAYS: int = 7

    # File Storage
    MAX_FILE_SIZE_MB: int = 5000  # 5GB
    UPLOAD_DIR: str = "./uploads"
    MODEL_STORAGE_DIR: str = "./models"

    # Machine Learning
    MAX_TRAINING_JOBS: int = 5
    DEFAULT_TEST_SPLIT: float = 0.2
    MAX_MODEL_SIZE_MB: int = 500

    # Celery
    CELERY_BROKER_URL: Optional[str] = None
    CELERY_RESULT_BACKEND: Optional[str] = None

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"

    # Email (Optional)
    SMTP_SERVER: Optional[str] = None
    SMTP_PORT: Optional[int] = None
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    SMTP_FROM_EMAIL: Optional[str] = None

    # AWS (Optional)
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_S3_BUCKET: Optional[str] = None
    AWS_REGION: str = "us-east-1"

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: any) -> List[str]:
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return [origin.strip() for origin in v.split(",")]
        return v

    @field_validator("CELERY_BROKER_URL", mode="before")
    @classmethod
    def default_celery_broker(cls, v: Optional[str], info) -> str:
        """Default Celery broker to Redis URL if not set."""
        if v is None:
            return info.data.get("REDIS_URL", "redis://localhost:6379/0")
        return v

    @field_validator("CELERY_RESULT_BACKEND", mode="before")
    @classmethod
    def default_celery_backend(cls, v: Optional[str], info) -> str:
        """Default Celery result backend to Redis URL with different DB."""
        if v is None:
            redis_url = info.data.get("REDIS_URL", "redis://localhost:6379/0")
            # Change database number for result backend
            return redis_url.rsplit("/", 1)[0] + "/1"
        return v

    def get_database_url(self, async_driver: bool = True) -> str:
        """Get database URL with optional async driver.
        
        Args:
            async_driver: Whether to use asyncpg driver
            
        Returns:
            Database URL string
        """
        if async_driver and self.DATABASE_URL.startswith("postgresql://"):
            return self.DATABASE_URL.replace(
                "postgresql://", "postgresql+asyncpg://"
            )
        return self.DATABASE_URL


# Global settings instance
settings = Settings()
