"""Base model for all database models.

Provides common functionality and declarative base for SQLAlchemy ORM.
"""

from datetime import datetime
from typing import Any

from sqlalchemy import Column, DateTime
from sqlalchemy.ext.declarative import as_declarative, declared_attr


@as_declarative()
class Base:
    """Base class for all database models.
    
    Provides:
    - Automatic table name generation from class name
    - Common timestamp fields (created_at, updated_at)
    - Dictionary conversion method
    """
    
    id: Any
    __name__: str
    
    # Generate __tablename__ automatically
    @declared_attr
    def __tablename__(cls) -> str:
        """Generate table name from class name in snake_case."""
        import re
        name = cls.__name__
        # Convert CamelCase to snake_case
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
    
    # Common timestamp columns
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )
    
    def dict(self) -> dict:
        """Convert model instance to dictionary.
        
        Returns:
            Dictionary representation of the model with all columns.
        """
        return {
            column.name: getattr(self, column.name)
            for column in self.__table__.columns
        }
