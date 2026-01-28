"""Dataset model for managing ML training and testing data.

Tracks uploaded datasets, their metadata, and storage locations.
"""

from enum import Enum
from typing import Optional

from sqlalchemy import Column, Integer, String, Text, Enum as SQLEnum, BigInteger, ForeignKey
from sqlalchemy.orm import relationship

from app.models.base import Base


class DatasetType(str, Enum):
    """Types of datasets supported by the platform."""
    
    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"
    NUMPY = "numpy"
    IMAGE = "image"
    TEXT = "text"


class DatasetStatus(str, Enum):
    """Processing status of uploaded datasets."""
    
    UPLOADING = "uploading"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"
    ARCHIVED = "archived"


class Dataset(Base):
    """Model representing a dataset for ML training/testing.
    
    Attributes:
        name: Human-readable dataset name
        description: Optional detailed description
        dataset_type: Type of dataset (CSV, JSON, etc.)
        status: Current processing status
        file_path: Storage path or URL to dataset file
        file_size: Size in bytes
        num_rows: Number of records/samples
        num_features: Number of features/columns
        owner_id: ID of user who uploaded the dataset
        metadata: Additional JSON metadata (schemas, stats, etc.)
    """
    
    __tablename__ = "datasets"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    
    # Dataset type and status
    dataset_type = Column(SQLEnum(DatasetType), nullable=False)
    status = Column(
        SQLEnum(DatasetStatus),
        nullable=False,
        default=DatasetStatus.UPLOADING,
        index=True,
    )
    
    # File information
    file_path = Column(String(512), nullable=False)
    file_size = Column(BigInteger, nullable=False)
    
    # Dataset dimensions
    num_rows = Column(Integer, nullable=True)
    num_features = Column(Integer, nullable=True)
    
    # Ownership
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
    # Additional metadata as JSON
    metadata = Column(Text, nullable=True)  # Store as JSON string
    
    # Relationships
    owner = relationship("User", back_populates="datasets")
    models = relationship("MLModel", back_populates="dataset")
    
    def __repr__(self) -> str:
        return f"<Dataset(id={self.id}, name='{self.name}', status={self.status})>"
