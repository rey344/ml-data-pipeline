"""Pydantic schemas for Dataset API requests and responses.

Defines validation models for dataset operations.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, validator

from app.models.dataset import DatasetType, DatasetStatus


class DatasetBase(BaseModel):
    """Base dataset schema with common fields."""
    
    name: str = Field(..., min_length=1, max_length=255, description="Dataset name")
    description: Optional[str] = Field(None, description="Detailed description")
    dataset_type: DatasetType = Field(..., description="Type of dataset")


class DatasetCreate(DatasetBase):
    """Schema for creating a new dataset."""
    
    file_path: str = Field(..., description="Path or URL to dataset file")
    file_size: int = Field(..., gt=0, description="File size in bytes")
    num_rows: Optional[int] = Field(None, ge=0, description="Number of rows")
    num_features: Optional[int] = Field(None, ge=0, description="Number of features")
    metadata: Optional[str] = Field(None, description="Additional metadata as JSON string")


class DatasetUpdate(BaseModel):
    """Schema for updating dataset fields."""
    
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    status: Optional[DatasetStatus] = None
    num_rows: Optional[int] = Field(None, ge=0)
    num_features: Optional[int] = Field(None, ge=0)
    metadata: Optional[str] = None


class DatasetResponse(DatasetBase):
    """Schema for dataset API responses."""
    
    id: int
    status: DatasetStatus
    file_path: str
    file_size: int
    num_rows: Optional[int]
    num_features: Optional[int]
    owner_id: int
    metadata: Optional[str]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True  # Allow creation from ORM models


class DatasetListResponse(BaseModel):
    """Schema for paginated dataset list."""
    
    items: list[DatasetResponse]
    total: int
    page: int
    page_size: int
    
    class Config:
        orm_mode = True
