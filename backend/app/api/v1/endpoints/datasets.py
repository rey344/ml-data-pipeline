"""Dataset management endpoints.

Provides endpoints for dataset upload, retrieval, exploration, and statistics.
"""
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from pydantic import BaseModel, Field
import pandas as pd
import json
from pathlib import Path
import shutil
from datetime import datetime
import uuid

# Import DatasetService
try:
    from app.services.dataset import DatasetService
except ImportError:
    from ...services.dataset import DatasetService

router = APIRouter(prefix="/datasets", tags=["datasets"])

# Dataset storage directory
DATASET_STORAGE_DIR = Path("datasets")
DATASET_STORAGE_DIR.mkdir(exist_ok=True)

# Metadata storage for datasets
METADATA_FILE = DATASET_STORAGE_DIR / "metadata.json"


class DatasetResponse(BaseModel):
    """Response model for dataset information."""
    id: str
    name: str
    filename: str
    size: int
    rows: int
    columns: int
    created_at: str
    file_format: str
    column_names: List[str] = []


class DatasetUploadResponse(BaseModel):
    """Response for dataset upload."""
    dataset_id: str
    message: str
    filename: str
    size: int
    rows: int
    columns: int


class DatasetStatistics(BaseModel):
    """Dataset statistics response."""
    dataset_id: str
    total_rows: int
    total_columns: int
    column_info: List[Dict[str, Any]]
    memory_usage: str
    missing_values: Dict[str, int]


class ColumnStatistics(BaseModel):
    """Column-level statistics."""
    name: str
    dtype: str
    missing: int
    unique: int
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None


def _load_metadata() -> Dict[str, Any]:
    """Load metadata from file."""
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r") as f:
            return json.load(f)
    return {}


def _save_metadata(metadata: Dict[str, Any]):
    """Save metadata to file."""
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)


@router.post("/upload", response_model=DatasetUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    file: UploadFile = File(..., description="CSV, JSON, or Parquet file"),
) -> dict:
    """Upload a new dataset.
    
    Supported formats: CSV, JSON, Parquet
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File name is required"
            )
        
        # Validate file extension
        allowed_extensions = {".csv", ".json", ".parquet", ".xlsx"}
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File format not supported. Allowed: {allowed_extensions}"
            )
        
        # Generate dataset ID
        dataset_id = f"dataset_{uuid.uuid4().hex[:12]}"
        
        # Save file
        dataset_path = DATASET_STORAGE_DIR / f"{dataset_id}{file_ext}"
        file_size = 0
        
        with open(dataset_path, "wb") as f:
            contents = await file.read()
            file_size = len(contents)
            f.write(contents)
        
        # Load and analyze dataset
        try:
            if file_ext == ".csv":
                df = pd.read_csv(dataset_path)
            elif file_ext == ".json":
                df = pd.read_json(dataset_path)
            elif file_ext == ".parquet":
                df = pd.read_parquet(dataset_path)
            elif file_ext == ".xlsx":
                df = pd.read_excel(dataset_path)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
        except Exception as e:
            dataset_path.unlink()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to parse file: {str(e)}"
            )
        
        # Update metadata
        metadata = _load_metadata()
        metadata[dataset_id] = {
            "id": dataset_id,
            "name": file.filename,
            "filename": file.filename,
            "file_format": file_ext[1:].upper(),
            "size": file_size,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "created_at": datetime.now().isoformat(),
            "path": str(dataset_path),
        }
        _save_metadata(metadata)
        
        return {
            "dataset_id": dataset_id,
            "message": "Dataset uploaded successfully",
            "filename": file.filename,
            "size": file_size,
            "rows": len(df),
            "columns": len(df.columns),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}"
        )


@router.get("/", response_model=Dict[str, Any])
async def list_datasets() -> dict:
    """List all uploaded datasets."""
    try:
        metadata = _load_metadata()
        datasets = list(metadata.values())
        
        return {
            "datasets": datasets,
            "total": len(datasets),
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(dataset_id: str) -> dict:
    """Get dataset details and metadata."""
    try:
        metadata = _load_metadata()
        if dataset_id not in metadata:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset {dataset_id} not found"
            )
        
        dataset_info = metadata[dataset_id]
        return DatasetResponse(**dataset_info)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/{dataset_id}/stats", response_model=DatasetStatistics)
async def get_dataset_statistics(dataset_id: str) -> dict:
    """Get comprehensive dataset statistics."""
    try:
        metadata = _load_metadata()
        if dataset_id not in metadata:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset {dataset_id} not found"
            )
        
        dataset_info = metadata[dataset_id]
        dataset_path = Path(dataset_info["path"])
        
        # Load dataset
        if dataset_info["file_format"] == "CSV":
            df = pd.read_csv(dataset_path)
        elif dataset_info["file_format"] == "JSON":
            df = pd.read_json(dataset_path)
        elif dataset_info["file_format"] == "PARQUET":
            df = pd.read_parquet(dataset_path)
        else:
            df = pd.read_excel(dataset_path)
        
        # Calculate statistics
        column_info = []
        for col in df.columns:
            col_stats = {
                "name": col,
                "dtype": str(df[col].dtype),
                "missing": int(df[col].isna().sum()),
                "unique": int(df[col].nunique()),
            }
            
            # Numeric statistics
            if pd.api.types.is_numeric_dtype(df[col]):
                col_stats.update({
                    "mean": float(df[col].mean()),
                    "median": float(df[col].median()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                })
            
            column_info.append(col_stats)
        
        return {
            "dataset_id": dataset_id,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "column_info": column_info,
            "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB",
            "missing_values": df.isna().sum().to_dict(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/{dataset_id}/preview")
async def preview_dataset(dataset_id: str, rows: int = 10) -> dict:
    """Get a preview of the dataset (first N rows)."""
    try:
        metadata = _load_metadata()
        if dataset_id not in metadata:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset {dataset_id} not found"
            )
        
        dataset_info = metadata[dataset_id]
        dataset_path = Path(dataset_info["path"])
        
        # Load dataset
        if dataset_info["file_format"] == "CSV":
            df = pd.read_csv(dataset_path)
        elif dataset_info["file_format"] == "JSON":
            df = pd.read_json(dataset_path)
        elif dataset_info["file_format"] == "PARQUET":
            df = pd.read_parquet(dataset_path)
        else:
            df = pd.read_excel(dataset_path)
        
        # Get preview
        preview_data = df.head(rows).to_dict(orient="records")
        
        return {
            "dataset_id": dataset_id,
            "preview": preview_data,
            "rows_shown": len(preview_data),
            "total_rows": len(df),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(dataset_id: str):
    """Delete a dataset and its metadata."""
    try:
        metadata = _load_metadata()
        if dataset_id not in metadata:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset {dataset_id} not found"
            )
        
        # Delete file
        dataset_path = Path(metadata[dataset_id]["path"])
        if dataset_path.exists():
            dataset_path.unlink()
        
        # Remove from metadata
        del metadata[dataset_id]
        _save_metadata(metadata)
        
        return None
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
