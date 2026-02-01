"""Dataset service.

Handles dataset operations including upload, retrieval, and processing.
"""
from typing import List, Optional
import io
import pandas as pd
from fastapi import UploadFile, HTTPException, status


class DatasetService:
    """Service for managing dataset operations."""
    
    ALLOWED_FORMATS = {"csv", "json", "parquet", "xlsx"}
    MAX_FILE_SIZE = 5 * 1024 * 1024 * 1024  # 5GB
    
    @staticmethod
    async def validate_upload(file: UploadFile) -> None:
        """Validate uploaded file."""
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Filename is required",
            )
        
        file_ext = file.filename.split(".")[-1].lower()
        if file_ext not in DatasetService.ALLOWED_FORMATS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File format .{file_ext} is not supported. Allowed: {DatasetService.ALLOWED_FORMATS}",
            )
    
    @staticmethod
    async def read_file(file: UploadFile) -> pd.DataFrame:
        """Read uploaded file and return DataFrame."""
        contents = await file.read()
        file_ext = file.filename.split(".")[-1].lower()
        
        try:
            if file_ext == "csv":
                return pd.read_csv(io.BytesIO(contents))
            elif file_ext == "json":
                return pd.read_json(io.BytesIO(contents))
            elif file_ext == "parquet":
                return pd.read_parquet(io.BytesIO(contents))
            elif file_ext == "xlsx":
                return pd.read_excel(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to read file: {str(e)}",
            )
    
    @staticmethod
    def get_dataset_stats(df: pd.DataFrame) -> dict:
        """Get statistical summary of dataset."""
        return {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "data_types": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "basic_stats": df.describe().to_dict(),
        }
