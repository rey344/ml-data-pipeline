"""Prediction and Inference endpoints.

Provides endpoints for making predictions with trained models.
"""
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime
import uuid
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predictions", tags=["predictions"])

# Model storage directory
MODEL_STORAGE_DIR = Path("models")
PREDICTION_HISTORY_DIR = Path("prediction_history")
PREDICTION_HISTORY_DIR.mkdir(exist_ok=True)


class PredictionRequest(BaseModel):
    """Request model for single or batch predictions."""
    model_id: str = Field(..., description="ID of the model to use for predictions")
    data: List[Dict[str, Any]] = Field(..., description="List of data points for prediction")
    return_confidence: bool = Field(default=False, description="Whether to return confidence scores")


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    prediction_id: str
    model_id: str
    num_predictions: int
    predictions: List[Any]
    confidence: Optional[List[float]] = None
    processing_time_ms: float
    timestamp: str


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    model_id: str = Field(..., description="ID of the model to use")
    data: List[Dict[str, Any]] = Field(..., description="Batch of data points")
    batch_size: int = Field(default=32, ge=1, le=10000, description="Batch size for processing")


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    batch_id: str
    model_id: str
    total_samples: int
    successful: int
    failed: int
    predictions: List[Any]
    errors: Optional[List[Dict[str, Any]]] = None
    processing_time_ms: float


class PredictionHistory(BaseModel):
    """Model for prediction history tracking."""
    prediction_id: str
    model_id: str
    timestamp: str
    num_samples: int
    processing_time_ms: float


@router.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict(request: PredictionRequest) -> dict:
    """Make predictions using a trained model.
    
    Takes a list of data points and returns predictions from the specified model.
    """
    import time
    start_time = time.time()
    
    try:
        # Validate model exists
        model_path = MODEL_STORAGE_DIR / f"{request.model_id}.pkl"
        if not model_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {request.model_id} not found"
            )
        
        # Validate input data
        if not request.data or len(request.data) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No data provided for prediction"
            )
        
        # Load the model
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load model {request.model_id}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load model: {str(e)}"
            )
        
        # Convert data to DataFrame
        try:
            df = pd.DataFrame(request.data)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid data format: {str(e)}"
            )
        
        # Make predictions
        try:
            predictions = model.predict(df).tolist()
            
            confidence = None
            if request.return_confidence and hasattr(model, "predict_proba"):
                try:
                    confidence_scores = model.predict_proba(df)
                    # Get max confidence for each prediction
                    confidence = [float(max(scores)) for scores in confidence_scores]
                except Exception as e:
                    logger.warning(f"Could not compute confidence scores: {str(e)}")
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prediction failed: {str(e)}"
            )
        
        # Record prediction
        processing_time = (time.time() - start_time) * 1000
        prediction_id = f"pred_{uuid.uuid4().hex[:12]}"
        
        # Save to history
        history_entry = {
            "prediction_id": prediction_id,
            "model_id": request.model_id,
            "timestamp": datetime.now().isoformat(),
            "num_samples": len(predictions),
            "processing_time_ms": processing_time,
        }
        
        return {
            "prediction_id": prediction_id,
            "model_id": request.model_id,
            "num_predictions": len(predictions),
            "predictions": predictions,
            "confidence": confidence,
            "processing_time_ms": processing_time,
            "timestamp": datetime.now().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )


@router.post("/batch-predict", response_model=BatchPredictionResponse, status_code=status.HTTP_200_OK)
async def batch_predict(request: BatchPredictionRequest) -> dict:
    """Make batch predictions using a trained model.
    
    Processes large batches of data efficiently with configurable batch size.
    """
    import time
    start_time = time.time()
    
    try:
        # Validate model
        model_path = MODEL_STORAGE_DIR / f"{request.model_id}.pkl"
        if not model_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {request.model_id} not found"
            )
        
        # Load model
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        # Convert to DataFrame
        df = pd.DataFrame(request.data)
        
        # Process in batches
        all_predictions = []
        errors = []
        successful = 0
        
        for i in range(0, len(df), request.batch_size):
            try:
                batch = df.iloc[i:i + request.batch_size]
                batch_predictions = model.predict(batch).tolist()
                all_predictions.extend(batch_predictions)
                successful += len(batch_predictions)
            except Exception as e:
                error_info = {
                    "batch_index": i // request.batch_size,
                    "error": str(e),
                    "samples_count": len(batch) if 'batch' in locals() else 0,
                }
                errors.append(error_info)
                logger.error(f"Batch {i // request.batch_size} prediction failed: {str(e)}")
        
        processing_time = (time.time() - start_time) * 1000
        batch_id = f"batch_{uuid.uuid4().hex[:12]}"
        
        return {
            "batch_id": batch_id,
            "model_id": request.model_id,
            "total_samples": len(df),
            "successful": successful,
            "failed": len(df) - successful,
            "predictions": all_predictions,
            "errors": errors if errors else None,
            "processing_time_ms": processing_time,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@router.get("/models/{model_id}/info")
async def get_model_prediction_info(model_id: str) -> dict:
    """Get information about a model's prediction capabilities."""
    try:
        model_path = MODEL_STORAGE_DIR / f"{model_id}.pkl"
        if not model_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {model_id} not found"
            )
        
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        # Get model info
        model_type = type(model).__name__
        has_proba = hasattr(model, "predict_proba")
        has_decision_function = hasattr(model, "decision_function")
        
        # Get expected input shape if available
        expected_features = None
        if hasattr(model, "n_features_in_"):
            expected_features = int(model.n_features_in_)
        
        return {
            "model_id": model_id,
            "model_type": model_type,
            "has_probability_estimates": has_proba,
            "has_decision_function": has_decision_function,
            "expected_input_features": expected_features,
            "supports_batch_prediction": True,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/health")
async def prediction_service_health() -> dict:
    """Check the health of the prediction service."""
    try:
        models_available = 0
        if MODEL_STORAGE_DIR.exists():
            models_available = len(list(MODEL_STORAGE_DIR.glob("*.pkl")))
        
        return {
            "status": "healthy",
            "service": "prediction",
            "models_available": models_available,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unavailable: {str(e)}"
        )
