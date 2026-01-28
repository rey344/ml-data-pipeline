"""ML Model and ModelVersion models for model registry.

Tracks trained models, their versions, and associated metadata.
"""

from enum import Enum
from typing import Optional

from sqlalchemy import Column, Integer, String, Text, Float, Enum as SQLEnum, ForeignKey
from sqlalchemy.orm import relationship

from app.models.base import Base


class ModelType(str, Enum):
    """Types of ML models supported."""
    
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    NLP = "nlp"
    COMPUTER_VISION = "computer_vision"
    TIME_SERIES = "time_series"
    RECOMMENDATION = "recommendation"


class ModelStatus(str, Enum):
    """Training and deployment status of models."""
    
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"


class MLModel(Base):
    """Model representing a trained ML model.
    
    Attributes:
        name: Model name
        description: Detailed description
        model_type: Type of ML model
        dataset_id: Associated training dataset
        owner_id: User who created the model
    """
    
    __tablename__ = "ml_models"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    model_type = Column(SQLEnum(ModelType), nullable=False, index=True)
    
    # Relationships
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False, index=True)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
    # Relationships
    dataset = relationship("Dataset", back_populates="models")
    owner = relationship("User", back_populates="models")
    versions = relationship("ModelVersion", back_populates="model", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="model")
    
    def __repr__(self) -> str:
        return f"<MLModel(id={self.id}, name='{self.name}', type={self.model_type})>"


class ModelVersion(Base):
    """Model version with training metrics and artifacts.
    
    Attributes:
        model_id: Parent model
        version: Version number (e.g., "1.0.0", "2.1.3")
        status: Training/deployment status
        algorithm: Algorithm used (e.g., "Random Forest", "XGBoost")
        hyperparameters: JSON string of hyperparameters
        metrics: Training metrics (accuracy, loss, etc.) as JSON
        artifact_path: Path to serialized model file
        file_size: Size of model artifact in bytes
    """
    
    __tablename__ = "model_versions"
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("ml_models.id"), nullable=False, index=True)
    version = Column(String(50), nullable=False, index=True)
    
    status = Column(
        SQLEnum(ModelStatus),
        nullable=False,
        default=ModelStatus.TRAINING,
        index=True,
    )
    
    # Model details
    algorithm = Column(String(100), nullable=True)
    hyperparameters = Column(Text, nullable=True)  # JSON string
    metrics = Column(Text, nullable=True)  # JSON string with training metrics
    
    # Artifact storage
    artifact_path = Column(String(512), nullable=True)
    file_size = Column(Integer, nullable=True)
    
    # Relationships
    model = relationship("MLModel", back_populates="versions")
    
    def __repr__(self) -> str:
        return f"<ModelVersion(id={self.id}, model_id={self.model_id}, version='{self.version}', status={self.status})>"
