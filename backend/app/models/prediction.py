"""Prediction model for tracking inference requests and results.

Stores predictions made by deployed models for monitoring and analytics.
"""

from sqlalchemy import Column, Integer, String, Text, Float, ForeignKey
from sqlalchemy.orm import relationship

from app.models.base import Base


class Prediction(Base):
    """Model representing a prediction/inference request.
    
    Tracks predictions for:
    - Model performance monitoring
    - Prediction history and audit trails
    - A/B testing and comparison
    - Usage analytics
    
    Attributes:
        model_id: ML model used for prediction
        input_data: Input features as JSON string
        output_data: Prediction result as JSON string
        confidence_score: Confidence/probability score (0.0-1.0)
        latency_ms: Inference latency in milliseconds
        user_id: User who requested the prediction (optional)
    """
    
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Model reference
    model_id = Column(Integer, ForeignKey("ml_models.id"), nullable=False, index=True)
    
    # Prediction data
    input_data = Column(Text, nullable=False)  # JSON string of input features
    output_data = Column(Text, nullable=False)  # JSON string of prediction result
    
    # Metrics
    confidence_score = Column(Float, nullable=True)  # Confidence/probability
    latency_ms = Column(Integer, nullable=True)  # Inference latency
    
    # User tracking (optional, for analytics)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    
    # Relationships
    model = relationship("MLModel", back_populates="predictions")
    user = relationship("User", back_populates="predictions")
    
    def __repr__(self) -> str:
        return f"<Prediction(id={self.id}, model_id={self.model_id}, confidence={self.confidence_score})>"
