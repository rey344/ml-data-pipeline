"""User model for authentication and authorization.

Manages user accounts, authentication, and access control.
"""

from sqlalchemy import Column, Integer, String, Boolean
from sqlalchemy.orm import relationship

from app.models.base import Base


class User(Base):
    """Model representing a system user.
    
    Handles:
    - User authentication
    - Account management
    - Ownership tracking for resources
    - Access control
    
    Attributes:
        email: Unique email address for login
        username: Display name/username
        hashed_password: Bcrypt hashed password
        full_name: User's full name
        is_active: Account active status
        is_superuser: Admin privileges flag
    """
    
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    full_name = Column(String(255), nullable=True)
    
    # Authentication
    hashed_password = Column(String(255), nullable=False)
    
    # Account status
    is_active = Column(Boolean, default=True, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)
    
    # Relationships - user owns many resources
    datasets = relationship("Dataset", back_populates="owner")
    models = relationship("MLModel", back_populates="owner")
    predictions = relationship("Prediction", back_populates="user")
    
    def __repr__(self) -> str:
        return f"<User(id={self.id}, email='{self.email}', username='{self.username}')>"
