"""API v1 Routes.

Version 1 of the ML Data Pipeline API.
Includes all domain-specific endpoint routers.
"""

from fastapi import APIRouter

from app.api.v1.endpoints import datasets, models, predictions

# Create v1 router
api_router = APIRouter()

# Include domain routers
api_router.include_router(
    datasets.router,
    prefix="/datasets",
    tags=["datasets"],
)

api_router.include_router(
    models.router,
    prefix="/models",
    tags=["models"],
)

api_router.include_router(
    predictions.router,
    prefix="/predictions",
    tags=["predictions"],
)

__all__ = ["api_router"]
