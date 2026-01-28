"""API Router Configuration.

Central router that includes all API endpoint modules.
Provides versioned API structure and route organization.
"""

from fastapi import APIRouter

from app.api.v1 import api_router as v1_router

# Main API router
router = APIRouter()

# Include v1 API routes
router.include_router(v1_router, prefix="/v1", tags=["v1"])

# Future API versions can be added here
# router.include_router(v2_router, prefix="/v2", tags=["v2"])

__all__ = ["router"]
