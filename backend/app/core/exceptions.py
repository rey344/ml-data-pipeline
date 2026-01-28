"""Custom Exception Classes.

Domain-specific exceptions for the ML Data Pipeline application.
All custom exceptions inherit from AppException for consistent handling.
"""

from typing import Any, Dict, Optional


class AppException(Exception):
    """Base exception for all application errors.
    
    Attributes:
        detail: Human-readable error message
        error_code: Machine-readable error code
        status_code: HTTP status code
        extra: Additional error context
    """

    def __init__(
        self,
        detail: str,
        error_code: str = "APP_ERROR",
        status_code: int = 500,
        extra: Optional[Dict[str, Any]] = None,
    ):
        self.detail = detail
        self.error_code = error_code
        self.status_code = status_code
        self.extra = extra or {}
        super().__init__(self.detail)


class ValidationError(AppException):
    """Raised when data validation fails."""

    def __init__(
        self,
        detail: str = "Validation error",
        extra: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            detail=detail,
            error_code="VALIDATION_ERROR",
            status_code=422,
            extra=extra,
        )


class NotFoundError(AppException):
    """Raised when a requested resource is not found."""

    def __init__(
        self,
        resource: str = "Resource",
        resource_id: Optional[Any] = None,
    ):
        detail = f"{resource} not found"
        if resource_id:
            detail += f": {resource_id}"
        super().__init__(
            detail=detail,
            error_code="NOT_FOUND",
            status_code=404,
        )


class AuthenticationError(AppException):
    """Raised when authentication fails."""

    def __init__(
        self,
        detail: str = "Authentication failed",
    ):
        super().__init__(
            detail=detail,
            error_code="AUTHENTICATION_ERROR",
            status_code=401,
        )


class AuthorizationError(AppException):
    """Raised when user lacks required permissions."""

    def __init__(
        self,
        detail: str = "Insufficient permissions",
        required_permission: Optional[str] = None,
    ):
        extra = {}
        if required_permission:
            extra["required_permission"] = required_permission
        super().__init__(
            detail=detail,
            error_code="AUTHORIZATION_ERROR",
            status_code=403,
            extra=extra,
        )


class DuplicateError(AppException):
    """Raised when attempting to create a duplicate resource."""

    def __init__(
        self,
        resource: str = "Resource",
        field: Optional[str] = None,
    ):
        detail = f"{resource} already exists"
        if field:
            detail += f" with this {field}"
        super().__init__(
            detail=detail,
            error_code="DUPLICATE_ERROR",
            status_code=409,
        )


class DataProcessingError(AppException):
    """Raised when data processing fails."""

    def __init__(
        self,
        detail: str = "Data processing failed",
        stage: Optional[str] = None,
    ):
        extra = {}
        if stage:
            extra["processing_stage"] = stage
        super().__init__(
            detail=detail,
            error_code="DATA_PROCESSING_ERROR",
            status_code=500,
            extra=extra,
        )


class ModelTrainingError(AppException):
    """Raised when model training fails."""

    def __init__(
        self,
        detail: str = "Model training failed",
        model_type: Optional[str] = None,
    ):
        extra = {}
        if model_type:
            extra["model_type"] = model_type
        super().__init__(
            detail=detail,
            error_code="MODEL_TRAINING_ERROR",
            status_code=500,
            extra=extra,
        )


class ModelNotFoundError(NotFoundError):
    """Raised when a model is not found."""

    def __init__(self, model_id: Any):
        super().__init__(resource="Model", resource_id=model_id)


class DatasetNotFoundError(NotFoundError):
    """Raised when a dataset is not found."""

    def __init__(self, dataset_id: Any):
        super().__init__(resource="Dataset", resource_id=dataset_id)


class FileSizeExceededError(ValidationError):
    """Raised when uploaded file exceeds size limit."""

    def __init__(self, max_size_mb: int, actual_size_mb: float):
        super().__init__(
            detail=f"File size ({actual_size_mb:.2f}MB) exceeds maximum allowed ({max_size_mb}MB)",
            extra={
                "max_size_mb": max_size_mb,
                "actual_size_mb": actual_size_mb,
            },
        )


class UnsupportedFileTypeError(ValidationError):
    """Raised when file type is not supported."""

    def __init__(self, file_type: str, supported_types: list):
        super().__init__(
            detail=f"Unsupported file type: {file_type}",
            extra={
                "file_type": file_type,
                "supported_types": supported_types,
            },
        )


class ConcurrentJobLimitError(AppException):
    """Raised when maximum concurrent jobs limit is reached."""

    def __init__(self, current_jobs: int, max_jobs: int):
        super().__init__(
            detail=f"Maximum concurrent jobs limit reached ({current_jobs}/{max_jobs})",
            error_code="CONCURRENT_JOB_LIMIT",
            status_code=429,
            extra={
                "current_jobs": current_jobs,
                "max_jobs": max_jobs,
            },
        )
