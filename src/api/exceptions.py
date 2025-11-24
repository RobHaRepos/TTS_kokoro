from typing import Optional
from fastapi import HTTPException
from src.api.errors import APIError

def raise_api_error(api_error: APIError, details: Optional[str] = None) -> HTTPException:
    """Create and raise an HTTPException based on the given APIError."""
    payload = {
        "code": api_error.code,
        "message": api_error.message,
        "details": details,
    }
    raise HTTPException(status_code=api_error.http_status, detail=payload)