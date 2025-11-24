from dataclasses import dataclass
from starlette import status

@dataclass(frozen=True)
class APIError:
    http_status: int
    code: str
    message: str
    
EMPTY_TEXT = APIError(
    http_status=status.HTTP_400_BAD_REQUEST,
    code="EMPTY_TEXT",
    message="Text for synthesis is empty."
    )

SYNTHESIS_FAILED = APIError(
    http_status=status.HTTP_500_INTERNAL_SERVER_ERROR,
    code="SYNTHESIS_FAILED",
    message="Internal server error during synthesis."
    )