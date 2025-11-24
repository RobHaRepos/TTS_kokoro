import asyncio
import os
import io
import logging
import soundfile
from typing import Any
import torch

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, field_validator
from contextlib import asynccontextmanager

from ..synthesize_TTS import synthesize_text, get_pipeline
from ..log_handler import HTTPLogHandler

from .errors import EMPTY_TEXT, SYNTHESIS_FAILED
from .exceptions import raise_api_error


VOICE = os.getenv("VOICE", "am_onyx")
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", 24000))
SPEED = float(os.getenv("SPEED", 1.0))
LANG_CODE = os.getenv("LANG_CODE", "a")
DEVICE = os.getenv("DEVICE", "cuda")
LOGGER_SERVICE_URL = os.getenv("LOGGER_SERVICE_URL", "http://logger_service:8004")

logger = logging.getLogger("synthesize_service")
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
remote = HTTPLogHandler(LOGGER_SERVICE_URL)
logger.addHandler(remote)
logger.setLevel(logging.INFO)


class SynthesizeRequest(BaseModel):
    """Request model for synthesis endpoint with validation."""
    text: str
    voice: str = VOICE
    speed: float = SPEED

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        if not v.strip():
            raise_api_error(EMPTY_TEXT)
        return v.strip()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize Kokoro TTS pipeline on startup."""
    logger.info(f"Initializing Kokoro TTS pipeline with device: {DEVICE}")
    get_pipeline(lang_code=LANG_CODE, repo_id='hexgrad/Kokoro-82M', device=DEVICE)
    try:
        yield
    finally:
        logger.info("Shutting down Kokoro TTS API.")

app = FastAPI(title="Kokoro TTS API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    )   

@app.exception_handler(HTTPException)
async def http_exception_handler(request : Request, exc: HTTPException):
    """Custom handler for HTTPException to format API errors."""
    detail: Any = exc.detail    
    if isinstance(detail, dict):
        code = detail.get("code", "HTTP_ERROR")
        message = detail.get("message", "")
        details = detail.get("details", None)
    else:
        code = "HTTP_ERROR"
        message = str(detail)
        details = None
    
    return JSONResponse(
        status_code=exc.status_code,
        content={"code": code, "message": message, "details": details}
    )

synthesize_lock = asyncio.Lock()    

@app.post("/synthesize", response_class=StreamingResponse)
async def synthesize(request: SynthesizeRequest):
    """Synthesize speech from text."""
    text = request.text
    audio = None
    logger.info(f"Synthesized text request received: {text}")
    
    async with synthesize_lock:
        try:
            audio, _, _ = await asyncio.to_thread(synthesize_text, text, voice=request.voice, speed=request.speed)
        except Exception as e:
            logger.error(f"Error during synthesis: {e}")
            raise_api_error(SYNTHESIS_FAILED, details=str(e))   
            
                     
        if audio is None:
            logger.error("Error after synthesis: No audio data generated.")
            raise_api_error(SYNTHESIS_FAILED, details="No audio data generated.")
            
        if audio is None or not isinstance(audio, torch.Tensor) or len(audio) == 0:
            logger.error(f"Error after synthesis: Invalid audio data type or empty ({type(audio)}).")
            raise_api_error(SYNTHESIS_FAILED, details="Generated audio data is invalid or empty.")
       
    buffer = io.BytesIO()
    soundfile.write(buffer, audio, SAMPLE_RATE, format='WAV')
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="audio/wav")
    
@app.post("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}