import asyncio
import os
import io
import logging
import soundfile
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager

from synthesize_TTS import synthesize_text, get_pipeline
from .log_handler import HTTPLogHandler

VOICE = os.getenv("VOICE", "am_onyx")
SAMPLE_RATE = 24000
LOGGER_SERVICE_URL = os.getenv("LOGGER_SERVICE_URL", "http://localhost:8005")

logger = logging.getLogger("synthesize_service")
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
remote = HTTPLogHandler(LOGGER_SERVICE_URL)
logger.addHandler(remote)
logger.setLevel(logging.INFO)

class SynthesizeRequest(BaseModel):
    text: str
    voice: str = VOICE
    speed: float = 1.0
    
@asynccontextmanager
async def lifespan(app: FastAPI):
    get_pipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M', device='cuda')
    try:
        yield
    finally:
        logger.info("Shutting down Kokoro TTS API.")

app = FastAPI(title="Kokoro TTS API", lifespan=lifespan)
        
synthesize_lock = asyncio.Lock()    

@app.post("/synthesize", response_class=StreamingResponse)
async def synthesize(request: SynthesizeRequest):
    """Synthesize speech from text."""
    text = request.text.strip() if request.text else ""
    if not text: 
        logger.warning("Received empty text for synthesis.")
        raise HTTPException(status_code=400, detail="Text for synthesis is empty.")
    
    async with synthesize_lock:
        try:
            audio, _, _ = await asyncio.to_thread(synthesize_text, text, voice=request.voice, speed=request.speed)
        except Exception as e:
            logger.error(f"Error during synthesis: {e}")
            raise HTTPException(status_code=500, detail=f"Internal server error during synthesis: {e}")
       
    buffer = io.BytesIO()
    soundfile.write(buffer, audio, SAMPLE_RATE, format='WAV')
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="audio/wav")
    
@app.post("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}