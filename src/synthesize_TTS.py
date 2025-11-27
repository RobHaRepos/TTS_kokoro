from kokoro import KPipeline
import soundfile
from pathlib import Path
import logging
import os
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", 24000))
SPEED = float(os.getenv("SPEED", 1.0))
VOICE = os.getenv("VOICE", "am_onyx")
LANG_CODE = os.getenv("LANG_CODE", "a")
DEVICE = os.getenv("DEVICE", "cuda")

_PIPELINE = None

logger = logging.getLogger("synthesize_service")

def get_pipeline(lang_code: str = LANG_CODE, repo_id: str = 'hexgrad/Kokoro-82M', device: Optional[str] = None):
    """Get Kokoro TTS pipeline."""
    global _PIPELINE
    if device is None:
        device = DEVICE
    if _PIPELINE is None:
        _PIPELINE = KPipeline(lang_code=lang_code, repo_id=repo_id, device=device)
    
    return _PIPELINE

def synthesize_text(text: str, voice: str = VOICE, speed: float = SPEED, lang_code: str = LANG_CODE, local_save: bool = False):
    """Synthesize speech from text using Kokoro TTS model."""
    import torch
    audio_chunks = []
    all_graphemes = []
    all_phonemes = []
    last_graphemes = None
    last_phonemes = None
    
    try:
        pipeline = get_pipeline(lang_code=lang_code)
        generator = pipeline(
            text, 
            voice=voice,
            speed=speed
            )

        logger.info(f"Synthesizing text: {text} with voice: {voice} at speed: {speed}")
        
        for result in generator:
            logger.debug(f"Received chunk with graphemes: {result.graphemes}, phonemes: {result.phonemes}")
            last_graphemes = result.graphemes
            last_phonemes = result.phonemes
            if result.output is None or result.output.audio is None:
                continue
            audio_chunks.append(result.output.audio)
            all_graphemes.append(result.graphemes)
            all_phonemes.append(result.phonemes)
        
        if not audio_chunks:
            return None, last_graphemes, last_phonemes
        
        if isinstance(audio_chunks[0], torch.Tensor):
            audio = torch.cat(audio_chunks, dim=0) if len(audio_chunks) > 1 else audio_chunks[0]
        else:
            import numpy as np
            audio = np.concatenate(audio_chunks) if len(audio_chunks) > 1 else audio_chunks[0]
            
        graphemes = " ".join(all_graphemes)
        phonemes = " ".join(all_phonemes)
        
        if local_save:
            soundfile.write(f'{OUTPUT_DIR}/synthesized.wav', audio, SAMPLE_RATE)
        
        logger.info(f"Synthesis complete for text: {text}")
                
        return audio, graphemes, phonemes
            
    except Exception as e:
        raise RuntimeError("No audio generated.") from e