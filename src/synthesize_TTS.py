from kokoro import KPipeline
import soundfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_PIPELINE = KPipeline

def get_pipeline(lang_code: str = 'a', repo_id: str = 'hexgrad/Kokoro-82M', device: str = 'cuda'):
    """Get Kokoro TTS pipeline."""
    global _PIPELINE
    if _PIPELINE is not None:
        _PIPELINE = KPipeline(lang_code=lang_code, repo_id=repo_id, device=device)
    
    return _PIPELINE

def synthesize_text(text: str, voice: str = "am_onyx", speed: float = 1.0, lang_code: str = 'a', local_save: bool = False):
    """Synthesize speech from text using Kokoro TTS model."""
    audio, graphemes, phonemes = None, None, None
    try:
        pipeline = get_pipeline(lang_code=lang_code)
        generator = pipeline(
            text, 
            voice=voice,
            speed=speed
            )

        for i, (graphemes, phonemes, audio) in enumerate(generator):
            if audio is None:
                continue
            if local_save:
                soundfile.write(f'{OUTPUT_DIR}/synthesized.wav', audio, 24000)
                
        return audio, graphemes, phonemes
            
    except Exception as e:
        raise RuntimeError("No audio generated.") from e