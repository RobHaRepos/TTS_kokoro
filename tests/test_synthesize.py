from src.synthesize_TTS import get_pipeline, synthesize_text
import pytest
import numpy as np
import soundfile
import torch
from dataclasses import dataclass
from typing import Optional

@dataclass
class MockOutput:
    audio: Optional[np.ndarray]
    pred_dur: Optional[torch.LongTensor] = None

@dataclass
class MockResult:
    graphemes: str
    phonemes: str
    output: Optional[MockOutput] = None

def fake_pipeline_generator(text, voice=None, speed=None):
    audio = np.zeros(24000 // 2, dtype='float32')
    output = MockOutput(audio=audio)
    yield MockResult(graphemes="Graphemes", phonemes="Phonemes", output=output)
    
    
def test_get_pipeline():
    """Test getting the Kokoro TTS pipeline."""
    pipeline = get_pipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M', device='cpu')
    assert pipeline is not None
    assert pipeline.lang_code == 'a'
    assert pipeline.repo_id == 'hexgrad/Kokoro-82M'
    
def test_synthesize_text_happy(monkeypatch):
    """Test synthesizing text successfully."""
    monkeypatch.setattr('src.synthesize_TTS.DEVICE', 'cpu')
    text = 'Hello, this is a test of the Kokoro TTS system.'
    audio, graphemes, phonemes = synthesize_text(text, voice='am_onyx', speed=1.0, lang_code='a', local_save=False)
    assert audio is not None
    assert graphemes is not None
    assert phonemes is not None
    assert len(audio) > 0
    
def test_synthesize_text_sad(monkeypatch):
    """Test synthesizing text fails when get_pipeline returns None."""
    monkeypatch.setattr('src.synthesize_TTS.get_pipeline', lambda *args, **kwargs: None, raising=False)
    with pytest.raises(RuntimeError) as excinfo:
        synthesize_text('This will fail', voice='am_onyx', speed=1.0, lang_code='a', local_save=False)
    assert "No audio generated." in str(excinfo.value)

    def fake_none_gen(text, voice=None, speed=None):
        yield MockResult(graphemes="G1", phonemes="P1", output=None)
        yield MockResult(graphemes="G2", phonemes="P2", output=None)

    monkeypatch.setattr(
        'src.synthesize_TTS.get_pipeline',
        lambda *args, **kwargs: (lambda text, voice=None, speed=None: fake_none_gen(text, voice, speed)),
        raising=False,
    )

    audio, graphemes, phonemes = synthesize_text(
        'This will have None audio yields', voice='am_onyx', speed=1.0, lang_code='a', local_save=False)
    assert audio is None
    assert graphemes == 'G2'
    assert phonemes == 'P2'
    
def test_local_save(monkeypatch, tmp_path):
    """Test local saving of synthesized audio."""
    monkeypatch.setattr("src.synthesize_TTS.get_pipeline", lambda *args, **kwargs: (lambda text, voice=None, speed=None: fake_pipeline_generator(text, voice, speed)))
    monkeypatch.setattr("src.synthesize_TTS.OUTPUT_DIR", str(tmp_path), raising=True)
 
    text = 'Testing local save functionality.'
    audio, graphemes, phonemes = synthesize_text(text, voice='am_onyx', speed=1.0, lang_code='a', local_save=True)
    
    assert audio is not None
    assert isinstance(audio, np.ndarray)
    assert graphemes == "Graphemes"
    assert phonemes == "Phonemes"
    
    saved_file = tmp_path / 'synthesized.wav'
    assert saved_file.exists()
    
    data, sr = soundfile.read(saved_file)
    assert sr == 24000
    assert data.size > 0
    