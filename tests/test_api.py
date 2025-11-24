from fastapi.testclient import TestClient
from fastapi.responses import JSONResponse
from fastapi import Request, HTTPException
from src.api.api import app, http_exception_handler

import torch
import io
import soundfile
import pytest
import json

def test_synthesize_empty_text():
    client = TestClient(app)
    response = client.post("/synthesize", json={"text": ""})
    assert response.status_code == 400
    assert "Text for synthesis is empty" in response.json().get("message", "")
    
def test_synthesize_happy(monkeypatch):
    def fake_synthesize_text(text, voice=None, speed=None):
        audio = torch.zeros(24000, dtype=torch.float32)
        return audio, "Graphemes", "Phonemes"
    
    monkeypatch.setattr("src.api.api.synthesize_text", fake_synthesize_text)
    
    client = TestClient(app)
    response = client.post("/synthesize", json={"text": "Hello, world!"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")
    
    wav_bytes = io.BytesIO(response.content)
    data, samplerate = soundfile.read(wav_bytes)
    assert samplerate == 24000
    assert data.size > 0
    
def test_synthesize_sad(monkeypatch):
    def fake_raise(text, voice=None, speed=None):
        raise RuntimeError("Synthesis failed.")
    
    monkeypatch.setattr("src.api.api.synthesize_text", fake_raise)
    
    client = TestClient(app)
    response = client.post("/synthesize", json={"text": "This will fail."})
    assert response.status_code == 500
    assert "Internal server error during synthesis" in response.json().get("message", "")
    
def test_synthesize_none_audio(monkeypatch):
    def fake_none_audio(text, voice=None, speed=None):
        return None, "Graphemes", "Phonemes"
    
    monkeypatch.setattr("src.api.api.synthesize_text", fake_none_audio)
    
    client = TestClient(app)
    response = client.post("/synthesize", json={"text": "This will return None audio."})
    assert response.status_code == 500
    data = response.json()
    assert data["code"] == "SYNTHESIS_FAILED"
    assert data["details"] == "No audio data generated."
    assert "Internal server error during synthesis" in data["message"]
    
def test_synthesize_empty_audio(monkeypatch):
    def fake_empty_audio(text, voice=None, speed=None):
        audio = torch.zeros(0, dtype=torch.float32)
        return audio, "Graphemes", "Phonemes"
    
    monkeypatch.setattr("src.api.api.synthesize_text", fake_empty_audio)
    
    client = TestClient(app)
    response = client.post("/synthesize", json={"text": "This will return empty audio."})
    assert response.status_code == 500
    data = response.json()
    assert data["code"] == "SYNTHESIS_FAILED"
    assert data["details"] == "Generated audio data is invalid or empty."
    
def test_synthesize_invalid_audio_type(monkeypatch):
    def fake_invalid_audio(text, voice=None, speed=None):
        audio = torch.zeros(24000, dtype=torch.float32).tolist() 
        return audio, "Graphemes", "Phonemes"
    
    monkeypatch.setattr("src.api.api.synthesize_text", fake_invalid_audio)
    
    client = TestClient(app)
    response = client.post("/synthesize", json={"text": "This will return invalid audio type."})
    assert response.status_code == 500
    data = response.json()
    assert data["code"] == "SYNTHESIS_FAILED"
    assert data["details"] == "Generated audio data is invalid or empty."
    
def test_lifespan(monkeypatch):
    # Force CPU mode for CI environments without CUDA
    monkeypatch.setattr('src.api.api.DEVICE', 'cpu')
    
    def fake_synthesize_text(text, voice=None, speed=None):
        audio = torch.zeros(24000, dtype=torch.float32)
        return audio, "Graphemes", "Phonemes"
    
    monkeypatch.setattr("src.api.api.synthesize_text", fake_synthesize_text)
    
    with TestClient(app) as client:
        input_text = "Test"
        response = client.post("/synthesize", json={"text": input_text})
        assert response.status_code == 200
        
@pytest.mark.asyncio
async def test_http_exception_handler_non_dict_direct():
    scope = {"type": "http", "method": "GET", "path": "/"}
    request = Request(scope=scope)
    exc = HTTPException(status_code=418, detail="I'm a teapot")
    
    response = await http_exception_handler(request, exc)
    
    assert response.status_code == 418
    assert isinstance(response, JSONResponse)
    data = json.loads(response.body)
    assert data == {
        "code": "HTTP_ERROR",
        "message": "I'm a teapot",
        "details": None
    }
    