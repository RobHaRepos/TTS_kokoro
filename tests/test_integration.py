import pytest
import requests
import os
from pathlib import Path

def _service_up(url: str) -> bool:
    """Return True when the retriever service health endpoint is reachable and OK."""
    try:
        r = requests.get(f"{url}/health", timeout=2)
        return r.status_code == 200
    except requests.RequestException:
        return False

@pytest.mark.integration
@pytest.mark.skipif(not _service_up("http://localhost:8005"), reason="TTS service is not running")
def test_tts_service_health():
    """Test the TTS service health endpoint."""
    r = requests.get("http://localhost:8005/health")
    assert r.status_code == 200
    
@pytest.mark.integration
@pytest.mark.skipif(not _service_up("http://localhost:8005"), reason="TTS service is not running")
def test_tts_service_synthesize():
    """Test the TTS service synthesize endpoint."""
    payload = {
        "text": "The iPhone 16 Pro has a 6.3-inch LTPO Super Retina XDR OLED display with a 120Hz refresh rate, HDR10, Dolby Vision support, 1000 nits typical brightness, and up to 2000 nits peak brightness (HBM). It has a resolution of 1206 x 2622 pixels with a 19.5:9 aspect ratio (~460 ppi density) and is protected by Ceramic Shield glass (2024 generation) with Mohs level 4 hardness.",
        "voice": "am_onyx",
        "speed": 1.0
    }
    r = requests.post("http://localhost:8005/synthesize", json=payload)
    assert r.status_code == 200
    assert r.headers["Content-Type"] == "audio/wav"
    assert len(r.content) > 0
    
    test_dir = Path(__file__).parent
    output_path = test_dir / "test_output.wav"
    
    with open(output_path, "wb") as f:
        f.write(r.content)
    
    try:
        os.startfile(str(output_path))
        print(f"\n✓ Audio saved and playing from: {output_path}")
    except Exception as e:
        print(f"\n✗ Could not play audio: {e}")
        print(f"Audio saved to: {output_path}")