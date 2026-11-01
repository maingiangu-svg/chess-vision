"""Optional integration check against an actual saved Redmi camera frame."""
import base64
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from application.server import app

ROOT = Path(__file__).resolve().parents[1]
IMAGE = ROOT / 'samples/5b8d7152d20b4e56a1f718a1548bc53a.jpg'
ENGINE = ROOT / 'vendor/stockfish/stockfish/stockfish-macos-universal'

@pytest.mark.skipif(not IMAGE.is_file() or not ENGINE.is_file(), reason='saved Redmi frame or Stockfish binary unavailable')
def test_real_camera_frame_to_engine_arrow(tmp_path, monkeypatch):
    monkeypatch.setattr('application.server.ROOT', tmp_path)
    monkeypatch.setenv('STOCKFISH_PATH', str(ENGINE))
    encoded = base64.b64encode(IMAGE.read_bytes()).decode()
    with TestClient(app) as client, client.websocket_connect('/ws') as ws:
        ws.send_json({'type': 'frame', 'jpeg': encoded, 'calibrate_templates': True})
        assert ws.receive_json()['type'] == 'calibrated'
        for _ in range(3):
            ws.send_json({'type': 'frame', 'jpeg': encoded})
            result = ws.receive_json()
        assert result['status'] == 'unchanged'
        assert result['move'] is not None
        assert all(0 <= value < limit for value,limit in zip(result['move']['from_pixel'],result['frame_size']))
