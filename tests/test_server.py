import base64
from pathlib import Path
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from application.server import app, decode_frame
from tests.test_templates import painted_start

def sample():
    image = np.zeros((480, 640, 3), np.uint8)
    ok, data = cv2.imencode('.jpg', image)
    assert ok
    return base64.b64encode(data).decode()

def test_frame_validation():
    assert decode_frame(sample()).shape == (480, 640, 3)
    with pytest.raises(ValueError): decode_frame('not base64')

def test_websocket_without_model():
    with TestClient(app) as client:
        assert client.get('/').status_code == 200
        with client.websocket_connect('/ws') as ws:
            ws.send_json({'type': 'frame', 'jpeg': sample()})
            result = ws.receive_json()
            assert result['type'] == 'result'
            assert result['move'] is None
            ws.send_json({'type': 'setup', 'fen': 'invalid'})
            assert ws.receive_json()['type'] == 'error'

def test_manual_corners_round_trip():
    with TestClient(app) as client, client.websocket_connect('/ws') as ws:
        corners = [[50, 50], [590, 50], [590, 430], [50, 430]]
        ws.send_json({'type': 'corners', 'points': corners})
        ws.send_json({'type': 'frame', 'jpeg': sample()})
        result = ws.receive_json()
        assert result['status'] == 'model_unavailable'
        assert np.allclose(result['corners'], corners)
        assert result['frame_size'] == [640, 480]

def test_calibration_to_real_engine_on_fixture(tmp_path, monkeypatch):
    binary = Path(__file__).resolve().parents[1] / 'vendor/stockfish/stockfish/stockfish-macos-universal'
    if not binary.is_file(): pytest.skip('local Stockfish binary unavailable')
    monkeypatch.setenv('STOCKFISH_PATH', str(binary))
    monkeypatch.setattr('application.server.ROOT', tmp_path)
    ok, encoded = cv2.imencode('.jpg', painted_start())
    assert ok
    frame = base64.b64encode(encoded).decode()
    with TestClient(app) as client, client.websocket_connect('/ws') as ws:
        ws.send_json({'type': 'corners', 'points': [[0,0],[799,0],[799,799],[0,799]]})
        ws.send_json({'type': 'frame', 'jpeg': frame, 'calibrate_templates': True})
        calibrated = ws.receive_json()
        assert calibrated['type'] == 'calibrated'
        for _ in range(3):
            ws.send_json({'type': 'frame', 'jpeg': frame})
            result = ws.receive_json()
        assert result['status'] == 'unchanged'
        assert result['fen'] is not None
        assert result['move'] is not None
        assert len(result['move']['from_pixel']) == 2
