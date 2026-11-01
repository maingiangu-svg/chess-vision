"""Contract tests use generated pixels and untrained weights, not camera evidence."""
from dataclasses import asdict
import json
import numpy as np
import pytest
from shared.schemas import Detection, PIECE_CLASSES
from vision.square_recognition import CLASSES, SquareRecognizer
from vision.template_recognition import TemplateRecognizer


@pytest.mark.parametrize("orientation", ["white", "black", "white_left", "white_right"])
def test_uncertain_empty_square_blocks_existing_tracker(orientation):
    from application.server import analyze_frame
    from chess_core.move_tracker import MoveTracker
    from chess_core.state_manager import ChessState
    from vision.board_detection import BoardGeometry
    recognizer = SquareRecognizer.__new__(SquareRecognizer)
    recognizer.threshold, recognizer.margin, recognizer.fallback = .9, .25, None
    recognizer.probabilities = lambda patches: np.full((64, 13), 1/13)
    assert isinstance(recognizer, SquareRecognizer)
    frame = np.zeros((800, 800, 3), np.uint8)
    occupied, confidence, errors = recognizer.predict(frame, orientation)
    assert occupied == {} and len(errors) == 64
    geometry = BoardGeometry(np.array([[0,0],[799,0],[799,799],[0,799]], np.float32))
    state = ChessState()
    status, _ = analyze_frame(frame, geometry, recognizer, MoveTracker(state), orientation)
    assert status == "uncertain" and state.fen() == ChessState().fen()


def test_detection_json_contract():
    detection = Detection(PIECE_CLASSES[0], .95, (1., 2., 3., 4.))
    decoded = json.loads(json.dumps(asdict(detection)))
    assert set(decoded) == {"label", "confidence", "bbox", "square", "metadata"}
    assert decoded["label"] == CLASSES[0] and len(decoded["bbox"]) == 4


def test_server_opt_in_square_loading_and_ambiguous_config(monkeypatch):
    from fastapi.testclient import TestClient
    from application.server import app
    sentinel = object()
    monkeypatch.setenv("CHESS_SQUARE_WEIGHTS", "fixture.pt")
    monkeypatch.delenv("CHESS_WEIGHTS", raising=False)
    monkeypatch.setattr("application.server.SquareRecognizer", lambda weights, device: sentinel)
    with TestClient(app):
        assert app.state.detector is sentinel
    monkeypatch.setenv("CHESS_WEIGHTS", "other.pt")
    with pytest.raises(ValueError, match="either"):
        with TestClient(app):
            pass


def test_orientation_change_retains_neural_model(monkeypatch):
    import base64
    import cv2
    from fastapi.testclient import TestClient
    from application.server import app
    monkeypatch.delenv("CHESS_SQUARE_WEIGHTS", raising=False)
    monkeypatch.delenv("CHESS_WEIGHTS", raising=False)
    recognizer = SquareRecognizer.__new__(SquareRecognizer)
    recognizer.threshold, recognizer.margin, recognizer.fallback = .9, .25, None
    recognizer.probabilities = lambda patches: np.full((64,13), 1/13)
    encoded = base64.b64encode(cv2.imencode(".jpg",np.zeros((800,800,3),np.uint8))[1]).decode()
    with TestClient(app) as client:
        app.state.detector = recognizer
        with client.websocket_connect("/ws") as ws:
            ws.send_json(dict(type="orientation", value="black"))
            ws.send_json(dict(type="corners", points=[[0,0],[799,0],[799,799],[0,799]]))
            ws.send_json(dict(type="frame", jpeg=encoded))
            assert ws.receive_json()["status"] == "uncertain"


@pytest.mark.parametrize("architecture", ["mobilenet_v3_small", "tiny_cnn"])
def test_checkpoint_round_trip_and_wrong_mapping(tmp_path, architecture):
    import torch
    from vision.square_recognition import build_model
    torch.set_num_threads(2)
    path = tmp_path / "fixture.pt"
    checkpoint = dict(state_dict=build_model(architecture=architecture).state_dict(),
                      classes=list(CLASSES), input_size=48, architecture=architecture)
    torch.save(checkpoint, path)
    recognizer = SquareRecognizer(path)
    probabilities = recognizer.probabilities(np.zeros((64,48,48,3), np.uint8))
    assert probabilities.shape == (64, 13)
    np.testing.assert_allclose(probabilities.sum(1), 1, atol=1e-6)
    checkpoint["classes"] = list(reversed(CLASSES))
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="mapping"):
        SquareRecognizer(path)


def test_yolo_requires_true_manual_boxes(tmp_path):
    from training.data import import_session
    from training.yolo import prepare_yolo
    from test_training_data import make_sample
    path = make_sample(tmp_path)
    data = json.loads(path.read_text())
    data["labels"]["a1"] = "white_rook"
    path.write_text(json.dumps(data))
    records = import_session(tmp_path, tmp_path / "manifest.jsonl", "s", "test-fixture", "own", "synthetic")
    with pytest.raises(ValueError, match="manually annotated"):
        prepare_yolo({"train": records}, tmp_path / "yolo")
