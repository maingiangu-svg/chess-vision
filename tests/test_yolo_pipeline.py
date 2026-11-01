"""Real Ultralytics trainer on explicitly synthetic software fixtures only."""
import json
from pathlib import Path
import chess
import cv2
import numpy as np
import torch
from test_templates import painted_start
from training.data import import_session, split_records, SQUARES
from training.train import validate_splits
from training.yolo import train_yolo
from vision.piece_detection import YoloPieceDetector
from vision.square_mapping import point_for_square
from vision.template_recognition import SYMBOL_TO_LABEL


def test_yolo_trainer_and_detection_schema_on_generated_fixture(tmp_path):
    torch.set_num_threads(2)
    board = chess.Board()
    truth = {s: SYMBOL_TO_LABEL[p.symbol()] if (p := board.piece_at(chess.parse_square(s))) else "empty" for s in SQUARES}
    records = []
    for i, orientation in enumerate(("white", "black", "white_left")):
        folder = tmp_path / str(i)
        folder.mkdir()
        image = folder / "fixture.png"
        cv2.imwrite(str(image), painted_start(orientation))
        # Boxes describe the generated glyph area. They are NOT annotations of real photos.
        boxes = []
        for square, label in truth.items():
            if label != "empty":
                x, y = point_for_square(square, orientation=orientation)
                boxes.append(dict(label=label, bbox=[x-40, y-25, x+35, y+35]))
        image.with_suffix(".labels.json").write_text(json.dumps(dict(image=image.name,
            corners=[[0,0],[799,0],[799,799],[0,799]], orientation=orientation, labels=truth, boxes=boxes)))
        records.extend(import_session(folder, tmp_path / "manifest.jsonl", f"fixture-{i}",
                                      "generated-test-glyphs", "own-test-fixture", "synthetic"))
    splits = split_records(records)
    validate_splits(splits)
    config = json.loads(Path("configs/training.json").read_text())
    config.update(epochs=1, device="cuda:0" if torch.cuda.is_available() else "cpu")
    train_yolo(splits, config, tmp_path / "yolo")
    detector = YoloPieceDetector(str(tmp_path / "yolo/detector/weights/best.pt"))
    detections = detector.predict(np.zeros((800,800,3), np.uint8))
    assert isinstance(detections, list)
    assert set(detector.model.names.values()) == set(SYMBOL_TO_LABEL.values())
    assert all(d.label in SYMBOL_TO_LABEL.values() for d in detections)
