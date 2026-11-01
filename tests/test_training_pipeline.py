"""End-to-end plumbing with generated test glyphs, not recognition evidence.

No test checkpoint is a trained camera model; all files remain in pytest temp.
"""
import json
import chess
import cv2
import numpy as np
import torch
from test_templates import painted_start
from training.data import import_session, split_records, SQUARES
from training.train import train_squares, validate_splits
from training.evaluate import evaluate_boards
from training.export import export_squares
from vision.square_recognition import SquareRecognizer
from vision.template_recognition import SYMBOL_TO_LABEL


def test_training_checkpoint_export_and_evaluation_plumbing(tmp_path):
    torch.set_num_threads(2)
    records = []
    board = chess.Board()
    truth = {s: SYMBOL_TO_LABEL[p.symbol()] if (p := board.piece_at(chess.parse_square(s))) else "empty" for s in SQUARES}
    for index, orientation in enumerate(("white", "black", "white_left")):
        directory = tmp_path / f"synthetic-session-{index}"
        directory.mkdir()
        image = directory / "fixture.png"
        cv2.imwrite(str(image), painted_start(orientation))
        image.with_suffix(".labels.json").write_text(json.dumps(dict(
            image=image.name, corners=[[0,0],[799,0],[799,799],[0,799]],
            orientation=orientation, labels=truth, tags=["synthetic-test-fixture"], fen=board.fen())))
        records.extend(import_session(directory, tmp_path / "manifest.jsonl", f"fixture-{index}",
                                      "generated-test-glyphs", "own-test-fixture", "synthetic"))
    splits = split_records(records)
    validate_splits(splits)
    config = json.loads(open("configs/training.json").read())
    config.update(epochs=1, square_batch=16, device="cuda:0" if torch.cuda.is_available() else "cpu")
    output = tmp_path / "training"
    output.mkdir()
    # No pretrained download or camera-accuracy claim in this software test.
    train_squares(splits, config, output, pretrained=False)
    recognizer = SquareRecognizer(output / "best.pt", device=config["device"])
    report, rows = evaluate_boards(splits["val"], recognizer, "squares", config)
    assert report["boards"] == 1 and len(rows[0]["predicted"]) == 64
    assert report["real_camera"] is None
    exported = output / "fixture.onnx"
    export_squares(output / "best.pt", exported, "onnx")
    onnx_model = SquareRecognizer(exported)
    patches = np.zeros((64,48,48,3), np.uint8)
    np.testing.assert_allclose(onnx_model.probabilities(patches),
                               SquareRecognizer(output / "best.pt").probabilities(patches), atol=1e-4)
