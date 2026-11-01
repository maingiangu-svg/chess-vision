"""Evaluate on independently hand-labeled raw Redmi frames; no synthetic metrics."""
import argparse
import json
from pathlib import Path
import chess
import cv2
import numpy as np
from vision.board_detection import BoardGeometry
from vision.template_recognition import TemplateRecognizer

def load_image(path: Path):
    image = cv2.imread(str(path))
    if image is None: raise ValueError(f"cannot read {path}")
    return image

def main():
    p = argparse.ArgumentParser()
    p.add_argument("calibration", type=Path, help="calibration-*.jpg saved from the phone")
    p.add_argument("labels", nargs="+", type=Path, help="independent *.labels.json files")
    args = p.parse_args()
    meta = json.loads(Path(f"{args.calibration}.json").read_text())
    board = chess.Board(meta["fen"])
    geometry = BoardGeometry(np.asarray(meta["corners"], np.float32))
    model = TemplateRecognizer.calibrate(geometry.rectify(load_image(args.calibration)), board, meta["orientation"])
    correct = total = uncertain = 0
    mistakes = []
    for path in args.labels:
        annotated = json.loads(path.read_text())
        image_path = Path(annotated["image"])
        if image_path.resolve() == args.calibration.resolve(): raise ValueError("calibration image cannot be evaluated as held-out data")
        image = load_image(image_path)
        geom = BoardGeometry(np.asarray(annotated["corners"], np.float32))
        predicted, _, errors = model.predict(geom.rectify(image), annotated["orientation"])
        labels = annotated.get("labels")
        if labels is None and "fen_placement" in annotated:
            truth = chess.Board(f"{annotated['fen_placement']} w - - 0 1")
            from vision.template_recognition import SYMBOL_TO_LABEL
            labels = {chess.square_name(s): SYMBOL_TO_LABEL[truth.piece_at(s).symbol()] if truth.piece_at(s) else "empty" for s in chess.SQUARES}
        if labels is None: raise ValueError(f"no labels in {path}")
        for square, actual in labels.items():
            total += 1
            if square in errors: uncertain += 1
            elif predicted.get(square, "empty") == actual: correct += 1
            else: mistakes.append({"square": square, "expected": actual, "predicted": predicted.get(square, "empty")})
    if total == 0: raise ValueError("no manually labeled squares")
    print(json.dumps({"labeled_squares": total, "correct": correct, "uncertain": uncertain, "incorrect_confident": mistakes, "accuracy_including_uncertain_as_error": correct/total}, indent=2))

if __name__ == "__main__": main()
