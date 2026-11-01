"""Session-local 2D piece recognizer calibrated from a user-confirmed position."""
from collections import defaultdict
import cv2
import numpy as np
import chess
from shared.schemas import PIECE_CLASSES
from vision.square_mapping import point_for_square

SYMBOL_TO_LABEL = {symbol: f"{color}_{name}" for color, symbols in (("white", "PNBRQK"), ("black", "pnbrqk")) for name, symbol in zip(("pawn", "knight", "bishop", "rook", "queen", "king"), symbols)}
LABELS = (*PIECE_CLASSES, "empty")

def square_image(rectified: np.ndarray, square: str, orientation: str) -> np.ndarray:
    size = rectified.shape[0]
    x, y = point_for_square(square, size, orientation)
    cell = size / 8
    x1, y1 = round(x-cell/2), round(y-cell/2)
    patch = rectified[y1:round(y+cell/2), x1:round(x+cell/2)]
    if patch.size == 0: raise ValueError(f"empty square crop: {square}")
    return cv2.resize(patch, (48, 48), interpolation=cv2.INTER_AREA)

def descriptor(patch: np.ndarray) -> np.ndarray:
    """Signed Lab difference from the square's estimated background color."""
    blurred = cv2.GaussianBlur(patch, (3,3), .7)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB).astype(np.float32)
    corners = np.concatenate((lab[:7, :7].reshape(-1, 3), lab[:7, -7:].reshape(-1, 3), lab[-7:, :7].reshape(-1, 3), lab[-7:, -7:].reshape(-1, 3)))
    background = np.median(corners, axis=0)
    signal = (lab[5:43, 5:43] - background) / 128.0
    color_feature = cv2.resize(signal, (24, 24), interpolation=cv2.INTER_AREA).reshape(-1)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray[5:43,5:43], 30, 90)
    edge_feature = cv2.resize(edges, (24,24), interpolation=cv2.INTER_AREA).astype(np.float32).reshape(-1)/255.0
    return np.concatenate((color_feature, edge_feature * .6))

class TemplateRecognizer:
    def __init__(self, templates: dict[str, np.ndarray], threshold: float = 0.03, margin: float = 0.00001):
        self.templates = templates
        self.threshold = threshold
        self.margin = margin

    @classmethod
    def calibrate(cls, rectified: np.ndarray, board: chess.Board, orientation: str) -> "TemplateRecognizer":
        if not board.is_valid(): raise ValueError("calibration requires a valid confirmed chess position")
        templates: dict[str, list[np.ndarray]] = defaultdict(list)
        for square in chess.SQUARES:
            name = chess.square_name(square)
            piece = board.piece_at(square)
            label = SYMBOL_TO_LABEL[piece.symbol()] if piece else "empty"
            templates[label].append(descriptor(square_image(rectified, name, orientation)))
        missing = set(LABELS) - set(templates)
        if missing: raise ValueError(f"calibration position lacks: {', '.join(sorted(missing))}")
        recognizer = cls({key: np.stack(samples) for key, samples in templates.items()})
        observed, _, errors = recognizer.predict(rectified, orientation)
        expected = {chess.square_name(square): SYMBOL_TO_LABEL[piece.symbol()] for square, piece in board.piece_map().items()}
        if errors or observed != expected:
            raise ValueError(f"calibration failed: {len(errors)} ambiguous squares, {sum(observed.get(s) != expected.get(s) for s in set(observed) | set(expected))} mismatches; adjust board corners or lighting")
        return recognizer

    def predict(self, rectified: np.ndarray, orientation: str) -> tuple[dict[str, str], float, list[str]]:
        occupied: dict[str, str] = {}
        confidences = []
        errors = []
        for square in chess.SQUARES:
            name = chess.square_name(square)
            feature = descriptor(square_image(rectified, name, orientation))
            distances = sorted((float(np.min(np.mean((samples-feature)**2, axis=1))), label) for label, samples in self.templates.items())
            best, label = distances[0]
            gap = distances[1][0] - best
            if best > self.threshold or gap < self.margin:
                errors.append(name)
                continue
            confidences.append(max(0.0, min(1.0, 1.0-best/self.threshold)))
            if label != "empty": occupied[name] = label
        return occupied, min(confidences, default=0.0), errors
