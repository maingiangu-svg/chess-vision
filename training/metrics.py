"""Unfiltered board metrics: abstention is an error, not an omitted square."""
import chess
import numpy as np
from training.data import CLASSES, SQUARES
from vision.template_recognition import SYMBOL_TO_LABEL

SYMBOLS = {label: symbol for symbol, label in SYMBOL_TO_LABEL.items()}


def plausible(predicted, fen=None):
    if any(predicted[s] not in CLASSES for s in SQUARES):
        return False
    board = chess.Board(fen) if fen else chess.Board(None)
    board.clear_board()
    for square, label in predicted.items():
        if label != "empty":
            board.set_piece_at(chess.parse_square(square), chess.Piece.from_symbol(SYMBOLS[label]))
    if fen:
        return board.is_valid()
    # No historical state is inferred from pixels. This is a plausibility check only.
    return any(_valid_turn(board, turn) for turn in (chess.WHITE, chess.BLACK))


def _valid_turn(board, turn):
    candidate = board.copy()
    candidate.turn = turn
    return candidate.is_valid()


def summarize(rows):
    if not rows:
        raise ValueError("No evaluated boards")
    matrix = np.zeros((13, 14), dtype=int)
    exact = invalid = 0
    failures = []
    for row in rows:
        truth, predicted = row["truth"], row["predicted"]
        if set(truth) != set(SQUARES) or set(predicted) != set(SQUARES):
            raise ValueError("Require all 64 squares including uncertain")
        errors = []
        for s in SQUARES:
            target, label = CLASSES.index(truth[s]), predicted[s]
            index = CLASSES.index(label) if label in CLASSES else 13
            matrix[target, index] += 1
            if label != truth[s]:
                errors.append(dict(square=s, truth=truth[s], predicted=label))
        exact += not errors
        invalid += not plausible(predicted, row.get("fen"))
        if errors:
            failures.append(dict(image=row.get("image"), tags=row.get("tags", []), errors=errors))
    diagonal = np.diag(matrix[:, :13])
    per_class = {}
    for i, label in enumerate(CLASSES):
        support, detections = int(matrix[i].sum()), int(matrix[:, i].sum())
        per_class[label] = dict(support=support,
                                precision=float(diagonal[i]/detections) if detections else None,
                                recall=float(diagonal[i]/support) if support else None)
    latencies = [r["latency_ms"] for r in rows]
    result = dict(boards=len(rows), squares=len(rows)*64,
                  square_accuracy=float(diagonal.sum()/(len(rows)*64)),
                  full_board_exact_match=exact/len(rows),
                  abstention_rate=float(matrix[:, 13].sum()/(len(rows)*64)),
                  invalid_position_frequency=invalid/len(rows),
                  invalid_position_definition="is_valid with supplied FEN context, otherwise either turn/no castling; uncertainty is invalid",
                  confusion_labels=list(CLASSES)+["uncertain"], confusion_matrix=matrix.tolist(),
                  per_class=per_class, latency_ms=dict(median=float(np.median(latencies)),
                                                      p95=float(np.percentile(latencies, 95))),
                  failures=failures)
    result["by_tag"] = {}
    for tag in sorted({tag for r in rows for tag in r.get("tags", [])}):
        subset = [r for r in rows if tag in r.get("tags", [])]
        result["by_tag"][tag] = dict(boards=len(subset), exact_match=sum(r["truth"] == r["predicted"] for r in subset)/len(subset))
    return result
