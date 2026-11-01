import numpy as np
from shared.schemas import Detection

FILES = "abcdefgh"
def square_from_point(x: float, y: float, board_size: int = 800, orientation: str = "white") -> str:
    col, row = int(x / (board_size / 8)), int(y / (board_size / 8))
    if not (0 <= col < 8 and 0 <= row < 8): raise ValueError("point outside board")
    if orientation == "white": return f"{FILES[col]}{8-row}"
    if orientation == "black": return f"{FILES[7-col]}{row+1}"
    if orientation == "white_left": return f"{FILES[row]}{col+1}"
    if orientation == "white_right": return f"{FILES[7-row]}{8-col}"
    raise ValueError("invalid orientation")

def point_for_square(square: str, board_size: int = 800, orientation: str = "white") -> tuple[float, float]:
    if len(square) != 2 or square[0] not in FILES or square[1] not in "12345678": raise ValueError(square)
    col, row = FILES.index(square[0]), 8-int(square[1])
    if orientation == "black": col, row = 7-col, 7-row
    elif orientation == "white_left": col, row = 7-row, col
    elif orientation == "white_right": col, row = row, 7-col
    elif orientation != "white": raise ValueError("invalid orientation")
    cell = board_size / 8
    return ((col + .5) * cell, (row + .5) * cell)

def assign_detections(detections: list[Detection], board_size: int = 800, orientation: str = "white") -> tuple[dict[str, str], list[str]]:
    """Use the lower center of each box as a piece base estimate; reject collisions."""
    squares: dict[str, tuple[str, float]] = {}
    errors = []
    for d in detections:
        x1, y1, x2, y2 = d.bbox
        try: square = square_from_point((x1+x2)/2, y1 + .85*(y2-y1), board_size, orientation)
        except ValueError:
            errors.append(f"outside board: {d.label}")
            continue
        if square in squares:
            errors.append(f"collision: {square}")
            if d.confidence > squares[square][1]: squares[square] = (d.label, d.confidence)
        else: squares[square] = (d.label, d.confidence)
    return {square: label for square, (label, _) in squares.items()}, errors
