"""Confirm a stable observed position by matching a unique legal transition."""
from collections import Counter
import chess
from chess_core.state_manager import ChessState

SYMBOLS = {f"{color}_{name}": symbol for color, symbols in (("white", "PNBRQK"), ("black", "pnbrqk")) for name, symbol in zip(("pawn", "knight", "bishop", "rook", "queen", "king"), symbols)}

def placement_from_squares(squares: dict[str, str]) -> str:
    board = chess.Board(None)
    for square, label in squares.items():
        board.set_piece_at(chess.parse_square(square), chess.Piece.from_symbol(SYMBOLS[label]))
    return board.board_fen()

class MoveTracker:
    def __init__(self, state: ChessState, stable_frames: int = 3):
        self.state = state
        self.stable_frames = stable_frames
        self.last: str | None = None
        self.count = 0

    def update(self, placement: str | None) -> tuple[str, str | None]:
        if not placement:
            self.last, self.count = None, 0
            return "uncertain", None
        if placement != self.last:
            self.last, self.count = placement, 1
        else:
            self.count += 1
        if self.count < self.stable_frames: return "waiting", None
        if placement == self.state.board.board_fen(): return "unchanged", None
        matches = []
        for move in self.state.board.legal_moves:
            candidate = self.state.board.copy(stack=False)
            candidate.push(move)
            if candidate.board_fen() == placement: matches.append(move)
        if len(matches) != 1: return "uncertain", None
        uci = matches[0].uci()
        self.state.push(uci)
        return "moved", uci
