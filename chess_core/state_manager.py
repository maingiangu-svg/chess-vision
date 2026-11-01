import chess

class ChessState:
    def __init__(self, fen: str | None = None):
        self.board = chess.Board(fen) if fen else chess.Board()
    def fen(self) -> str: return self.board.fen()
    def legal(self, uci: str) -> bool:
        try: return chess.Move.from_uci(uci) in self.board.legal_moves
        except ValueError: return False
    def push(self, uci: str) -> None:
        move = chess.Move.from_uci(uci)
        if move not in self.board.legal_moves: raise ValueError(f"illegal move: {uci}")
        self.board.push(move)
    def pop(self) -> str: return self.board.pop().uci()
    def observe_fen(self, fen: str) -> bool:
        observed = chess.Board(fen)
        return observed.board_fen() == self.board.board_fen() and observed.turn == self.board.turn
