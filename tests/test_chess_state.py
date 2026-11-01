import chess
import pytest
from chess_core.state_manager import ChessState

def test_legal_and_fen_roundtrip():
    s = ChessState(); assert s.legal("e2e4"); s.push("e2e4"); assert s.board.turn == chess.BLACK; assert s.board.piece_at(chess.E4) == chess.Piece(chess.PAWN, chess.WHITE)
    assert s.legal("e7e5")

def test_special_moves():
    s = ChessState("r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq - 0 1")
    assert s.legal("e1g1"); s.push("e1g1")
    s = ChessState("8/8/8/3pP3/8/8/8/4K3 w - d6 0 1"); assert s.legal("e5d6")

def test_illegal_move():
    with pytest.raises(ValueError): ChessState().push("e2e5")

def test_promotion_and_endings():
    promotion = ChessState("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    assert promotion.legal("a7a8q")
    promotion.push("a7a8q")
    assert promotion.board.piece_at(chess.A8) == chess.Piece(chess.QUEEN, chess.WHITE)
    mate = ChessState("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
    assert mate.board.is_checkmate()
    stale = ChessState("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert stale.board.is_stalemate()
