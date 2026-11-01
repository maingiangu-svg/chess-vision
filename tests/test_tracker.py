import chess
from chess_core.state_manager import ChessState
from chess_core.move_tracker import MoveTracker, placement_from_squares

def test_confirm_only_unique_legal_stable_move():
    state = ChessState()
    tracker = MoveTracker(state, stable_frames=2)
    next_board = state.board.copy()
    next_board.push_uci("e2e4")
    assert tracker.update(next_board.board_fen()) == ("waiting", None)
    assert tracker.update(next_board.board_fen()) == ("moved", "e2e4")
    assert state.fen() == next_board.fen()

def test_reject_impossible_observation():
    state = ChessState()
    tracker = MoveTracker(state, stable_frames=1)
    assert tracker.update(chess.Board(None).board_fen()) == ("uncertain", None)
    assert state.fen() == chess.Board().fen()

def test_placement():
    assert placement_from_squares({"e1": "white_king", "e8": "black_king"}) == "4k3/8/8/8/8/8/8/4K3"
