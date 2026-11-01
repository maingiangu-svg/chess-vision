import os
import shutil
import chess
import pytest
from pathlib import Path
from chess_core.stockfish_engine import StockfishEngine

LOCAL = Path(__file__).resolve().parents[1] / 'vendor/stockfish/stockfish/stockfish-macos-universal'
@pytest.mark.skipif(not (os.environ.get('STOCKFISH_PATH') or shutil.which('stockfish') or LOCAL.is_file()), reason='Stockfish executable unavailable')
def test_real_uci_analysis():
    with StockfishEngine() as engine:
        board = chess.Board()
        result = engine.analyse(board, time=0.1)
        assert chess.Move.from_uci(result.best_move) in board.legal_moves
        assert result.pv[0] == result.best_move
