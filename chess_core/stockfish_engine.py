from dataclasses import dataclass
import shutil
import os
from pathlib import Path
import chess
import chess.engine

@dataclass(frozen=True)
class Analysis:
    best_move: str
    score_cp: int | None
    pv: tuple[str, ...]

class StockfishEngine:
    def __init__(self, path: str | None = None, threads: int = 1, hash_mb: int = 128):
        local = Path(__file__).resolve().parents[1] / "vendor/stockfish/stockfish/stockfish-macos-universal"
        binary = path or os.environ.get("STOCKFISH_PATH") or shutil.which("stockfish") or (str(local) if local.is_file() else None)
        if not binary: raise FileNotFoundError("Stockfish binary not found; install it and set STOCKFISH_PATH")
        self.engine = chess.engine.SimpleEngine.popen_uci(binary)
        self.engine.configure({"Threads": threads, "Hash": hash_mb})
    def analyse(self, board: chess.Board, *, time: float = 0.25, depth: int | None = None) -> Analysis:
        limit = chess.engine.Limit(time=time, depth=depth) if depth else chess.engine.Limit(time=time)
        info = self.engine.analyse(board, limit)
        score = info["score"].pov(board.turn).score(mate_score=100000)
        pv = tuple(m.uci() for m in info.get("pv", ()))
        return Analysis(pv[0] if pv else "", score, pv)
    def close(self) -> None:
        try: self.engine.quit()
        except chess.engine.EngineTerminatedError: pass
    def __enter__(self): return self
    def __exit__(self, *_): self.close()
