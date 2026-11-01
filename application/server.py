"""LAN server for phone camera frames. No cloud upload or public tunnel."""
import asyncio
import base64
import os
import shutil
import time
import uuid
import json
from contextlib import asynccontextmanager
from pathlib import Path

import chess
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from chess_core.state_manager import ChessState
from chess_core.move_tracker import MoveTracker, placement_from_squares
from chess_core.stockfish_engine import StockfishEngine
from vision.board_detection import BoardGeometry, detect_board_corners, order_corners
from vision.piece_detection import YoloPieceDetector
from vision.square_recognition import SquareRecognizer
from vision.square_mapping import assign_detections, point_for_square
from vision.template_recognition import TemplateRecognizer

ROOT = Path(__file__).resolve().parents[1]

@asynccontextmanager
async def lifespan(app: FastAPI):
    square_weights, yolo_weights = os.getenv("CHESS_SQUARE_WEIGHTS"), os.getenv("CHESS_WEIGHTS")
    if square_weights and yolo_weights:
        raise ValueError("Set either CHESS_SQUARE_WEIGHTS or CHESS_WEIGHTS")
    if square_weights:
        app.state.detector = SquareRecognizer(square_weights, device=os.getenv("CHESS_DEVICE", "cpu"))
    elif yolo_weights:
        app.state.detector = YoloPieceDetector(yolo_weights)
    else:
        app.state.detector = None
    local = ROOT / "vendor/stockfish/stockfish/stockfish-macos-universal"
    path = os.environ.get("STOCKFISH_PATH") or shutil.which("stockfish") or (str(local) if local.is_file() else None)
    app.state.engine = StockfishEngine(path) if path else None
    yield
    if app.state.engine: app.state.engine.close()

app = FastAPI(lifespan=lifespan)

@app.get("/")
def index(): return FileResponse(ROOT / "frontend" / "index.html")

@app.get("/app.js")
def script(): return FileResponse(ROOT / "frontend" / "app.js", media_type="application/javascript")

@app.get("/style.css")
def style(): return FileResponse(ROOT / "frontend" / "style.css", media_type="text/css")

def decode_frame(encoded: str) -> np.ndarray:
    if len(encoded) > 1_500_000: raise ValueError("frame too large")
    raw = base64.b64decode(encoded, validate=True)
    frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if frame is None or frame.shape[0] > 1280 or frame.shape[1] > 1280: raise ValueError("invalid frame")
    return frame

def analyze_frame(frame, geometry, detector, tracker, orientation):
    if detector is None: return "model_unavailable", 0.0
    rectified = geometry.rectify(frame)
    if isinstance(detector, (TemplateRecognizer, SquareRecognizer)):
        squares, confidence, errors = detector.predict(rectified, orientation)
    else:
        detections = detector.predict(rectified)
        squares, errors = assign_detections(detections, geometry.size, orientation)
        confidence = min((d.confidence for d in detections), default=0.0)
    if errors or sum(v.endswith("_king") for v in squares.values()) != 2:
        tracker.update(None)
        return "uncertain", 0.0
    status, _ = tracker.update(placement_from_squares(squares))
    return status, confidence

@app.websocket("/ws")
async def websocket(ws: WebSocket):
    await ws.accept()
    state = ChessState()
    tracker = MoveTracker(state)
    corners = None
    orientation = "white"
    analysis = None
    analyzed_fen = None
    recognizer = app.state.detector
    try:
        while True:
            msg = await ws.receive_json()
            action = msg.get("type")
            if action == "setup":
                try:
                    board = chess.Board(msg["fen"])
                    if not board.is_valid(): raise ValueError("invalid chess position")
                    state = ChessState(board.fen()); tracker = MoveTracker(state)
                    analysis = analyzed_fen = None
                    await ws.send_json({"type": "setup_ok", "fen": state.fen()})
                except (KeyError, ValueError) as exc: await ws.send_json({"type": "error", "message": str(exc)})
                continue
            if action == "reset":
                state, corners, analysis, analyzed_fen = ChessState(), None, None, None
                tracker = MoveTracker(state)
                recognizer = app.state.detector
                await ws.send_json({"type": "reset_ok", "fen": state.fen()})
                continue
            if action == "detect":
                corners = None
                continue
            if action == "orientation":
                if msg.get("value") in ("white", "black", "white_left", "white_right") and msg["value"] != orientation:
                    orientation = msg["value"]
                    tracker.update(None)
                    analysis = analyzed_fen = None
                    if isinstance(recognizer, TemplateRecognizer): recognizer = None
                continue
            if action == "corners":
                try:
                    corners = BoardGeometry(order_corners(np.asarray(msg["points"], dtype=np.float32)))
                except (KeyError, ValueError, TypeError): await ws.send_json({"type": "error", "message": "invalid corners"})
                continue
            if action != "frame": continue
            started = time.perf_counter()
            try: frame = decode_frame(msg["jpeg"])
            except (KeyError, ValueError) as exc:
                await ws.send_json({"type": "error", "message": str(exc)})
                continue
            if msg.get("save_sample"):
                directory = ROOT / "samples"
                directory.mkdir(exist_ok=True)
                filename = f"{uuid.uuid4().hex}.jpg"
                cv2.imwrite(str(directory / filename), frame)
                await ws.send_json({"type": "sample_saved", "filename": filename})
                continue
            geometry = corners or detect_board_corners(frame)
            if msg.get("calibrate_templates"):
                try:
                    if geometry is None: raise ValueError("Chưa tìm thấy 4 góc bàn cờ")
                    if state.board.fen() != chess.Board().fen(): raise ValueError("Hiệu chuẩn mẫu cần bàn cờ ở vị trí xuất phát")
                    candidate = TemplateRecognizer.calibrate(geometry.rectify(frame), state.board, orientation)
                    directory = ROOT / "samples"
                    directory.mkdir(exist_ok=True)
                    filename = f"calibration-{uuid.uuid4().hex}.jpg"
                    cv2.imwrite(str(directory / filename), frame)
                    (directory / f"{filename}.json").write_text(json.dumps({"fen": state.fen(), "corners": geometry.corners.tolist(), "orientation": orientation, "confirmed_by_user": True}))
                    recognizer = candidate
                    tracker.update(None)
                    await ws.send_json({"type": "calibrated", "filename": filename})
                except ValueError as exc: await ws.send_json({"type": "error", "message": str(exc)})
                continue
            status, confidence = "board_not_found", 0.0
            if geometry:
                status, confidence = analyze_frame(frame, geometry, recognizer, tracker, orientation)
            else: tracker.update(None)
            if status in ("moved", "unchanged") and app.state.engine and state.fen() != analyzed_fen:
                analyzed_fen = state.fen()
                try: analysis = await asyncio.to_thread(app.state.engine.analyse, state.board.copy())
                except Exception: analysis = None; status = "engine_error"
            move = None
            if geometry and analysis and analysis.best_move and status in ("moved", "unchanged"):
                uci = analysis.best_move
                board_points = np.array([[point_for_square(uci[:2], geometry.size, orientation), point_for_square(uci[2:4], geometry.size, orientation)]], dtype=np.float32)
                pixels = cv2.perspectiveTransform(board_points, geometry.homography()[1])[0].tolist()
                move = {"uci": uci, "san": state.board.san(chess.Move.from_uci(uci)), "from_pixel": pixels[0], "to_pixel": pixels[1], "score_cp": analysis.score_cp, "pv": analysis.pv[:5]}
            await ws.send_json({"type": "result", "status": status, "confidence": confidence, "corners": geometry.corners.tolist() if geometry else None, "board_size": geometry.size if geometry else None, "frame_size": [frame.shape[1], frame.shape[0]], "fen": state.fen() if status in ("moved", "unchanged") else None, "turn": "white" if state.board.turn else "black", "move": move, "latency_ms": round((time.perf_counter()-started)*1000, 1)})
    except WebSocketDisconnect: pass
