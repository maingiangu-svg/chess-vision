import argparse
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from vision.board_detection import BoardGeometry, detect_board_corners, order_corners
from vision.piece_detection import YoloPieceDetector
from vision.square_mapping import assign_detections
from chess_core.state_manager import ChessState
from chess_core.move_tracker import MoveTracker, placement_from_squares
from chess_core.stockfish_engine import StockfishEngine
from application.visualizer import draw_move
from shared.schemas import HudMove

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--weights")
    p.add_argument("--stockfish", default=os.environ.get("STOCKFISH_PATH"))
    p.add_argument("--fen")
    p.add_argument("--orientation", choices=("white", "black"), default="white")
    args = p.parse_args()
    detector = YoloPieceDetector(args.weights) if args.weights else None
    engine = StockfishEngine(args.stockfish) if args.stockfish or shutil.which("stockfish") else None
    state = ChessState(args.fen)
    tracker = MoveTracker(state)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        if engine: engine.close()
        raise RuntimeError(f"Cannot open camera {args.camera}")
    points = []
    manual = None
    def click(event, x, y, flags, userdata):
        nonlocal manual
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            if len(points) == 4:
                manual = BoardGeometry(order_corners(np.array(points)))
                points.clear()
    cv2.namedWindow("Chess Vision AI")
    cv2.setMouseCallback("Chess Vision AI", click)
    executor = ThreadPoolExecutor(max_workers=1)
    pending, analysis, analyzed_fen = None, None, None
    status = "board not found"
    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            geometry = manual or detect_board_corners(frame)
            if geometry: cv2.polylines(frame, [geometry.corners.astype(int)], True, (0,255,0), 2)
            if geometry and detector:
                detections = detector.predict(geometry.rectify(frame))
                squares, errors = assign_detections(detections, geometry.size, args.orientation)
                if errors or sum(label.endswith("_king") for label in squares.values()) != 2:
                    status, _ = tracker.update(None)
                else:
                    status, _ = tracker.update(placement_from_squares(squares))
                if args.debug:
                    cv2.putText(frame, f"detections={len(detections)} errors={len(errors)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,255,255), 2)
            elif geometry: status = "provide --weights for recognition"
            if pending and pending.done():
                try: analysis = pending.result()
                except Exception as exc: status = f"engine error: {exc}"
                pending = None
            fen = state.fen()
            if engine and geometry and not pending and fen != analyzed_fen and status != "uncertain":
                analyzed_fen = fen
                pending = executor.submit(engine.analyse, state.board.copy())
            if geometry and analysis and analysis.best_move and analyzed_fen == fen and status not in ("uncertain", "waiting"):
                move = analysis.best_move
                draw_move(frame, HudMove(move, move[:2], move[2:4], analysis.score_cp, analysis.pv), geometry.corners, args.orientation)
                cv2.putText(frame, f"{move} cp={analysis.score_cp}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 2)
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 2)
            cv2.imshow("Chess Vision AI", frame)
            key = cv2.waitKey(1) & 0xff
            if key == 27: break
            if key == ord("r"): manual = None
    finally:
        cap.release()
        cv2.destroyAllWindows()
        executor.shutdown(wait=True)
        if engine: engine.close()

if __name__ == "__main__": main()
