import cv2
from vision.square_mapping import point_for_square
from shared.schemas import HudMove

def draw_move(frame, move: HudMove, corners, orientation="white"):
    import numpy as np
    src = np.asarray(corners, dtype=np.float32)
    dst = np.array([[0,0],[799,0],[799,799],[0,799]], dtype=np.float32)
    inv = cv2.getPerspectiveTransform(dst, src)
    pts = cv2.perspectiveTransform(np.array([[point_for_square(move.from_square)], [point_for_square(move.to_square)]], dtype=np.float32), inv).reshape(2,2).astype(int)
    return cv2.arrowedLine(frame, tuple(pts[0]), tuple(pts[1]), (0, 220, 80), 5, tipLength=.2)
