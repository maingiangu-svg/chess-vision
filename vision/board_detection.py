from dataclasses import dataclass
import cv2
import numpy as np

@dataclass(frozen=True)
class BoardGeometry:
    corners: np.ndarray  # TL, TR, BR, BL in source-image coordinates
    size: int = 800

    def homography(self) -> tuple[np.ndarray, np.ndarray]:
        dst = np.array([[0, 0], [self.size - 1, 0], [self.size - 1, self.size - 1], [0, self.size - 1]], dtype=np.float32)
        h = cv2.getPerspectiveTransform(self.corners.astype(np.float32), dst)
        return h, np.linalg.inv(h)

    def rectify(self, frame: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(frame, self.homography()[0], (self.size, self.size))

def order_corners(points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=np.float32).reshape(4, 2)
    s, d = p.sum(axis=1), np.diff(p, axis=1).ravel()
    return np.array([p[np.argmin(s)], p[np.argmin(d)], p[np.argmax(s)], p[np.argmax(d)]], dtype=np.float32)

def detect_board_corners(frame: np.ndarray) -> BoardGeometry | None:
    def score_corners(corners: np.ndarray) -> float:
        patch = BoardGeometry(corners, 256).rectify(frame)
        lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32)
        cells = np.array([[np.median(lab[r*32+10:r*32+22, col*32+10:col*32+22], axis=(0,1)) for col in range(8)] for r in range(8)])
        even = np.mean([cells[r,col] for r in range(8) for col in range(8) if (r+col)%2==0], axis=0)
        odd = np.mean([cells[r,col] for r in range(8) for col in range(8) if (r+col)%2], axis=0)
        return float(np.linalg.norm(even-odd))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    area_limit = frame.shape[0] * frame.shape[1] * 0.08
    for c in contours:
        area = cv2.contourArea(c)
        if area < area_limit: continue
        approx = cv2.approxPolyDP(c, 0.03 * cv2.arcLength(c, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            corners = order_corners(approx.reshape(4, 2))
            if np.min(corners[:,0]) <= 2 or np.max(corners[:,0]) >= frame.shape[1]-3 or np.min(corners[:,1]) <= 2 or np.max(corners[:,1]) >= frame.shape[0]-3:
                continue
            contrast = score_corners(corners)
            score = contrast + 3 * area/(frame.shape[0]*frame.shape[1])
            if contrast >= 8 and (best is None or score > best[0]): best = (score, corners)
    # Chess.com green-board fallback when the outer border touches the frame.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (30, 25, 25), (95, 255, 255))
    radius = max(9, round(min(frame.shape[:2]) * .025))
    joined = cv2.dilate(green, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius*2+1, radius*2+1)))
    regions, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for region in regions:
        if best is not None: break
        if cv2.contourArea(region) < area_limit: continue
        box = cv2.boxPoints(cv2.minAreaRect(region)).astype(np.float32)
        corners = order_corners(box)
        if np.min(corners[:,0]) <= 2 or np.max(corners[:,0]) >= frame.shape[1]-3 or np.min(corners[:,1]) <= 2 or np.max(corners[:,1]) >= frame.shape[0]-3:
            continue
        contrast = score_corners(corners)
        if contrast >= 8: best = (contrast, corners)
    return BoardGeometry(best[1]) if best else None
