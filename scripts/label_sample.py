"""Manually annotate a saved camera frame, one board square at a time.

Click four board corners if no companion calibration JSON exists. Then click a
square and press 0=empty, 1..6=white P/N/B/R/Q/K, qwerty=black P/N/B/R/Q/K.
Press s to save JSON, Esc to quit. No labels are inferred automatically.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from vision.board_detection import BoardGeometry, order_corners
from vision.square_mapping import square_from_point, point_for_square

KEYS = dict(zip("0123456qwerty", ("empty", "white_pawn", "white_knight", "white_bishop", "white_rook", "white_queen", "white_king", "black_pawn", "black_knight", "black_bishop", "black_rook", "black_queen", "black_king")))

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("image", type=Path)
    p.add_argument("--orientation", choices=("white", "black", "white_left", "white_right"), default="white")
    args = p.parse_args()
    frame = cv2.imread(str(args.image))
    if frame is None: p.error("image cannot be read")
    companion = Path(f"{args.image}.json")
    metadata = json.loads(companion.read_text()) if companion.exists() else {}
    orientation = metadata.get("orientation", args.orientation)
    corners = list(metadata.get("corners", []))
    labels: dict[str, str] = {}
    selected = None
    def click(event, x, y, flags, userdata):
        nonlocal selected
        if event != cv2.EVENT_LBUTTONDOWN: return
        if len(corners) < 4: corners.append([x, y])
        else:
            geometry = BoardGeometry(order_corners(np.asarray(corners, np.float32)))
            point = cv2.perspectiveTransform(np.array([[[x, y]]], dtype=np.float32), geometry.homography()[0])[0, 0]
            try: selected = square_from_point(*point, geometry.size, orientation)
            except ValueError: selected = None
    cv2.namedWindow("Label camera sample")
    cv2.setMouseCallback("Label camera sample", click)
    while True:
        display = frame.copy()
        if len(corners) == 4:
            geometry = BoardGeometry(order_corners(np.asarray(corners, np.float32)))
            cv2.polylines(display, [geometry.corners.astype(int)], True, (0,255,0), 2)
            inv = geometry.homography()[1]
            for square, label in labels.items():
                point = np.array([[point_for_square(square, geometry.size, orientation)]], dtype=np.float32)
                x, y = cv2.perspectiveTransform(point, inv)[0,0].astype(int)
                cv2.putText(display, f"{square}:{label[:2]}", (x-20,y), cv2.FONT_HERSHEY_SIMPLEX, .35, (0,255,255), 1)
        cv2.putText(display, f"Corners {len(corners)}/4 | selected {selected or '-'} | labels {len(labels)}/64", (10,25), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 2)
        cv2.imshow("Label camera sample", display)
        key = cv2.waitKey(30) & 0xff
        if key == 27: break
        if selected and chr(key) in KEYS: labels[selected] = KEYS[chr(key)]
        if key == ord("s") and len(corners) == 4:
            output = args.image.with_suffix(".labels.json")
            output.write_text(json.dumps({"image": str(args.image), "corners": geometry.corners.tolist(), "orientation": orientation, "labels": labels}, indent=2))
            print(f"Saved {len(labels)}/64 labels to {output}")
    cv2.destroyAllWindows()

if __name__ == "__main__": main()
