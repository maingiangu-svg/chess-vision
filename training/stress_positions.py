"""Diagnostic transforms of public validation renders; NOT real-camera evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import cv2
import numpy as np
from training.data import CLASSES, write_json
from training.evaluate_positions import IMAGE_SQUARES
from training.metrics import summarize
from vision.board_detection import BoardGeometry
from vision.square_recognition import SquareRecognizer


def transform(frame, name, labels):
    image = frame.copy()
    orientation = "white"
    if name == "blur":
        image = cv2.GaussianBlur(image, (7,7), 1.5)
    elif name == "jpeg35":
        image = cv2.imdecode(cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY,35])[1], cv2.IMREAD_COLOR)
    elif name == "rotate90":
        image, orientation = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), "white_left"
    elif name == "rotate180":
        image, orientation = cv2.rotate(image, cv2.ROTATE_180), "black"
    elif name in ("highlight", "occlusion"):
        index = next(i for i,v in enumerate(labels) if v != 12)
        row, col = divmod(index,8)
        tile = image[row*50:(row+1)*50,col*50:(col+1)*50]
        if name == "highlight":
            tile[:] = cv2.addWeighted(tile,.65,np.full_like(tile,(40,230,230)),.35,0)
        else:
            tile[15:35,15:35] = 140
    elif name == "perspective_rectified":
        source = np.array([[0,0],[399,0],[399,399],[0,399]],np.float32)
        corners = np.array([[120,50],[550,100],[500,560],[40,480]],np.float32)
        warped = cv2.warpPerspective(image,cv2.getPerspectiveTransform(source,corners),(600,600))
        image = BoardGeometry(corners,size=400).rectify(warped)
    return image, orientation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("weights", type=Path)
    p.add_argument("splits", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--boards", type=int, default=512)
    args = p.parse_args()
    import torch
    torch.set_num_threads(4)
    records = json.loads(args.splits.read_text())["val"]
    subset = random.Random(42).sample(records,min(args.boards,len(records)))
    model = SquareRecognizer(args.weights,device="cuda:0")
    report = dict(domain="diagnostic transforms of synthetic validation renders", real_camera_accuracy=None,
                  seed=42, boards_per_case=len(subset), confidence=.9, margin=.25,
                  weights_sha256=hashlib.sha256(args.weights.read_bytes()).hexdigest(),
                  splits_sha256=hashlib.sha256(args.splits.read_bytes()).hexdigest(), cases={})
    for case in ("clean","blur","jpeg35","rotate90","rotate180","highlight","occlusion","perspective_rectified"):
        rows = []
        for record in subset:
            frame = cv2.imread(record["image"])
            image, orientation = transform(frame,case,record["labels"])
            predicted, _ = model.predict_labels(image,orientation)
            truth = {s:CLASSES[v] for s,v in zip(IMAGE_SQUARES,record["labels"])}
            rows.append(dict(truth=truth,predicted=predicted,tags=[case],latency_ms=0.))
        metrics = summarize(rows)
        report["cases"][case] = {k:metrics[k] for k in ("boards","square_accuracy","full_board_exact_match","abstention_rate","per_class","confusion_matrix")}
        print(json.dumps(dict(case=case,square_accuracy=metrics["square_accuracy"],full_board_exact_match=metrics["full_board_exact_match"])),flush=True)
    write_json(args.output, report)


if __name__ == "__main__":
    main()
