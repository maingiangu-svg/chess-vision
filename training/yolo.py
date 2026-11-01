"""YOLO dataset conversion uses manual piece boxes, never fabricated square boxes."""
from collections import Counter
import os
from pathlib import Path
import cv2
import numpy as np
import yaml
from shared.schemas import PIECE_CLASSES
from training.data import board_and_labels


def prepare_yolo(splits, root):
    root = Path(root).resolve()
    for split, rows in splits.items():
        images, labels = root / "images" / split, root / "labels" / split
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        for record in rows:
            board, data = board_and_labels(record)
            boxes = data.get("boxes", [])
            expected = Counter(v for v in data["labels"].values() if v != "empty")
            if Counter(b["label"] for b in boxes) != expected:
                raise ValueError("YOLO requires one manually annotated piece box per occupied square, with matching class counts")
            from vision.board_detection import BoardGeometry
            geometry = BoardGeometry(np.asarray(data["corners"], np.float32))
            homography = geometry.homography()[0]
            lines = []
            for box in boxes:
                x1, y1, x2, y2 = box["bbox"]
                points = np.array([[[x1,y1],[x2,y1],[x2,y2],[x1,y2]]], np.float32)
                points = cv2.perspectiveTransform(points, homography)[0]
                if (points < -1).any() or (points > 800).any():
                    raise ValueError("Box extends outside rectified board; review annotation")
                lo, hi = np.clip(points.min(0), 0, 800), np.clip(points.max(0), 0, 800)
                center, extent = (lo+hi)/1600, (hi-lo)/800
                if (extent <= 0).any():
                    raise ValueError("Degenerate transformed piece box")
                lines.append(" ".join(map(str, [PIECE_CLASSES.index(box["label"]), *center, *extent])))
            name = record["sha256"]
            if not cv2.imwrite(str(images / f"{name}.png"), board):
                raise OSError("Cannot write rectified image")
            (labels / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")
    path = root / "dataset.yaml"
    path.write_text(yaml.safe_dump(dict(path=str(root), train="images/train", val="images/val",
                                     test="images/test", names=dict(enumerate(PIECE_CLASSES)))) , encoding="utf-8")
    return path


def train_yolo(splits, config, output):
    # Keep font/config downloads on the workspace volume (AppData can be redirected on Windows).
    os.environ.setdefault("YOLO_CONFIG_DIR", str((Path(output) / "runtime").resolve()))
    from ultralytics import YOLO
    data = prepare_yolo(splits, output / "dataset")
    model = YOLO(config["pretrained_detector"])
    # Ultralytics performs training, AMP checks, validation, checkpointing and early stopping.
    model.train(data=str(data), epochs=config["epochs"], patience=config["patience"],
                imgsz=config["imgsz"], batch=config["yolo_batch"], device=config["device"].replace("cuda:", ""),
                workers=config["workers"], seed=config["seed"], deterministic=True, amp=True,
                project=str(output), name="detector", exist_ok=False, pretrained=True,
                lr0=config["lr"], degrees=6, translate=.03, scale=.15, perspective=.0001,
                fliplr=0., flipud=0., mosaic=0., mixup=0., hsv_h=.01, hsv_s=.15, hsv_v=.15)
