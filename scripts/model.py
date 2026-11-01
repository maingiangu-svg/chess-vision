"""Train/validate/export a local chess-piece detector using real YOLO data."""
import argparse
from pathlib import Path
import yaml
from shared.schemas import PIECE_CLASSES

def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("train", "val", "export"))
    p.add_argument("--data", help="YOLO dataset YAML with train/val image paths")
    p.add_argument("--weights", required=True)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    from ultralytics import YOLO
    if args.action in ("train", "val"):
        if not args.data: p.error("--data is required for train/val")
        config = yaml.safe_load(Path(args.data).read_text())
        names = config["names"]
        ordered = [names[i] for i in range(12)] if isinstance(names, dict) else names
        if tuple(ordered) != PIECE_CLASSES: p.error("dataset names must match shared.schemas.PIECE_CLASSES in order")
    model = YOLO(args.weights)
    if args.action == "train": model.train(data=args.data, epochs=args.epochs, imgsz=640, batch=-1, device=args.device)
    elif args.action == "val": model.val(data=args.data, device=args.device)
    else: model.export(format="onnx")

if __name__ == "__main__": main()
