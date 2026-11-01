"""Evaluate untouched boards and detection boxes, recording timing and failures."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
from training.data import SQUARES, board_and_labels, write_json
from training.metrics import summarize


def synchronize(device):
    import torch
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def evaluate_boards(records, model, approach, config):
    rows = []
    device = config["device"]
    for index, record in enumerate(records):
        board, data = board_and_labels(record)
        if index == 0:
            for _ in range(3):
                if approach == "squares":
                    model.predict_labels(board, data["orientation"])
                else:
                    model.predict(board, confidence=config["yolo_confidence"])
            synchronize(device)
        start = time.perf_counter()
        if approach == "squares":
            predicted, scores = model.predict_labels(board, data["orientation"])
            errors = [s for s, v in predicted.items() if v == "uncertain"]
        else:
            from vision.square_mapping import assign_detections
            detections = model.predict(board, confidence=config["yolo_confidence"])
            occupied, errors = assign_detections(detections, 800, data["orientation"])
            predicted = {s: occupied.get(s, "empty") for s in SQUARES}
            scores = {s: None for s in SQUARES}
        synchronize(device)
        latency = (time.perf_counter()-start)*1000
        rows.append(dict(image=record["image"], session=record["session"], domain=record["domain"],
                         tags=sorted(set(record.get("tags", [])) | {data["orientation"]}),
                         truth=data["labels"], predicted=predicted, scores=scores,
                         mapping_errors=errors, latency_ms=latency, fen=data.get("fen")))
    report = summarize(rows)
    report["mapping_error_board_frequency"] = sum(bool(r["mapping_errors"]) for r in rows)/len(rows)
    # Preserve strict all-64 equality and additionally report collision-free exact boards.
    report["usable_exact_match"] = sum(r["truth"] == r["predicted"] and not r["mapping_errors"] for r in rows)/len(rows)
    real = [r for r in rows if r["domain"] == "real_camera"]
    report["real_camera"] = summarize(real) if real else None
    report["latency_scope"] = "rectified board to 64 labels, crop/preprocess/transfer/synchronization included; excludes geometry/camera/network/engine"
    return report, rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("approach", choices=("squares", "yolo"))
    p.add_argument("weights", type=Path)
    p.add_argument("splits", type=Path)
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--config", type=Path, default=Path("configs/training.json"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--boxes", action="store_true", help="Also compute detection P/R/mAP; requires true manual boxes")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite evaluation report; preserve test history")
    from training.train import validate_splits
    splits = json.loads(args.splits.read_text())
    validate_splits(splits)
    config = json.loads(args.config.read_text())
    if args.approach == "squares":
        from vision.square_recognition import SquareRecognizer
        model = SquareRecognizer(args.weights, config["device"], config["confidence"], config["margin"])
    else:
        from vision.piece_detection import YoloPieceDetector
        model = YoloPieceDetector(str(args.weights))
        # Keep the existing adapter; Ultralytics selects device here explicitly.
        if args.weights.suffix == ".pt":
            model.model.to(config["device"])
    report, rows = evaluate_boards(splits[args.split], model, args.approach, config)
    report["detection"] = None
    if args.approach == "yolo" and args.boxes:
        from training.yolo import prepare_yolo
        data = prepare_yolo(splits, args.output.parent / (args.output.stem+"-boxes"))
        metrics = model.model.val(data=str(data), split=args.split, imgsz=config["imgsz"],
                                  device=config["device"].replace("cuda:", ""), plots=True,
                                  project=str(args.output.parent), name=args.output.stem+"-detection", exist_ok=False)
        report["detection"] = dict(precision=float(metrics.box.mp), recall=float(metrics.box.mr),
                                    mAP50=float(metrics.box.map50), mAP50_95=float(metrics.box.map))
    report.update(approach=args.approach, split=args.split, config=config, hardware=platform.platform(),
                  weights_sha256=hashlib.sha256(args.weights.read_bytes()).hexdigest(),
                  splits_sha256=hashlib.sha256(args.splits.read_bytes()).hexdigest())
    write_json(args.output, report)
    write_json(args.output.with_suffix(".boards.json"), rows)
    print(json.dumps({k: report[k] for k in ("boards", "square_accuracy", "full_board_exact_match", "real_camera", "latency_ms")}, indent=2))


if __name__ == "__main__":
    main()
