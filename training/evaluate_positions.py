"""Independent public-2D test metrics. No Redmi/camera generalization claims."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from training.data import CLASSES, write_json
from training.metrics import summarize, plausible
from training.train_positions import PositionBoards, normalize, seed_worker
from vision.square_recognition import SquareRecognizer

IMAGE_SQUARES = tuple(f"{f}{r}" for r in range(8,0,-1) for f in "abcdefgh")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("weights", type=Path)
    p.add_argument("splits", type=Path)
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--confidence", type=float, default=.9)
    p.add_argument("--margin", type=float, default=.25)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve previous evaluation history")
    torch.set_num_threads(4)
    records = json.loads(args.splits.read_text())[args.split]
    recognizer = SquareRecognizer(args.weights, args.device, args.confidence, args.margin)
    loader = DataLoader(PositionBoards(records), batch_size=16, num_workers=args.workers,
                        worker_init_fn=seed_worker, pin_memory=args.device.startswith("cuda"))
    rows, raw_rows = [], []
    start = time.perf_counter()
    offset = 0
    for tiles, labels in loader:
        x = normalize(tiles.flatten(0,1))
        if recognizer.backend == ".onnx":
            logits = torch.from_numpy(recognizer.model.run(None, {"squares": x.numpy()})[0])
        else:
            with torch.inference_mode():
                logits = recognizer.model(x.to(args.device)).cpu()
        probs = logits.float().softmax(1)
        scores, indices = probs.topk(2, dim=1)
        confident = (scores[:,0] >= args.confidence) & (scores[:,0]-scores[:,1] >= args.margin)
        indices = indices[:,0].reshape(-1,64).numpy()
        confident = confident.reshape(-1,64).numpy()
        for i, target in enumerate(labels.numpy()):
            record = records[offset+i]
            truth = {s: CLASSES[int(label)] for s,label in zip(IMAGE_SQUARES,target)}
            raw = {s: CLASSES[int(label)] for s,label in zip(IMAGE_SQUARES,indices[i])}
            predicted = {s: raw[s] if confident[i,j] else "uncertain" for j,s in enumerate(IMAGE_SQUARES)}
            row = dict(image=record["image"], truth=truth, predicted=predicted,
                       tags=["synthetic_2d"], latency_ms=0.)
            rows.append(row)
            raw_rows.append({**row, "predicted": raw})
        offset += len(labels)
        if offset % 2000 == 0:
            print(f"Evaluated {offset}/{len(records)} boards", flush=True)
    elapsed = time.perf_counter()-start
    report = summarize(rows)
    raw_report = summarize(raw_rows)
    # Single-board adapter timing excludes JPEG decode, includes crops/normalization/transfers.
    latencies = []
    import cv2
    for record in records[:131]:
        frame = cv2.imread(record["image"])
        start = time.perf_counter()
        recognizer.predict_labels(frame, "white")
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        latencies.append((time.perf_counter()-start)*1000)
    latencies = latencies[3:]
    report["latency_ms"] = dict(median=float(np.median(latencies)), p95=float(np.percentile(latencies,95)),
        boards=len(latencies), scope="single rectified board to 64 predictions; crop/normalization/transfer included, image decode/geometry/network/Stockfish excluded")
    report.update(domain="synthetic_2d", real_camera_accuracy=None, split=args.split,
        raw_argmax=dict(square_accuracy=raw_report["square_accuracy"], full_board_exact_match=raw_report["full_board_exact_match"],
                        confusion_matrix=raw_report["confusion_matrix"], per_class=raw_report["per_class"]),
        ground_truth_invalid_position_frequency=sum(not plausible(r["truth"]) for r in rows)/len(rows),
        confidence=args.confidence, margin=args.margin, batch_evaluation_seconds=elapsed,
        hardware=dict(os=platform.platform(), device=args.device, torch=torch.__version__,
                      gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        weights_sha256=hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        splits_sha256=hashlib.sha256(args.splits.read_bytes()).hexdigest())
    # Keep all failures on disk, but avoid duplicating them in the console.
    write_json(args.output, report)
    print(json.dumps({k:report[k] for k in ("boards","square_accuracy","full_board_exact_match","abstention_rate","latency_ms")},indent=2))


if __name__ == "__main__":
    main()
