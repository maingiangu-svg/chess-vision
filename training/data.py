"""Explicit camera-label import, provenance, grouped split and rectification."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

import cv2
import numpy as np
from shared.schemas import PIECE_CLASSES
from vision.board_detection import BoardGeometry
from vision.template_recognition import square_image

CLASSES = (*PIECE_CLASSES, "empty")
SQUARES = tuple(f"{f}{r}" for r in range(1, 9) for f in "abcdefgh")
ORIENTATIONS = ("white", "black", "white_left", "white_right")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def load_annotation(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    image = Path(data["image"])
    if not image.is_absolute():
        image = path.parent / image
    if not image.is_file():
        # Mac labels may retain the old absolute path; require the local basename.
        image = path.parent / data["image"].replace("\\", "/").split("/")[-1]
    frame = cv2.imread(str(image))
    if frame is None:
        raise ValueError(f"Unreadable image for {path}")
    labels = data.get("labels", {})
    if set(labels) != set(SQUARES) or set(labels.values()) - set(CLASSES):
        raise ValueError(f"{path}: require all 64 explicit labels from shared.schemas plus empty")
    if data.get("orientation") not in ORIENTATIONS:
        raise ValueError(f"{path}: invalid orientation")
    corners = np.asarray(data.get("corners"), dtype=np.float32)
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        raise ValueError(f"{path}: require four finite TL/TR/BR/BL corners")
    h, w = frame.shape[:2]
    if (corners < 0).any() or (corners[:, 0] >= w).any() or (corners[:, 1] >= h).any():
        raise ValueError(f"{path}: corners outside image")
    contour = corners.reshape(-1, 1, 2)
    if not cv2.isContourConvex(contour) or cv2.contourArea(contour) < 64:
        raise ValueError(f"{path}: degenerate or crossed corners")
    # The existing app stores corners in this order. Do not silently reorder labels.
    if not (corners[0].sum() == corners.sum(axis=1).min() and
            corners[2].sum() == corners.sum(axis=1).max()):
        raise ValueError(f"{path}: corners must be TL, TR, BR, BL")
    for box in data.get("boxes", []):
        coords = np.asarray(box.get("bbox"), dtype=float)
        if box.get("label") not in PIECE_CLASSES or coords.shape != (4,) or not np.isfinite(coords).all():
            raise ValueError(f"{path}: invalid manually annotated box")
        x1, y1, x2, y2 = coords
        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            raise ValueError(f"{path}: bounding box outside image")
    data["image"] = str(image.resolve())
    return data, frame


def import_session(directory, manifest, session, source, license_id, domain):
    if not all(str(v).strip() for v in (session, source, license_id)):
        raise ValueError("session, source and license are mandatory")
    if domain not in ("real_camera", "screenshot", "synthetic"):
        raise ValueError("invalid domain")
    records = []
    for path in sorted(Path(directory).rglob("*.labels.json")):
        data, frame = load_annotation(path)
        records.append(dict(annotation=str(path.resolve()), image=data["image"], session=session,
                            source=source, license=license_id, domain=domain,
                            annotation_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            sha256=hashlib.sha256(frame.tobytes()).hexdigest(),
                            tags=data.get("tags", [])))
    if not records:
        raise ValueError("No .labels.json files found; real camera data is missing")
    manifest = Path(manifest)
    existing = read_manifest(manifest) if manifest.exists() else []
    paths = {r["annotation"] for r in existing}
    if paths & {r["annotation"] for r in records}:
        raise ValueError("annotation already imported")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("".join(json.dumps(r)+"\n" for r in existing+records), encoding="utf-8")
    return records


def read_manifest(path):
    records = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError("Empty manifest")
    return records


def split_records(records, seed=42):
    hashes = [r["sha256"] for r in records]
    if len(set(hashes)) != len(hashes):
        raise ValueError("duplicate decoded images; deduplicate before splitting")
    sessions = sorted({r["session"] for r in records})
    if len(sessions) < 3:
        raise ValueError("At least 3 independent capture sessions/source groups are required")
    random.Random(seed).shuffle(sessions)
    ntest = max(1, round(len(sessions)*.15))
    nval = max(1, round(len(sessions)*.15))
    groups = dict(test=set(sessions[:ntest]), val=set(sessions[ntest:ntest+nval]),
                  train=set(sessions[ntest+nval:]))
    return {split: [r for r in records if r["session"] in group] for split, group in groups.items()}


def audit(records):
    counts = Counter()
    boxes = 0
    for record in records:
        if hashlib.sha256(Path(record["annotation"]).read_bytes()).hexdigest() != record["annotation_sha256"]:
            raise ValueError("Annotation changed after import; reimport and freeze splits again")
        data, frame = load_annotation(record["annotation"])
        if hashlib.sha256(frame.tobytes()).hexdigest() != record["sha256"]:
            raise ValueError("Image changed after import")
        counts.update(data["labels"].values())
        boxes += len(data.get("boxes", []))
    return dict(images=len(records), sessions=len({r["session"] for r in records}),
                domains=dict(Counter(r["domain"] for r in records)),
                classes={name: counts[name] for name in CLASSES}, manual_boxes=boxes,
                missing_classes=[name for name in CLASSES if not counts[name]])


def board_and_labels(record):
    data, frame = load_annotation(record["annotation"])
    return BoardGeometry(np.asarray(data["corners"], np.float32)).rectify(frame), data


def crops(record):
    board, data = board_and_labels(record)
    return np.stack([square_image(board, s, data["orientation"]) for s in SQUARES]), data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    imp = commands.add_parser("import")
    imp.add_argument("directory", type=Path)
    imp.add_argument("manifest", type=Path)
    for flag in ("session", "source", "license", "domain"):
        imp.add_argument(f"--{flag}", required=True)
    split = commands.add_parser("split")
    split.add_argument("manifest", type=Path)
    split.add_argument("output", type=Path)
    split.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.command == "import":
        records = import_session(args.directory, args.manifest, args.session, args.source, args.license, args.domain)
        print(json.dumps(audit(records), indent=2))
    else:
        records = read_manifest(args.manifest)
        report = audit(records)
        splits = split_records(records, args.seed)
        report["splits"] = {s: audit(rows) for s, rows in splits.items()}
        write_json(args.output / "splits.json", splits)
        write_json(args.output / "audit.json", report)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
