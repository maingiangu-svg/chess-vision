"""Public 2D boards: labels from filename FEN; position-group split before tiles."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import chess
import cv2
import numpy as np
from training.data import CLASSES, write_json
from vision.template_recognition import SYMBOL_TO_LABEL


def labels_from_filename(path):
    placement = Path(path).stem.replace("-", "/")
    board = chess.Board(None)
    board.set_board_fen(placement)  # Validate rank widths and FEN symbols; legality is separate.
    labels = [CLASSES.index(SYMBOL_TO_LABEL[p.symbol()]) if (p := board.piece_at(chess.square(col, 7-row))) else 12
              for row in range(8) for col in range(8)]
    return board.board_fen(), labels


def inspect_image(path):
    placement, labels = labels_from_filename(path)
    raw = path.read_bytes()
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape != (400, 400, 3):
        raise ValueError(f"Expected 400x400 RGB board: {path}")
    return dict(image=str(path.resolve()), placement=placement, labels=labels,
                pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                file_sha256=hashlib.sha256(raw).hexdigest(), original_split=path.parent.name)


def group_split(records, seed=42):
    pixels, groups, duplicate_images = {}, {}, 0
    for row in records:
        row = dict(row)
        key = row["pixel_sha256"]
        if key in pixels:
            if pixels[key]["placement"] != row["placement"]:
                raise ValueError("Identical pixels have conflicting FEN labels")
            if row["original_split"] == "test":
                pixels[key]["original_split"] = "test"
            duplicate_images += 1
            continue
        pixels[key] = row
        groups.setdefault(row["placement"], []).append(row)
    splits = dict(train=[], val=[], test=[])
    for position, rows in sorted(groups.items()):
        # Keep publisher test groups held out. FEN duplicates touching test move entirely to test.
        if any(r["original_split"] == "test" for r in rows):
            split = "test"
        else:
            value = int(hashlib.sha256(f"{seed}:{position}".encode()).hexdigest()[:8], 16)/2**32
            split = "val" if value < .125 else "train"
        splits[split].extend(rows)
    if not all(splits.values()):
        raise ValueError("Need nonempty train/val/test; source must include publisher test images")
    return splits, duplicate_images


def prepare(root, output):
    start = time.perf_counter()
    paths = sorted(Path(root).rglob("*.jpeg"))
    if not paths:
        raise ValueError("No downloaded JPEG files")
    records = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for index, row in enumerate(pool.map(inspect_image, paths)):
            records.append(row)
            if (index+1) % 10000 == 0:
                print(f"Audited {index+1}/{len(paths)} boards", flush=True)
    splits, duplicates = group_split(records)
    report = dict(source="koryakinp/chess-positions", domain="synthetic_2d", license="CC0-1.0",
                  images_downloaded=len(records), exact_duplicates_removed=duplicates,
                  unique_position_groups=len({r["placement"] for r in records}),
                  split_definition="publisher test reserved by whole FEN group; 12.5% of remaining FEN groups validation; no tile-level random splitting",
                  theme_generalization="same source assets may appear across splits; not an unseen-theme or camera benchmark",
                  splits={}, audit_seconds=time.perf_counter()-start)
    for name, rows in splits.items():
        counts = Counter(label for row in rows for label in row["labels"])
        report["splits"][name] = dict(boards=len(rows), squares=len(rows)*64,
            position_groups=len({r["placement"] for r in rows}),
            class_counts={label: counts[i] for i, label in enumerate(CLASSES)})
    write_json(output / "splits.json", splits)
    write_json(output / "audit.json", report)
    write_json("reports/positions-audit.json", report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--output", type=Path, default=Path("data/positions-frozen"))
    args = p.parse_args()
    if (args.output / "splits.json").exists():
        raise FileExistsError("Frozen splits already exist; preserve the original evaluation split")
    prepare(args.root, args.output)


if __name__ == "__main__":
    main()
