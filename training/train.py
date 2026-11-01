"""Train a 13-class MobileNet or a 12-class YOLO on frozen session splits."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import subprocess

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from training.data import CLASSES, SQUARES, audit, crops, write_json
from vision.square_recognition import build_model, tensor_from_bgr


def validate_splits(splits):
    seen_sessions, seen_hashes = set(), set()
    for name in ("train", "val", "test"):
        rows = splits[name]
        if not rows:
            raise ValueError(f"empty {name} split")
        sessions = {r["session"] for r in rows}
        hashes = [r["sha256"] for r in rows]
        if sessions & seen_sessions or set(hashes) & seen_hashes or len(set(hashes)) != len(hashes):
            raise ValueError("session/image leakage across splits")
        seen_sessions.update(sessions)
        seen_hashes.update(hashes)
        audit(rows)
    missing = audit(splits["train"])["missing_classes"]
    if missing:
        raise ValueError(f"training split lacks classes: {missing}")


class SquaresDataset(Dataset):
    def __init__(self, records, augment=False):
        # Cache small 48px crops, not high-resolution camera frames.
        self.patches, self.labels = [], []
        for record in records:
            patches, data = crops(record)
            self.patches.extend(patches)
            self.labels.extend(CLASSES.index(data["labels"][s]) for s in SQUARES)
        self.augment = augment
        self.jitter = transforms.ColorJitter(brightness=.15, contrast=.15, saturation=.1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        patch = self.patches[index].copy()
        if self.augment:
            # Small residual geometry error after board rectification; no flipped glyphs.
            matrix = cv2.getRotationMatrix2D((24, 24), random.uniform(-4, 4), random.uniform(.95, 1.05))
            matrix[:, 2] += [random.uniform(-1, 1), random.uniform(-1, 1)]
            patch = cv2.warpAffine(patch, matrix, (48, 48), borderMode=cv2.BORDER_REFLECT_101)
            if random.random() < .3:
                patch = cv2.GaussianBlur(patch, (3, 3), random.uniform(.1, .8))
            if random.random() < .3:
                ok, encoded = cv2.imencode(".jpg", patch, [cv2.IMWRITE_JPEG_QUALITY, random.randint(65, 95)])
                if ok:
                    patch = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            rgb = torch.from_numpy(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).copy()).float()/255
            rgb = self.jitter(rgb)
            x = transforms.functional.normalize(rgb, [.485, .456, .406], [.229, .224, .225])
        else:
            x = tensor_from_bgr([patch])[0]
        return x, self.labels[index]


def fit_epoch(model, loader, device, optimizer=None, scaler=None):
    model.train(optimizer is not None)
    loss_sum, correct, count = 0., 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None), torch.autocast(
                device_type=torch.device(device).type, enabled=str(device).startswith("cuda")):
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite loss; refusing corrupt checkpoint")
        if optimizer is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        count += len(y)
        loss_sum += float(loss.detach())*len(y)
        correct += int((logits.argmax(1) == y).sum())
    return dict(loss=loss_sum/count, accuracy=correct/count)


def train_squares(splits, config, output, pretrained=True):
    device = config["device"]
    model = build_model(pretrained).to(device)
    dataset = SquaresDataset(splits["train"], augment=True)
    validation = SquaresDataset(splits["val"])
    batch = min(config["square_batch"], len(dataset))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))
    # A real forward/backward preflight before the first epoch. Reduce on CUDA OOM only.
    while True:
        try:
            probe_x, probe_y = next(iter(DataLoader(dataset, batch_size=batch)))
            fit_epoch(model, [(probe_x, probe_y)], device, optimizer, scaler)
            break
        except torch.cuda.OutOfMemoryError:
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            if batch <= 2:
                raise
            batch = max(2, batch//2)
    write_json(output / "memory_preflight.json", dict(batch=batch, device=device,
               peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if device.startswith("cuda") else None))
    generator = torch.Generator().manual_seed(config["seed"])
    loader = DataLoader(dataset, batch_size=batch, shuffle=True, generator=generator, num_workers=0, drop_last=True)
    val_loader = DataLoader(validation, batch_size=batch, num_workers=0)
    best, stalled = float("inf"), 0
    for epoch in range(config["epochs"]):
        train = fit_epoch(model, loader, device, optimizer, scaler)
        val = fit_epoch(model, val_loader, device)
        entry = dict(epoch=epoch+1, train=train, val=val)
        with (output / "epochs.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry)+"\n")
        checkpoint = dict(state_dict={k: v.cpu() for k, v in model.state_dict().items()},
                          classes=list(CLASSES), input_size=48, config=config, epoch=epoch+1,
                          val=val, optimizer=optimizer.state_dict(), scaler=scaler.state_dict())
        torch.save(checkpoint, output / "last.pt")
        if val["loss"] < best:
            best, stalled = val["loss"], 0
            torch.save(checkpoint, output / "best.pt")
        else:
            stalled += 1
        print(json.dumps(entry), flush=True)
        if stalled >= config["patience"]:
            break


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("approach", choices=("squares", "yolo"))
    p.add_argument("splits", type=Path)
    p.add_argument("--config", type=Path, default=Path("configs/training.json"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    splits = json.loads(args.splits.read_text())
    validate_splits(splits)
    if config["device"].startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; run scripts.inspect_gpu")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "config.json", config)
    write_json(args.output / "splits.json", splits)
    write_json(args.output / "provenance.json", dict(
        splits_sha256=hashlib.sha256(args.splits.read_bytes()).hexdigest(),
        packages={name: importlib.metadata.version(name) for name in ("torch", "torchvision", "numpy", "opencv-python")},
        git_commit=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        real_camera_accuracy_claim=False))
    if args.approach == "squares":
        train_squares(splits, config, args.output)
    else:
        from training.yolo import train_yolo
        train_yolo(splits, config, args.output)


if __name__ == "__main__":
    main()
