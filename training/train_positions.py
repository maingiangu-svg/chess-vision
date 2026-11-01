"""Memory-bounded GPU training on public FEN-labeled boards, not camera data."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from training.data import CLASSES, write_json
from vision.square_recognition import build_model


class PositionBoards(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        image = cv2.imread(row["image"])
        if image is None or image.shape != (400, 400, 3):
            raise ValueError(f"Unreadable or wrong-size board: {row['image']}")
        rgb = cv2.cvtColor(cv2.resize(image, (384,384), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        tiles = rgb.reshape(8,48,8,48,3).transpose(0,2,4,1,3).reshape(64,3,48,48).copy()
        return torch.from_numpy(tiles), torch.tensor(row["labels"], dtype=torch.long)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def normalize(x, augment=False, camera_augment=False):
    x = x.float()/255
    if augment:
        brightness = torch.empty(len(x),1,1,1,device=x.device).uniform_(.85,1.15)
        contrast = torch.empty(len(x),1,1,1,device=x.device).uniform_(.85,1.15)
        x = ((x-.5)*contrast+.5)*brightness
        x = x.clamp(0,1)
        if camera_augment:
            from torchvision.transforms.functional import gaussian_blur
            # Class labels are invariant to quarter-turns; unlike board coordinates.
            x = torch.rot90(x, random.randrange(4), dims=(2,3))
            if random.random() < .6:
                sigma = random.uniform(.2,1.6)
                x = gaussian_blur(x, [7,7], [sigma,sigma])
            if random.random() < .25:
                tint = x.new_tensor([.9,.85,.15]).view(1,3,1,1)
                opacity = random.uniform(.05,.3)
                x = (1-opacity)*x+opacity*tint
    mean = x.new_tensor([.485,.456,.406]).view(1,3,1,1)
    std = x.new_tensor([.229,.224,.225]).view(1,3,1,1)
    return (x-mean)/std


def epoch(model, loader, device, weights, optimizer=None, scaler=None, confidence=.9, margin=.25,
          camera_augment=False):
    model.train(optimizer is not None)
    count = correct = raw_exact = exact = uncertain = 0
    loss_sum = 0.
    start = time.perf_counter()
    for step, (tiles, labels) in enumerate(loader):
        labels = labels.to(device, non_blocking=True)
        x = normalize(tiles.flatten(0,1).to(device, non_blocking=True), augment=optimizer is not None,
                      camera_augment=camera_augment)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None), torch.autocast("cuda", enabled=device.startswith("cuda")):
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, labels.flatten(), weight=weights)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        if optimizer is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        probabilities = logits.detach().float().softmax(1)
        scores, indices = probabilities.topk(2, dim=1)
        matches = (indices[:,0].reshape_as(labels) == labels)
        confident = ((scores[:,0] >= confidence) & (scores[:,0]-scores[:,1] >= margin)).reshape_as(labels)
        count += labels.numel()
        correct += int(matches.sum())
        raw_exact += int(matches.all(1).sum())
        exact += int((matches & confident).all(1).sum())
        uncertain += int((~confident).sum())
        loss_sum += float(loss.detach())*labels.numel()
        if step and step % 200 == 0:
            print(json.dumps(dict(progress_batches=step, squares=count, elapsed_seconds=time.perf_counter()-start)), flush=True)
    return dict(loss=loss_sum/count, raw_square_accuracy=correct/count,
                raw_board_exact_match=raw_exact/(count/64), full_board_exact_match=exact/(count/64),
                abstention_rate=uncertain/count, squares=count, seconds=time.perf_counter()-start)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("splits", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-boards", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--lr", type=float, default=.001)
    p.add_argument("--initial", type=Path, help="Fine-tune a trusted compatible local checkpoint")
    p.add_argument("--camera-augment", action="store_true", help="Synthetic blur/quarter-turn/tint, not real-camera data")
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("RTX CUDA training was requested; GPU unavailable")
    args.output.mkdir(parents=True, exist_ok=False)
    splits = json.loads(args.splits.read_text())
    positions, pixels = set(), set()
    for name, rows in splits.items():
        pset, hset = {r["placement"] for r in rows}, {r["pixel_sha256"] for r in rows}
        if pset & positions or hset & pixels:
            raise ValueError("FEN/image leakage across splits")
        positions.update(pset)
        pixels.update(hset)
    config = {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    config.update(architecture="tiny_cnn", device="cuda:0", confidence=.9, margin=.25,
                  domain="synthetic_2d", real_camera_accuracy=None)
    write_json(args.output / "config.json", config)
    write_json(args.output / "data-provenance.json", dict(splits_sha256=hashlib.sha256(args.splits.read_bytes()).hexdigest(),
        sizes={s: len(rows) for s,rows in splits.items()}, torch=torch.__version__, cuda=torch.version.cuda))
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = build_model(architecture="tiny_cnn").cuda()
    if args.initial:
        initial = torch.load(args.initial,map_location="cpu",weights_only=True)
        if tuple(initial["classes"]) != CLASSES or initial.get("architecture") != "tiny_cnn":
            raise ValueError("Initial model architecture/class mapping mismatch")
        model.load_state_dict(initial["state_dict"])
    counts = Counter(label for row in splits["train"] for label in row["labels"])
    if set(counts) != set(range(13)):
        raise ValueError("Train split must contain all 13 classes")
    class_weights = torch.tensor([1/(counts[i]**.5) for i in range(13)], device="cuda")
    class_weights /= class_weights.mean()
    generator = torch.Generator().manual_seed(args.seed)
    options = dict(batch_size=args.batch_boards, num_workers=args.workers,
                   pin_memory=True, worker_init_fn=seed_worker, persistent_workers=args.workers > 0)
    train = DataLoader(PositionBoards(splits["train"]), shuffle=True, generator=generator, **options)
    val = DataLoader(PositionBoards(splits["val"]), shuffle=False, **options)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    best, stalled = (-1., -1.), 0
    for index in range(args.epochs):
        torch.cuda.reset_peak_memory_stats()
        training = epoch(model, train, "cuda:0", class_weights, optimizer, scaler,
                         camera_augment=args.camera_augment)
        validation = epoch(model, val, "cuda:0", class_weights)
        scheduler.step()
        entry = dict(epoch=index+1, train=training, val=validation,
                     peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
        with (args.output / "epochs.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry)+"\n")
        checkpoint = dict(state_dict={k:v.cpu() for k,v in model.state_dict().items()},
                          classes=list(CLASSES), architecture="tiny_cnn", input_size=48,
                          config=config, epoch=index+1, val=validation)
        torch.save(checkpoint, args.output / "last.pt")
        criterion = validation["full_board_exact_match"], validation["raw_square_accuracy"]
        if criterion > best:
            best, stalled = criterion, 0
            torch.save(checkpoint, args.output / "best.pt")
        else:
            stalled += 1
        write_json(args.output / "latest.json", entry)
        print(json.dumps(entry), flush=True)
        if stalled >= args.patience:
            break


if __name__ == "__main__":
    main()
