"""Exercise pretrained models on random tensors. Never an accuracy benchmark."""
import argparse
import json
from pathlib import Path
import time
import torch
from training.data import write_json
from vision.square_recognition import build_model


def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from tensors(child)


def measure(model, shape):
    # Ultralytics loads pretrained inference weights with requires_grad=False.
    model = model.cuda().train().requires_grad_(True)
    x = torch.rand(*shape, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.float16):
        outputs = list(tensors(model(x)))
        # Output-energy backward checks kernels/autograd without inventing chess labels.
        loss = sum(t.float().square().mean() for t in outputs if t.requires_grad)
    loss.backward()
    torch.cuda.synchronize()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    report = dict(input_shape=list(shape), amp=True, forward_backward_ms=(time.perf_counter()-start)*1000,
                  peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                  finite_output=bool(torch.isfinite(loss)),
                  finite_gradients=all(bool(torch.isfinite(g).all()) for g in gradients),
                  gradient_tensors=len(gradients))
    model.zero_grad(set_to_none=True)
    del x, loss, outputs, gradients
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("reports/model-smoke.json"))
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this hardware check")
    torch.manual_seed(42)
    small = build_model(pretrained=True)
    square_report = measure(small, (64, 3, 48, 48))
    del small
    torch.cuda.empty_cache()
    from ultralytics import YOLO
    detector = YOLO("yolo11n.pt")
    yolo_report = measure(detector.model, (4, 3, 640, 640))
    report = dict(gpu=torch.cuda.get_device_name(0), square_mobilenet=square_report,
                  yolo11n=yolo_report, real_camera_images=0, real_camera_accuracy=None,
                  note="Pretrained initialization + random tensor output-energy backward; not supervised chess training, not accuracy or inference latency.")
    write_json(args.output, report)
    print(json.dumps(report, indent=2))
    if not all(r["finite_output"] and r["finite_gradients"] and r["gradient_tensors"] for r in (square_report, yolo_report)):
        raise RuntimeError("Nonfinite model smoke result")


if __name__ == "__main__":
    main()
