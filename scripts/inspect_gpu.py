"""Record hardware and execute CUDA forward/backward; no accuracy claims."""
import argparse
import importlib.metadata
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path


def inspect() -> dict:
    report = {"os": platform.platform(), "python": sys.version, "cuda_verified": False}
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap", "--format=csv"],
        capture_output=True, text=True, check=False,
    )
    report["nvidia_smi"] = result.stdout.strip() or result.stderr.strip()
    report["packages"] = {}
    for name in ("torch", "torchvision", "ultralytics", "numpy", "opencv-python"):
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = None
    try:
        import torch
        report.update(cuda_runtime=torch.version.cuda, cuda_available=torch.cuda.is_available())
        if not torch.cuda.is_available():
            return report
        report.update(gpu=torch.cuda.get_device_name(0), capability=torch.cuda.get_device_capability(0),
                      compiled_architectures=torch.cuda.get_arch_list())
        torch.manual_seed(42)
        torch.cuda.reset_peak_memory_stats()
        layer = torch.nn.Conv2d(3, 16, 3, padding=1).cuda()
        x = torch.randn(4, 3, 640, 640, device="cuda")
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16):
            loss = layer(x).square().mean()
        loss.backward()
        torch.cuda.synchronize()
        cold_ms = (time.perf_counter()-start)*1000
        timings = []
        for _ in range(5):
            layer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.float16):
                loss = layer(x).square().mean()
            loss.backward()
            torch.cuda.synchronize()
            timings.append((time.perf_counter()-start)*1000)
        report.update(cuda_verified=bool(torch.isfinite(loss) and torch.isfinite(layer.weight.grad).all()),
                      amp_fp16=True, cold_forward_backward_ms=cold_ms,
                      warm_forward_backward_median_ms=statistics.median(timings),
                      peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                      note="Random tensor hardware smoke check, not model training or accuracy evaluation.")
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/gpu.json"))
    args = parser.parse_args()
    report = inspect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["cuda_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
