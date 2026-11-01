"""Package selected offline model files with hashes; never add weights to Git."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("weights", type=Path)
    p.add_argument("onnx", type=Path)
    p.add_argument("--card", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    files = {"model.pt": args.weights, "model.onnx": args.onnx,
             "model.onnx.json": Path(str(args.onnx)+".json"), "MODEL_CARD.md": args.card,
             "docs/vision-training.md": Path("docs/vision-training.md")}
    files.update({p.as_posix():p for p in Path("reports").iterdir()
                  if p.is_file() and p.name != "model-package.json"})
    if args.output.exists():
        raise FileExistsError(args.output)
    payloads = {name:path.read_bytes() for name,path in files.items()}
    for name,data in payloads.items():
        if name.endswith(".json"):
            json.loads(data.decode("utf-8-sig"))
    payloads["SOURCE_COMMIT"] = subprocess.run(["git","rev-parse","HEAD"],check=True,
                                               capture_output=True).stdout
    hashes = {name:hashlib.sha256(data).hexdigest() for name,data in payloads.items()}
    payloads["SHA256SUMS"] = "".join(f"{value}  {name}\n" for name,value in hashes.items()).encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output,"w",compression=zipfile.ZIP_DEFLATED) as bundle:
        for name,data in payloads.items():
            bundle.writestr(name,data)
    # Read back every member so a corrupted/partial package is never delivered silently.
    with zipfile.ZipFile(args.output) as bundle:
        for name,digest in hashes.items():
            if hashlib.sha256(bundle.read(name)).hexdigest() != digest:
                raise RuntimeError(f"Package verification failed: {name}")
    print(json.dumps(dict(archive=args.output.name,bytes=args.output.stat().st_size,
                         sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),members=hashes),indent=2))


if __name__ == "__main__":
    main()
