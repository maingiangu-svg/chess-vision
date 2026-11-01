"""Download public Chess Positions through the official KaggleHub SDK."""
import argparse
from pathlib import Path
import json
import kagglehub
import requests
from training.data import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/chess-positions"))
    args = parser.parse_args()
    # Metadata comes from the publisher's public API, not the mirror's row counter.
    response = requests.get("https://www.kaggle.com/api/v1/datasets/list", params={"search": "chess-positions"}, timeout=60)
    response.raise_for_status()
    metadata = next(r for r in response.json() if r["ref"] == "koryakinp/chess-positions")
    if metadata["licenseName"] != "CC0: Public Domain":
        raise ValueError("Dataset license changed; review the publisher metadata")
    write_json("reports/kaggle-source.json", {k: metadata.get(k) for k in
               ("ref", "title", "licenseName", "totalBytes", "lastUpdated", "currentVersionNumber")})
    handle = f"koryakinp/chess-positions/versions/{metadata['currentVersionNumber']}"
    path = kagglehub.dataset_download(handle, output_dir=str(args.output.resolve()))
    images = list(Path(path).rglob("*.jpeg"))
    report = dict(path=path, handle=handle, jpeg_images=len(images), tool="kagglehub", real_camera=False)
    write_json("reports/download-positions.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
