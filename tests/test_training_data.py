"""Generated fixtures test plumbing only, never camera recognition accuracy."""
import json
import cv2
import numpy as np
import pytest
from training.data import CLASSES, import_session, read_manifest, split_records


def make_sample(root, index=0):
    image = root / f"frame{index}.png"
    cv2.imwrite(str(image), np.full((80, 80, 3), index, np.uint8))
    annotation = image.with_suffix(".labels.json")
    labels = {f"{f}{r}": "empty" for f in "abcdefgh" for r in range(1, 9)}
    annotation.write_text(json.dumps(dict(image=image.name, corners=[[0,0],[79,0],[79,79],[0,79]],
                                         orientation="white", labels=labels)))
    return annotation


def test_import_requires_complete_explicit_labels_and_provenance(tmp_path):
    sample = make_sample(tmp_path)
    manifest = tmp_path / "manifest.jsonl"
    import_session(tmp_path, manifest, "session-1", "redmi", "own", "real_camera")
    records = read_manifest(manifest)
    assert len(records) == 1 and records[0]["session"] == "session-1"
    assert len(CLASSES) == 13
    data = json.loads(sample.read_text())
    del data["labels"]["a1"]
    sample.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="64"):
        import_session(tmp_path, tmp_path / "bad.jsonl", "s", "redmi", "own", "real_camera")


def test_session_split_reproducible_and_no_pixel_leakage():
    records = [dict(session=f"s{i//2}", sha256=str(i)) for i in range(12)]
    first = split_records(records, 42)
    assert first == split_records(records, 42)
    assert set(first) == {"train", "val", "test"}
    groups = [{r["session"] for r in rows} for rows in first.values()]
    assert all(not a & b for i, a in enumerate(groups) for b in groups[i+1:])
    records[-1]["sha256"] = records[0]["sha256"]
    with pytest.raises(ValueError, match="duplicate"):
        split_records(records, 42)
    with pytest.raises(ValueError, match="3"):
        split_records(records[:2], 42)
