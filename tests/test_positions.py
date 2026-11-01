from pathlib import Path
import pytest
from training.positions import labels_from_filename, group_split
from training.data import CLASSES


def test_filename_fen_colors_empty_and_invalid_ranks():
    placement, labels = labels_from_filename(Path("r6k-8-8-8-8-8-8-K6R.jpeg"))
    assert labels[0] == CLASSES.index("black_rook")
    assert labels[7] == CLASSES.index("black_king")
    assert labels[56] == CLASSES.index("white_king")
    assert labels[63] == CLASSES.index("white_rook")
    assert labels.count(12) == 60
    with pytest.raises(ValueError):
        labels_from_filename("9-8-8-8-8-8-8-8.jpeg")


def test_fen_groups_publisher_test_and_pixel_duplicates_stay_together():
    records = [dict(placement=str(i), pixel_sha256=str(i), original_split="train") for i in range(100)]
    records += [dict(placement="0", pixel_sha256="test-0", original_split="test"), dict(records[1])]
    splits, duplicates = group_split(records)
    assert duplicates == 1
    assert sum(r["placement"] == "0" for r in splits["test"]) == 2
    positions = [{r["placement"] for r in rows} for rows in splits.values()]
    assert all(not a & b for i, a in enumerate(positions) for b in positions[i+1:])
    assert group_split(records) == (splits, duplicates)
    alias = dict(records[1], original_split="test")
    with_test_alias, _ = group_split(records+[alias])
    assert any(r["placement"] == "1" for r in with_test_alias["test"])
    assert records[1]["original_split"] == "train"


def test_vectorized_tiles_match_existing_crop_preprocessing(tmp_path):
    import cv2
    import numpy as np
    from training.train_positions import PositionBoards, normalize
    from vision.square_recognition import tensor_from_bgr
    from vision.template_recognition import square_image
    rng = np.random.default_rng(42)
    image = rng.integers(0, 256, size=(400,400,3), dtype=np.uint8)
    path = tmp_path / "fixture.png"
    cv2.imwrite(str(path), image)
    tiles, labels = PositionBoards([dict(image=str(path), labels=[12]*64)])[0]
    squares = [f"{f}{r}" for r in range(8,0,-1) for f in "abcdefgh"]
    expected = tensor_from_bgr([square_image(image, s, "white") for s in squares])
    np.testing.assert_allclose(normalize(tiles).numpy(), expected.numpy(), atol=1e-6)


def test_camera_augmentation_is_finite_and_does_not_mutate_source_pixels():
    import torch
    import random
    from training.train_positions import normalize
    torch.manual_seed(42)
    random.seed(42)
    pixels = torch.randint(0,256,(8,3,48,48),dtype=torch.uint8)
    original = pixels.clone()
    output = normalize(pixels, augment=True, camera_augment=True)
    assert output.shape == pixels.shape and torch.isfinite(output).all()
    assert torch.equal(pixels,original)
