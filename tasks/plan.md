# Vision training

Preserve Mac application behavior and shared labels. Work on feature/vision-training.
Use existing geometry and 48px square crop for both training and inference.

1. Probe hardware with an actual CUDA AMP forward/backward, save JSON.
2. Import explicit annotations; require provenance, reject incomplete labels,
   mutated images and duplicates. Freeze session-disjoint train/val/test splits.
3. Train pretrained YOLO11n from true manual boxes and MobileNetV3-small from
   rectified squares. Check VRAM, log config/seed/packages/data fingerprint,
   save checkpoints and stop early on validation data only.
4. Evaluate all 64 squares including abstentions; report exact boards, confusion,
   position plausibility, tags and timing. Select on validation, test once.
5. Export and compare formats. Keep M1 measurements separate from Windows.
6. Run regression tests, commit increments, prepare PR without merging main.

Data risk: samples and Mac engine are ignored by Git and absent here. Public
screenshots/synthetic renders cannot establish Redmi real-camera accuracy.
No fake labels or inferred tight piece boxes. No accuracy claim before independent
real-camera evaluation. Targets remain >=99% squares and >=95% exact boards.

User direction update: prioritize existing public 2D datasets, no manual capture
or annotation requirement for baseline. Download original Kaggle Chess Positions
via KaggleHub after license check. Decode filename FEN, deduplicate whole images,
reserve publisher test by FEN group and split validation from remaining positions.
Train a small CNN on all available training boards with memory-bounded loaders.
Validate/select before opening test; export PT/ONNX and measure both. Keep camera
claims and Mac-specific measurements explicitly unverified when unavailable.
