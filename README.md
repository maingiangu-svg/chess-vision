# Chess Vision AI

## Redmi Turbo 3 + MacBook M1 LAN demo

The Mac serves a small mobile web interface and WebSocket processor. The Redmi camera sends compressed 640-pixel JPEG frames at most roughly 3 times per second; only one analysis frame is in flight. The phone draws the detected board and projected engine move. This frontend is plain TypeScript-free browser JavaScript so it can be served directly without a Node build step.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[server,dev]'
brew install mkcert
ipconfig getifaddr en0
```

Stockfish 19 macOS universal has been installed locally at `vendor/stockfish/stockfish/stockfish-macos-universal` and is detected automatically. This binary is ignored by Git; on a fresh clone, obtain the macOS universal package from the [official Stockfish download](https://stockfishchess.org/download/) or set `STOCKFISH_PATH`. Use the IP printed by the last command below. On this Mac it was `192.168.1.232` when tested. The Mac and Redmi must share Wi-Fi. Create a certificate for the current LAN IP:

```bash
mkcert -install
mkdir -p certs
mkcert -cert-file certs/lan.pem -key-file certs/lan-key.pem 192.168.1.232
mkcert -CAROOT
.venv/bin/python -m uvicorn application.server:app --host 0.0.0.0 --port 8443 --ssl-certfile certs/lan.pem --ssl-keyfile certs/lan-key.pem
```

`mkcert -install` requires a macOS administrator password to trust the CA on the Mac; it could not complete unattended in this environment. Copy only `rootCA.pem` from the `mkcert -CAROOT` directory to your Redmi, then install it as a CA certificate in Android Settings (Security → Encryption & credentials → Install a certificate → CA certificate; menu names vary by Android version). Never copy `rootCA-key.pem` to the phone or share it. In Chrome on Redmi open `https://192.168.1.232:8443/` using your actual IP. Confirm Chrome shows a trusted connection before granting camera permission. If Android policy does not trust the installed CA for Chrome, use a certificate issued by a trusted public CA for a domain that resolves to the Mac on the local network; do not bypass certificate warnings.

The server binds only to the LAN host you specify; this command creates no public tunnel. If the Mac firewall asks, allow incoming connections for the local network. For a camera-only check, no model is needed; the UI will display `Chưa có model nhận diện`. Use `CHESS_WEIGHTS=/absolute/path/to/12-class.pt` when you have validated local weights. The model must expose all 12 names in `shared/schemas.py`. Tap `Chọn 4 góc` and touch the board corners if automatic detection fails; recalibrate after moving the phone. `Lưu ảnh mẫu` stores real camera images under `samples/` for later annotation and validation. `Thiết lập thế cờ` requires a full FEN when the board is not at the standard starting position.

Verified on Mac: Stockfish 19 UCI analysis, HTTPS GET over LAN with CA validation, WSS exchange of an encoded JPEG and four selected corners, and automated tests. The Redmi camera has been shown streaming in a user-provided screenshot. Piece recognition accuracy and the AR arrow on Redmi remain unverified. Without YOLO weights or a successful local template calibration, no camera-derived move is displayed.

### Local 2D template calibration

Show the complete standard starting board on the Mac. Choose the correct orientation on Redmi (white at bottom, left, top, or right); for a sideways board with white on the left, choose `Trắng bên trái`. Use `Chọn 4 góc` if the outline is inaccurate. Press `Lấy mẫu quân (ván mới)` and confirm only when all 32 pieces are visible and no animation or hand obscures the board. The server saves that real camera frame and confirmed FEN/corners under `samples/`, extracts examples of 12 piece classes and empty squares, and rejects calibration if it cannot read the known starting board back consistently. Subsequent frames must classify all squares and agree with a unique legal move for three stable frames before Stockfish runs. This is an experimental, session-local template recognizer; its confidence number is a matching heuristic, not a measured probability. Theme changes, glare, moiré, zoom, highlights and movement may cause uncertainty.

Initial Redmi image audit (1 November 2026): a saved starting-board frame was used for calibration. On one manually transcribed development frame, 61/64 squares matched, 2 were uncertain, and 1 was misclassified. On a separate manually transcribed frame, 60/64 matched, 2 were uncertain, and 2 were misclassified. Both frames came from the same phone and screen in one session; these are small provisional checks, not a general accuracy benchmark. Queen/knight and piece-color errors remain unresolved. The legal-transition gate prevents these whole-board observations from becoming arbitrary Stockfish positions.

To manually label an image saved with `Lưu ảnh mẫu` for later evaluation or training:

```bash
.venv/bin/python -m scripts.label_sample samples/your-image.jpg --orientation white_left
.venv/bin/python -m scripts.evaluate_templates samples/calibration-xxxx.jpg samples/heldout-1.labels.json samples/heldout-2.labels.json
```

If no companion corner JSON exists, click four corners first. Click each square, then press `0` for empty, `1`–`6` for white pawn/knight/bishop/rook/queen/king, or `qwerty` for the corresponding black pieces. Press `s` to save `.labels.json`. Accuracy should only be calculated on independent labeled camera frames. The phone UI screenshot is not a raw held-out frame and cannot establish piece accuracy.

A modular MVP for physical-board geometry, chess-state validation, Stockfish analysis, and an optional Ultralytics YOLO piece detector. It is intended for practice/research, not prohibited competitive assistance.

## Current status

Implemented: typed detection/HUD interfaces, automatic quadrilateral detection, manual four-click corner selection, perspective rectification, orientation mapping, legal move tracking with stable-frame confirmation, Stockfish UCI wrapper, optional YOLO inference, and camera overlay. YOLO inference requires real chess-piece weights; no model accuracy is claimed.

Not verified: piece accuracy on raw camera frames, GPU/CUDA, and end-to-end recognition on Redmi. The local Stockfish binary and Mac-side HTTPS/WebSocket were verified separately.

## MacBook M1 setup

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -e '.[dev,vision]'
pytest -q
brew install stockfish
python -m application.main --camera 0 --weights /absolute/path/chess-pieces.pt --stockfish /opt/homebrew/bin/stockfish --debug
```

Set `STOCKFISH_PATH` or use `--stockfish`. Start at the normal initial position, or pass `--fen` for a known setup. Click four board corners when automatic detection fails; press `r` to return to automatic detection, `Esc` to quit. Set `--orientation black` when black's home rank is nearest the bottom of the camera image.

## RTX setup and model training

Create a CUDA-enabled PyTorch environment using the command from the official PyTorch selector for the installed CUDA driver, then:

```bash
python -m pip install -e '.[dev,vision]'
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

The repository does not invent a dataset or weights. A candidate public [chess-piece dataset/model](https://universe.roboflow.com/ashanes-workspace/chess-full-npama) lists the twelve colored pieces plus a generic `bishop` class under CC BY 4.0; its model uses hosted inference, so it is not yet validated as offline weights for this app. Export a YOLO-format dataset with the 12 labels in `shared/schemas.py`, create a dataset YAML, and train with Ultralytics, for example `yolo detect train data=/absolute/path/data.yaml model=yolo11n.pt imgsz=640 batch=-1`. Evaluate with `yolo detect val ...`, then point `YoloPieceDetector` at the resulting `.pt` file. Validate on images from your camera before relying on live recognition.

## Architecture

`vision` emits detections and board geometry; `chess_core` owns legal game state and Stockfish; `application` owns camera/rendering; `shared` contains stable dataclasses. The current JSON-compatible detection fields are `label`, `confidence`, `bbox`, `square`, and `metadata`; HUD output is `uci`, `from_square`, `to_square`, `score_cp`, and `pv`.

## Next integration steps

The app estimates each piece base from its bounding box, waits for stable observations, then accepts a position only if a unique legal move produces it. Preserve castling/en-passant/counters from the tracked game rather than reconstructing them from pixels. The inverse homography in `application.visualizer.draw_move` projects a chosen move back onto the camera frame. This simple temporal filter needs real-camera tuning for hands, occlusion and pieces whose bases are not near the lower part of their detection box. A future AR display can consume `HudMove` and board geometry without embedding Stockfish in the display.
