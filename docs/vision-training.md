# Huấn luyện RTX và triển khai MacBook

Mô hình chưa được đánh giá trên camera thật vì chưa có ảnh Redmi. Baseline ưu
tiên dataset 2D Internet, không yêu cầu gán nhãn thủ công. Xem báo cáo thí nghiệm
trong `reports/` để phân biệt kết quả synthetic 2D và camera thật.

## Dataset 2D có sẵn — hướng ưu tiên

```powershell
.\.venv\Scripts\python.exe -m pip install -e '.[training,datasets,export]'
.\.venv\Scripts\python.exe -m scripts.download_positions
.\.venv\Scripts\python.exe -m training.positions data/chess-positions --output data/positions-frozen
.\.venv\Scripts\python.exe -m training.train_positions data/positions-frozen/splits.json --output runs/positions-001 --epochs 1 --batch-boards 32 --workers 4
.\.venv\Scripts\python.exe -m training.evaluate_positions runs/positions-001/best.pt data/positions-frozen/splits.json --split val --output runs/positions-001/val.json
.\.venv\Scripts\python.exe -m training.stress_positions runs/positions-001/best.pt data/positions-frozen/splits.json --output runs/positions-001/stress.json
.\.venv\Scripts\python.exe -m training.train_positions data/positions-frozen/splits.json --output runs/positions-002 --initial runs/positions-001/best.pt --camera-augment --epochs 2 --lr 0.0005 --batch-boards 32 --workers 4
.\.venv\Scripts\python.exe -m training.stress_positions runs/positions-002/last.pt data/positions-frozen/splits.json --output runs/positions-002/stress.json
```

Nguồn gốc Kaggle CC0 dùng KaggleHub chính thức; FEN filename cho nhãn 64 ô,
không cần người dùng gán từng ảnh. `training.positions` kiểm tra kích thước,
ký hiệu FEN, hash file/pixel, loại bản sao và giữ nhóm FEN trong cùng split.
Test gốc được giữ riêng; validation là 12,5% nhóm còn lại. Một số thế tổng hợp
không hợp luật; vẫn giữ nhãn ảnh để không làm đẹp benchmark. Không đưa các
thế đó vào engine.

CNN baseline gồm ba Conv/BatchNorm/ReLU/Pool và đầu phân loại 13 lớp. Training
nạp theo ảnh (64 ô mỗi ảnh), không cache 6,4 triệu crop vào RAM/VRAM. Mỗi batch
32 ảnh tương đương 2.048 ô. Mixed precision, class weights căn bậc hai nghịch
tần suất, AdamW, cosine learning rate, early stopping theo validation, lưu
checkpoint tốt nhất theo exact-board có ngưỡng confidence. Baseline thực tế dừng
sau một epoch khi đạt tiêu chí validation đã ghi trước. Fine-tune thêm hai epoch
với `--camera-augment`: xoay theo góc 90°, blur sigma 0,2–1,6 và tint nhẹ, tất cả
là phép biến đổi synthetic. Không chứng minh chống moiré/phản chiếu camera.

Chọn checkpoint dùng validation sạch và cùng 512 bàn validation biến đổi, không
chọn bằng test. Xem `reports/finetune-protocol.json` và `model-selection.json`.
`best.pt` của trainer chọn theo validation sạch; checkpoint cuối được chọn sau
so sánh độ bền có thể là `last.pt`. Giữ cả hai, không đổi tên để che lịch sử.

Export checkpoint bằng `training.export squares ... --format onnx`. Sau khi
khóa model/ngưỡng, `training.evaluate_positions --split test` đo test độc lập.
Đánh giá cả ONNX bằng cùng lệnh, `--device cpu`; luôn ghi tên report mới.
`raw_argmax` và kết quả có `uncertain` được báo riêng. Chỉ số invalid-position
cần đọc cùng tỷ lệ nhãn thật bất hợp lệ trong dữ liệu tổng hợp.

## Môi trường Windows

Tạo venv, cài CUDA PyTorch trước các optional dependencies. Dùng
[hướng dẫn chính thức](https://pytorch.org/get-started/locally/) và kiểm tra
build hỗ trợ kiến trúc GPU. Driver báo CUDA tối đa không có nghĩa CUDA runtime
đã cài. Máy này là RTX 5060 Laptop 8151 MiB, compute capability 12.0.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -e '.[training,dev,server,export]' httpx
.\.venv\Scripts\python.exe -m scripts.inspect_gpu --output reports/gpu.json
```

Không chạy pip sửa một venv đồng thời với pytest/inference đang nạp thư viện
nhị phân trên Windows. `gpu.json` ghi phiên bản thực và phép Conv2d AMP
forward/backward, không chỉ `cuda.is_available()`. Đo VRAM của phép kiểm tra này
không phải VRAM YOLO. Training square có bước forward/backward thực và giảm
batch khi CUDA OOM. YOLO dùng batch 4, 640px và AMP; kiểm tra lượt đầu, giảm batch
trong config nếu OOM. Không cam kết mọi cấu hình đều vừa 8 GB.

## Nhập dữ liệu

Ảnh cùng phiên và annotation `.labels.json` nằm cạnh nhau. Trường `image` có
thể là tên file tương đối; importer tìm basename cạnh JSON khi chuyển từ Mac.
Mỗi JSON cần `image`, `corners`, `orientation`, `labels` gồm đủ 64 ô.
Xem [kiểm toán dữ liệu](../reports/dataset-audit.md) về nhãn, hộp và giấy phép.

```powershell
.\.venv\Scripts\python.exe -m training.data import samples/session-a data/manifest.jsonl --session redmi-a --source own-redmi-monitor --license own --domain real_camera
.\.venv\Scripts\python.exe -m training.data import samples/session-b data/manifest.jsonl --session redmi-b --source own-redmi-monitor --license own --domain real_camera
.\.venv\Scripts\python.exe -m training.data import samples/session-c data/manifest.jsonl --session redmi-c --source own-redmi-monitor --license own --domain real_camera
.\.venv\Scripts\python.exe -m training.data split data/manifest.jsonl data/frozen --seed 42
```

ID phiên phải toàn cục và đủ rộng để chứa ảnh tương quan. Import từ chối nhập
trùng annotation; split từ chối ảnh trùng. Train/evaluate kiểm tra lại hash ảnh,
annotation và giao nhau giữa các phiên. Sau khi sửa nhãn phải nhập/khóa lại.
Lưu bản manifest và splits bất biến cùng artifact thí nghiệm, không đưa ảnh riêng
tư lên Git. Training yêu cầu train split chứa đủ 13 lớp.

## Huấn luyện và đánh giá

```powershell
.\.venv\Scripts\python.exe -m training.train squares data/frozen/splits.json --output runs/squares-001
.\.venv\Scripts\python.exe -m training.train yolo data/frozen/splits.json --output runs/yolo-001
.\.venv\Scripts\python.exe -m training.evaluate squares runs/squares-001/best.pt data/frozen/splits.json --split val --output runs/squares-001/val.json
.\.venv\Scripts\python.exe -m training.evaluate yolo runs/yolo-001/detector/weights/best.pt data/frozen/splits.json --split val --boxes --output runs/yolo-001/val.json
```

Config nằm trong `configs/training.json`: seed, learning rate, epochs, patience,
batch, resolution, device và ngưỡng confidence/margin. MobileNetV3-small dùng
ImageNet pretrained, 13 đầu ra theo `PIECE_CLASSES` rồi `empty`; giữ crop 48px
và RGB/normalization thống nhất. YOLO11n dùng pretrained COCO; mapping YAML
12 lớp đúng thứ tự shared. Tải pretrained cần mạng ở lần đầu.

Augmentation square: dịch/rotate nhỏ, sáng/tương phản/màu, blur nhẹ, JPEG.
YOLO: phối cảnh/dịch/scale nhẹ, không mosaic/mixup/flips. Chưa mô phỏng moiré,
phản chiếu hay highlight một cách được kiểm chứng; cần ảnh thật của các lỗi đó.
Không coi augmentation là bằng chứng tổng quát hóa.

Lưu `config.json`, `provenance.json`, splits, `epochs.jsonl`, VRAM preflight,
`best.pt`/`last.pt`. Square checkpoint chọn theo validation loss, early stopping;
so sánh kiến trúc/ngưỡng cuối theo exact-board trên validation. Phục hồi optimizer
và scaler có trong checkpoint nhưng CLI chưa hỗ trợ tự resume thí nghiệm dở.
YOLO dùng logging/checkpoint/early stopping sẵn của Ultralytics. Seed giúp lặp
lại; không cam kết bitwise giữa Windows/CUDA và M1.

Sau khi khóa mô hình và ngưỡng, thay `--split val` bằng `--split test`, xuất tên
báo cáo mới. Không ghi đè báo cáo cũ. Không có hộp thì bỏ `--boxes`: vẫn đo được
64 ô/toàn bàn, `detection` là null. Chỉ so sánh mAP khi có hộp thật.

## Export và M1

```powershell
.\.venv\Scripts\python.exe -m training.export squares runs/squares-001/best.pt --format onnx
.\.venv\Scripts\python.exe -m training.export yolo runs/yolo-001/detector/weights/best.pt --format onnx
```

ONNX square đi kèm `.onnx.json` chứa class mapping và preprocessing. Export
kiểm tra sai số số học bằng tensor ngẫu nhiên; đây không phải đo độ chính xác.
Copy checkpoint hoặc ONNX cùng sidecar sang Mac. Dataset frozen chứa đường dẫn
Windows nên phải nhập/khóa lại từ cùng bản ảnh/annotation trên Mac và xác nhận
cùng SHA256 và cùng nhóm, hoặc cập nhật đường dẫn bản sao mà giữ nguyên nhóm.
Không chia lại ngẫu nhiên để chọn tập test khác.

Trên Mac tạo venv, cài PyTorch macOS theo nguồn chính thức. Với CNN đã xuất,
`python -m pip install torch onnxruntime` và `python -m pip install -e '.[server]'`
đủ cho inference; không cần bộ huấn luyện YOLO. Muốn chạy scripts đánh giá/export
thì cài thêm các extra tương ứng. Dùng config bản sao `device=cpu`, sau đó `mps`
nếu khả dụng. Chạy
evaluation trên cùng test cho `.pt` và `.onnx`, so sánh chính xác và median/p95.
ONNX adapter hiện dùng CPUExecutionProvider; không gán nhãn đó là Apple Neural
Engine. Latency báo từ bàn đã rectify tới 64 nhãn; chưa gồm camera/network,
geometry, Stockfish. Ba lượt warmup trước đo. Chưa có benchmark M1 tại Windows.

Core ML thực hiện trên Mac có `coremltools` tương thích với phiên bản PyTorch:

```bash
python -m training.export squares runs/squares-001/best.pt --format coreml
python -m training.export yolo runs/yolo-001/detector/weights/best.pt --format coreml
```

Cài `python -m pip install coremltools` trên Mac trước bước này và kiểm tra
ma trận phiên bản tương thích. Core ML runtime không được xác minh trên Windows.

Core ML square batch cố định 64, đầu vào NCHW float32 đã chuẩn hóa; copy sidecar
`.mlpackage.json`. Chạy lại evaluation trên `.mlpackage` để đo camera accuracy
và latency. Không suy độ chính xác chỉ từ việc export thành công. Core ML và MPS
chưa được xác minh trên Mac trong lượt này.

## Adapter và cổng an toàn

YOLO `.pt` đã có đường nạp bằng `CHESS_WEIGHTS`; chỉ đặt sau khi xác minh model.
SquareRecognizer giữ hợp đồng `(occupied, min_confidence, errors)` của
TemplateRecognizer. Server chỉ được bổ sung lựa chọn model qua
`CHESS_SQUARE_WEIGHTS` và nhánh phân loại adapter trong `analyze_frame`;
không sửa giao thức, frontend, Stockfish hoặc logic trạng thái cờ.

```python
from vision.square_recognition import SquareRecognizer

recognizer = SquareRecognizer("/absolute/path/best.pt", device="cpu", confidence=.9, margin=.25)
occupied, confidence, errors = recognizer.predict(rectified_board, "white")
# Nếu errors: reset observation / chờ frame tiếp; không tạo FEN đưa vào Stockfish.
```

Trên Mac:

```bash
export CHESS_SQUARE_WEIGHTS=/absolute/path/best.pt
export CHESS_DEVICE=cpu
python -m uvicorn application.server:app --host 0.0.0.0 --port 8443 --ssl-certfile certs/lan.pem --ssl-keyfile certs/lan-key.pem
```

Chỉ đặt một trong `CHESS_SQUARE_WEIGHTS` và `CHESS_WEIGHTS`. ONNX dùng thêm
sidecar `.onnx.json` và onnxruntime. RTX chỉ phục vụ huấn luyện offline.

Fallback tùy chọn: `fallback=session_template_recognizer` đã được người dùng
calibrate. Khi neural không chắc, trả kết quả template đầy đủ và vẫn đi qua
MoveTracker. Mặc định không fallback để không che lỗi trong đánh giá. Không
điền quân dựa trên vị trí xuất phát, không chữa nhãn để ép thành thế cờ hợp lệ.
Softmax là score chưa được calibrate xác suất; phải chọn ngưỡng bằng validation.

Thư mục `models/`, `data/`, `runs/` và định dạng trọng số đã ignore. Phân phối
weights qua release artifact hoặc kho có version/checksum; không commit binary
vào Git thường. M1 chạy model cục bộ, không cần RTX online.
