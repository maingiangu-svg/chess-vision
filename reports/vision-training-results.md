# Kết quả huấn luyện Chess Vision AI — 01/11/2026

Đã huấn luyện CNN 13 lớp thực tế bằng RTX 5060 Laptop, xuất PyTorch/ONNX và
kiểm tra adapter với server hiện tại. Model cuối đạt **19.994/20.000 bàn đúng
hoàn toàn (99,97%)** trên test ảnh 2D tổng hợp sạch. Đây **không phải độ chính xác
camera Redmi**; chưa kiểm thử camera thật hoặc benchmark MacBook M1.

## Dữ liệu và chống rò rỉ

- Nguồn: [Chess Positions, koryakinp](https://www.kaggle.com/datasets/koryakinp/chess-positions),
  v1, giấy phép CC0 theo API nhà phát hành, tải bằng KaggleHub 1.0.3.
- Gói có 200.000 tệp JPEG nhưng 100.000 bản sao nguyên pixel. Sau dedup còn
  **100.000 ảnh, 100.000 FEN**, mỗi ảnh 400×400. Nhãn tự động đọc từ FEN filename,
  không cần chụp/gán nhãn thủ công.
- Train **70.034 bàn / 4.482.176 ô**; validation **9.966 / 637.824**;
  test gốc **20.000 / 1.280.000**. Toàn bộ nhóm FEN cùng split; test gốc giữ
  riêng, validation lấy bằng hash có seed từ phần còn lại. Không chia theo ô.
- Kiểm tra mọi ảnh/FEN, hash ảnh và byte file, từ chối pixel giống mà FEN khác.
  Không phát hiện xung đột. Đầy đủ 13 lớp trong mỗi split. Khoảng 84% là ô trống.
- Asset quân/board có thể lặp giữa các split. Đây là nhận diện vị trí mới trong
  miền renderer cùng nguồn, không phải kiểm thử theme mới hay ảnh điện thoại.
- Mirror Hugging Face tại revision đã khóa chỉ có 10.000 JPEG và không có test/;
  không coi mô tả 100.000 ảnh trên card là số file thực. Xem [dataset audit](dataset-audit.md).

## Môi trường và thí nghiệm

Windows 11 build 26200, RTX 5060 Laptop 8151 MiB, driver 616.87, capability 12.0.
PyTorch **2.11.0+cu128**, torchvision 0.26.0+cu128, CUDA runtime 12.8; đã chạy
forward/backward AMP thật. Driver báo CUDA UMD tối đa 13.4, khác runtime PyTorch.
Chi tiết trong [gpu.json](gpu.json), môi trường pip trong
[requirements-windows.txt](requirements-windows.txt), chỉ dùng lock này cho Windows.

CNN **335.789 tham số**: 3 Conv/BatchNorm/ReLU/Pool, đầu dense 128 và 13 logits.
Input RGB 48×48, chuẩn hóa ImageNet; mapping là `PIECE_CLASSES` rồi `empty`.
Batch 32 bàn = 2.048 ô; AMP FP16, AdamW, seed 42, trọng số lớp theo nghịch căn
tần suất. Peak allocator PyTorch cao nhất quan sát trong các epoch: **1363,56 MiB**.

1. Baseline: một epoch hoàn tất, learning rate 0,001. Train 282,42 giây, validation
   26,71 giây, chưa gồm startup. Dừng sau checkpoint khi đạt quy tắc validation
   đã ghi trước; phần epoch tiếp theo chưa hoàn tất bị bỏ, không dùng để báo kết quả.
2. Fine-tune: thêm **hai epoch** từ baseline, lr 0,0005 với cosine schedule.
   Blur Gaussian sigma 0,2–1,6, xoay 90°, sáng/tương phản và tint nhẹ, đều synthetic.
   Train 292,14 và 262,57 giây; validation 25,61 và 11,79 giây.
3. Chọn checkpoint cuối bằng validation sạch và cùng 512 bàn validation biến
   đổi. Mean exact-board qua 8 điều kiện tăng **62,52% → 95,61%**. Không dùng test
   để chọn checkpoint/ngưỡng. Protocol và checksum ở [model-selection.json](model-selection.json).

Baseline test được báo riêng; model cuối được chọn sau chẩn đoán validation.
Test đã được đánh giá cho baseline và model cuối, không phải một benchmark mù
chưa từng mở. Không chỉnh ngưỡng bằng test. Confidence giữ **0,90**, margin **0,25**.

Đã đo thử channels-last: chậm hơn trên cấu hình này, nên giữ NCHW. Đây là phép
so sánh kernel huấn luyện, không phải latency inference.

## Test độc lập 2D sạch

| Model / runtime | Ô đúng, tính uncertain là lỗi | Bàn đúng cả 64 ô | Uncertain | Median / p95 |
| --- | --- | --- | --- | --- |
| Baseline PyTorch / RTX | 1.280.000/1.280.000 | 20.000/20.000 | 0 | 5,10 / 8,27 ms |
| Model cuối PyTorch / RTX | **1.279.993/1.280.000 (99,999453%)** | **19.994/20.000 (99,97%)** | 7 ô, 6 bàn | **5,05 / 8,50 ms** |
| Model cuối ONNX / Windows CPU | **1.279.993/1.280.000 (99,999453%)** | **19.994/20.000 (99,97%)** | 7 ô, 6 bàn | **15,93 / 19,61 ms** |

Latency đo một bàn đã rectify → 64 nhãn, gồm crop/normalize/transfer; 3 lượt
warmup và 128 bàn đo. Không gồm JPEG decode, geometry, mạng, Stockfish. Không
thay số đo Windows này bằng nhãn M1. Chi tiết confusion, precision/recall từng
lớp và từng lỗi ở [test-final-pytorch.json](test-final-pytorch.json) và
[test-final-onnx.json](test-final-onnx.json).

Không có nhãn sai nào được chấp nhận trên test sạch sau ngưỡng: cả 7 lỗi là
abstention. Chúng gồm 5 black queen, 1 black pawn, 1 empty. Nếu ép argmax, có
1 black queen bị nhầm black king; ngưỡng đã chặn ca đó. Argmax không phải kết
quả được dùng để tính 99,97% ở trên. Confusion matrix giữ cả cột uncertain.

Tỷ lệ vị trí không hợp lệ theo kiểm tra plausibility: **15,55%**; bản thân nhãn
thật tổng hợp đã có **15,53%** không hợp lệ. Không lọc các bàn đó để tăng accuracy.
Đây không phải tần suất app gửi sai thế cờ vào engine: app vẫn giữ cổng nước đi
hợp lệ và ba frame ổn định, không dựng FEN lịch sử từ ảnh.

## Chẩn đoán độ bền trên 512 bàn validation

| Biến đổi được xác định trong code | Baseline exact-board | Model cuối exact-board |
| --- | --- | --- |
| Sạch | 100% | 100% |
| Gaussian 7×7, sigma 1,5 | 12,70% | **91,02%** |
| JPEG quality 35 | 100% | 100% |
| Xoay 90°, hướng ô cập nhật đúng | 0% | **99,80%** |
| Xoay 180° | 0,59% | **100%** |
| Tint highlight trên một ô có quân | 97,66% | 99,61% |
| Che vùng 20×20 trong một ô có quân | **89,26%** | **74,41%** |
| Warp phối cảnh rồi rectify bằng góc chính xác | 100% | 100% |

Che khuất bị giảm sau fine-tune; không che giấu tradeoff này. Đây là chẩn đoán
tổng hợp, góc rectify đã biết chính xác. Không mô phỏng được phản chiếu/màn hình,
moiré hoặc sự sai lệch calibration thật. Các lỗi màu knight, queen/knight,
bishop/pawn trên test sạch không được coi là đã giải quyết trên camera.
Softmax chưa được calibrate thành xác suất. Không triển khai như recognizer
camera đã được chứng nhận; dùng errors/uncertain và fallback đã hiệu chuẩn.

## Export, tích hợp và kiểm thử

PyTorch **1.351.675 byte**, ONNX **1.343.991 byte**, cùng sidecar class mapping.
Sai số logits lớn nhất trên tensor kiểm tra export: **1,0491×10⁻⁵**; cùng kết quả
test đầy đủ ở hai runtime. PT/ONNX chạy offline; RTX không cần online khi app
chạy trên Mac. Core ML có script nhưng chưa xuất/kiểm chứng trên Mac.

Server chỉ thêm `CHESS_SQUARE_WEIGHTS` / `CHESS_DEVICE` và nhánh adapter; frontend,
Stockfish, schema và logic chess-state được giữ. Không đặt đồng thời hai loại
weights. Xem [hướng dẫn Mac và pipeline](../docs/vision-training.md).

**34 passed, 3 skipped** trong bộ kiểm thử cuối. Ba skip: thiếu ảnh Redmi/engine,
thiếu Stockfish cho calibration-to-engine và thiếu Stockfish cho UCI integration.
Trainer YOLO thực chạy một epoch trên fixture rõ ràng là synthetic để kiểm thử
API; không dùng các số đó làm chất lượng detector. Không có benchmark YOLO công
bằng vì các nguồn hộp được yêu cầu còn mapping/lớp dư hoặc giấy phép chưa xác
minh. Precision/recall ở báo cáo classifier không được gọi là mAP detection.

Trọng số và dataset không vào Git thường. Model được đóng gói riêng với SHA256;
giữ kết quả thí nghiệm, config, seed và protocol trong nhánh feature/vision-training.
Mục tiêu camera >=99% ô / >=95% toàn bàn **chưa được xác nhận**.

Báo cáo công khai dùng đường dẫn tương đối và không chứa hostname máy. Bản log
gốc được giữ cục bộ trong thư mục `runs/` bị ignore; số đo không bị thay đổi.
