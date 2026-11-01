# Kiểm toán dữ liệu — 01/11/2026

## Dữ liệu cục bộ

Repo không có `samples/`, ảnh camera hoặc `.labels.json`. Người dùng xác nhận
chưa có dữ liệu Redmi trên Windows và yêu cầu ưu tiên dataset Internet, không
gán nhãn thủ công. Đã tải Chess Positions v1 bằng KaggleHub chính thức, API nguồn
công bố CC0. Kết quả template trước đây trên Mac không phải kết quả mô hình mới.

## Bốn nguồn theo yêu cầu

| Nguồn | Kiểm tra thực tế | Quyết định |
| --- | --- | --- |
| [smallchess/kaggle-chess-positions](https://huggingface.co/datasets/smallchess/kaggle-chess-positions) | Dataset card CC0, nói 100.000 ảnh. SDK `HfApi` tại revision `35fed168fce84b775473c72f643543c0c64d695a` liệt kê 10.000 JPEG và không có test/. Hai ảnh tải bằng `hf_hub_download`, kiểm tra trực quan khớp FEN trong tên file. | Dùng bản gốc Kaggle để có đủ bộ dữ liệu. Không báo mirror này có đủ 100.000 ảnh. |
| [Kaggle gốc](https://www.kaggle.com/datasets/koryakinp/chess-positions) | API nhà phát hành: CC0: Public Domain, v1, 4.305.437.124 byte. Gói tải chứa JPEG ở cả root train/test và dataset/train/test. | Nguồn chính cho CNN. Audit toàn bộ pixel/FEN và loại bản sao trước khi chia. Xem `positions-audit.json` về số lượng sau audit. |
| [Roboflow chess.com-pieces](https://universe.roboflow.com/chess-pieces-8qwqx/chess.com-pieces) | Trang ghi CC BY 4.0, 183 ảnh, 25 lớp: các số 0–12 và 12 mã `bb/bk/bn/bp/bq/br/wb/wk/wn/wp/wq/wr`. Không có giải thích mapping các số. | Không đoán mapping. Không dùng trực tiếp làm detector 12 lớp. Chưa tải export hoặc kiểm tra hộp từng ảnh. |
| [Roboflow 2d-chessboard-and-chess-pieces v4](https://universe.roboflow.com/chess-project/2d-chessboard-and-chess-pieces/dataset/4) | URL v4 bị robots chặn qua web tool. Trang project công khai ghi CC BY 4.0, 5.000 ảnh, 19 lớp gồm board, ký hiệu quân phân biệt hoa/thường và chữ dư a/c/e/h/o/s. Trang hiện chỉ hiển thị 1 version. | Không khẳng định v4 tải được hoặc mapping sạch. Cần export chính thức và loại/xác minh lớp dư trước YOLO; chưa dùng trong thí nghiệm. |
| [Zai-Kun/2d-chess-pieces-detection](https://github.com/Zai-Kun/2d-chess-pieces-detection) | README nói YOLO11n, dữ liệu tự sinh từ nhiều bộ quân, weights pt/onnx trong releases. GitHub contents API không thấy LICENSE; README không nêu giấy phép hay benchmark độc lập. | Chỉ tham khảo kiến trúc/mapping; chưa sao chép mã hoặc weights không rõ quyền phân phối. |

Tệp `huggingface-source.json`, `kaggle-source.json` và `download-positions.json`
ghi metadata đo thực tế. Giấy phép CC0 là tuyên bố của nhà phát hành dataset;
không suy ra quyền độc quyền với từng tài sản giao diện bên thứ ba.

Audit hoàn tất: 200.000 tệp tải về chứa đúng 100.000 bản sao pixel; còn 100.000
ảnh và 100.000 FEN khác nhau. Train 70.034 bàn (4.482.176 ô), validation 9.966
bàn (637.824 ô), test 20.000 bàn (1.280.000 ô). Tất cả 13 lớp có mặt trong mỗi
split. Ô trống chiếm khoảng 84%; do đó phải đọc per-class và full-board, không
chỉ accuracy tổng. Không có ảnh lỗi kích thước/FEN hoặc xung đột nhãn pixel.

## Nhãn và chống rò rỉ Chess Positions

Tên JPEG là FEN phần bố trí quân, thay `/` bằng `-`. Parser dùng python-chess
kiểm tra độ rộng hàng/ký hiệu, phân biệt chữ hoa quân trắng và chữ thường quân
đen; sinh 64 nhãn theo pixel từ a8 tới h1. Vị trí tổng hợp có thể bất hợp lệ về
luật cờ; không loại chúng để làm đẹp kết quả và không đưa vào Stockfish.

Không chia theo ô. Giữ toàn bộ nhóm FEN có ảnh thuộc test gốc trong test;
12,5% nhóm FEN còn lại vào validation bằng hash có seed, phần còn lại train.
Hash pixel phát hiện bản sao nguyên ảnh; pixel giống nhưng FEN khác là lỗi.
Mọi kiểu hiển thị của cùng FEN ở cùng split. Báo cáo này chỉ kiểm tra khả năng
nhận diện thế cờ mới trên asset cùng nguồn; không chứng minh theme mới hoặc
camera Redmi. Không yêu cầu người dùng chụp/gán nhãn cho baseline này.

## Nguồn phụ đã xem trước khi nhận danh sách mới

| Nguồn | Phù hợp và giấy phép | Quyết định |
| --- | --- | --- |
| [Roboflow chess-full-npama](https://universe.roboflow.com/ashanes-workspace/chess-full-npama) | Trang ghi CC BY 4.0; ảnh quân vật lý chụp từ góc cố định; 13 lớp gồm `bishop` không có màu. Trang có số lượng ảnh không nhất quán giữa bộ đếm và mô tả. | Không dùng trực tiếp. Cần bản export, kiểm tra từng nhãn/hộp và quyền của tài sản nguồn; không tự gán màu cho `bishop`. Khác miền ảnh màn hình. |
| [hti/2dchess](https://universe.roboflow.com/hti/2dchess) | Kết quả tìm kiếm từ trang nguồn ghi CC BY 4.0. Không tải được trang chi tiết/bản export trong lần kiểm tra này. | Chỉ là ứng viên; chưa xác minh 12 lớp, hộp, nhóm nguồn hay chất lượng. Không sử dụng để huấn luyện/đánh giá. |
| [ChessReD, nguồn tác giả](https://github.com/tmasouris/end-to-end-chess-recognition) | Tác giả mô tả 10.800 ảnh điện thoại của 100 ván, có vị trí quân và góc bàn. Ảnh bàn vật lý. MIT trong repo là giấy phép mã; chưa xác minh giấy phép dữ liệu tại kho tải. | Không suy ra giấy phép dữ liệu từ mã. Khác miền; cần kiểm tra giấy phép bản dữ liệu và tách theo ván/nguồn trước khi nhập. |
| [2d-chess-ocr, nguồn tác giả](https://github.com/AndrewSpano/2d-chess-ocr) | Mã MIT; cung cấp nhãn YOLO tự động và nguồn gốc, nhưng không cung cấp ảnh do bản quyền. | Có giá trị tham khảo cho 2D; không phải bộ ảnh có nhãn sẵn để tải về huấn luyện. Nhãn tự động cần kiểm tra thủ công. |

Không có ứng viên nào được coi là dữ liệu Redmi độc lập. Không tái sử dụng
benchmark của tác giả khác làm số đo cho hệ thống này. Ultralytics có
[điều kiện AGPL/Enterprise](https://www.ultralytics.com/license); cần giữ đúng
điều kiện của phiên bản/model khi phân phối. Không sao chép tài sản Chess.com.

## Quy trình thu thập và khóa bộ đánh giá

- Chụp JPEG thô bằng Redmi, lưu góc TL/TR/BR/BL và một trong `white`, `black`,
  `white_left`, `white_right`. Dùng công cụ gán nhãn sẵn có: `python -m
  scripts.label_sample path/to/frame.jpg --orientation white`.
- Gán rõ cả 64 ô, kể cả `empty`; không tự suy từ vị trí xuất phát. Kiểm tra chéo
  các quân trắng/đen, queen/knight, bishop/pawn. FEN đầy đủ là tùy chọn nếu đã biết
  lịch sử; nhãn quân phải khớp FEN. Không đoán quyền nhập thành từ ảnh.
- Dùng ID phiên chụp duy nhất toàn bộ corpus. Các ảnh liền nhau/cùng nguồn xuất
  phải ở cùng ID, kể cả đổi tên file. Với nguồn công khai không biết phiên, nhóm
  cả nguồn vào một ID để tránh chia ngẫu nhiên ảnh tương quan.
- Thu nhiều phiên thực sự khác nhau về ánh sáng, góc, mức zoom, màn hình, theme
  và thế cờ. Cần ít nhất 3 nhóm để chạy; 3 nhóm không đủ để khẳng định tổng quát.
  Bộ test cần đa dạng và đủ lớn; báo cả số bàn và số phiên, không chỉ phần trăm.
- Gắn `tags`: `highlight`, `glare`, `moire`, `blur`, `perspective`, `occlusion`.
  Hướng bàn tự thêm vào báo cáo. Trường hợp không có mẫu phải ghi chưa đo.
- Pipeline chia 70/15/15 theo số phiên (làm tròn, ít nhất 1 phiên val/test), không
  theo số ô; từ chối ảnh pixel trùng. Hash cả ảnh đã giải mã và annotation.
  Hash chỉ phát hiện trùng chính xác; người nhập phải nhóm ảnh gần trùng và
  thế cờ lặp lại có tương quan. Không coi hash là giải pháp chống mọi rò rỉ.
- Chọn kiến trúc, epoch và ngưỡng bằng validation. Sau khi khóa lựa chọn mới
  chạy test; giữ mọi báo cáo, không chọn lần test đẹp nhất. Đánh giá hai mô hình
  trên cùng các bàn và metadata góc, không trên hai tập thuận lợi khác nhau.
- YOLO yêu cầu `boxes`: danh sách `{ "label": "white_pawn", "bbox": [x1,y1,x2,y2] }`
  theo pixel ảnh gốc, một hộp thủ công cho mỗi quân. Gán nhãn bằng công cụ hộp
  ngoài và thêm vào JSON; pipeline không tạo nhãn hộp từ ô. Số lượng mỗi lớp
  phải khớp nhãn 64 ô. Cần xem overlay hộp và sửa các hộp sai trước huấn luyện.

## Chỉ số cố định

Độ chính xác ô tính cả 64 ô, `uncertain` là lỗi. Full-board exact-match đòi hỏi
đúng cả 64 ô. Ghi thêm tỷ lệ bàn có lỗi ánh xạ và exact-match không collision.
Confusion matrix có 13 hàng nhãn thật, 14 cột dự đoán (gồm uncertain). Mỗi lớp
có precision/recall/support; không có mẫu thì chỉ số là null. YOLO mAP dùng hộp
thủ công qua `--boxes`, không suy mAP từ ô.

`invalid_position_frequency` là kiểm tra `python-chess.is_valid` với FEN nếu
cung cấp; nếu không có lịch sử, kiểm tra khả dĩ ở một trong hai lượt, không có
quyền nhập thành. Đây không phải bằng chứng vị trí đi tới được bằng nước hợp lệ.
Khi chạy app, MoveTracker vẫn là cổng xác thực nước đi và ba frame ổn định.

Mục tiêu không đổi: >=99% ô và >=95% bàn chính xác trên test camera độc lập.
Không có dữ liệu để xác nhận đạt/chưa đạt. Không dùng ảnh fixture trong test
phần mềm làm chứng cứ độ chính xác.
