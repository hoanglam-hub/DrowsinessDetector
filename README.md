# Driver Drowsiness Detection – Phát hiện ngủ gật và cảnh báo âm thanh

Dự án xây dựng hệ thống **phát hiện trạng thái ngủ gật của người lái xe** bằng camera, trí tuệ nhân tạo và cảnh báo âm thanh theo nhiều cấp độ.  
Hệ thống sử dụng **MediaPipe Face Mesh** để trích xuất đặc trưng khuôn mặt, sau đó huấn luyện mô hình **GRU** để phân loại trạng thái người lái.

---

## 1. Mục tiêu dự án

Dự án hướng tới việc phát hiện sớm dấu hiệu buồn ngủ/ngủ gật của người lái xe, từ đó đưa ra cảnh báo âm thanh kịp thời nhằm giảm nguy cơ mất tập trung và tai nạn khi điều khiển phương tiện.

Hệ thống tập trung vào các dấu hiệu chính:

- Mắt nhắm lâu hoặc lim dim.
- Tần suất chớp mắt bất thường.
- Ngáp.
- Gục đầu hoặc cúi đầu.
- Trạng thái buồn ngủ kéo dài qua nhiều khung hình liên tiếp.

---

## 2. Chức năng chính

- Thu thập dữ liệu video từ người lái.
- Trích xuất đặc trưng khuôn mặt bằng MediaPipe Face Mesh.
- Tính toán các chỉ số hỗ trợ phát hiện ngủ gật:
  - EAR – Eye Aspect Ratio.
  - MAR – Mouth Aspect Ratio.
  - Góc đầu: pitch, yaw.
- Gom nhiều frame liên tiếp thành một sequence.
- Huấn luyện mô hình GRU để phân loại trạng thái.
- Xuất mô hình `.keras` và `.tflite`.
- Cảnh báo âm thanh theo 3 mức độ nguy hiểm.

---

## 3. Các mức cảnh báo âm thanh

Hệ thống có thể cảnh báo theo 3 mức:

| Mức cảnh báo | Trạng thái | Mô tả | Âm thanh gợi ý |
|---|---|---|---|
| Mức 1 | Cảnh báo nhẹ | Người lái có dấu hiệu mệt mỏi, mắt lim dim hoặc ngáp | Âm báo ngắn |
| Mức 2 | Cảnh báo nguy hiểm | Người lái có dấu hiệu ngủ gật rõ hơn, mắt nhắm lâu hơn | Âm báo liên tục |
| Mức 3 | Cảnh báo khẩn cấp | Người lái ngủ gật kéo dài hoặc gục đầu | Âm báo lớn, lặp nhanh |

> Lưu ý: Mô hình hiện tại huấn luyện theo 2 lớp `normal` và `drowsy`.  
> Ba mức cảnh báo được xác định ở giai đoạn chạy thực tế dựa trên xác suất dự đoán, thời gian kéo dài trạng thái drowsy, EAR, MAR và số frame liên tiếp có nguy cơ.

---

## 4. Công nghệ sử dụng

| Thành phần | Công nghệ |
|---|---|
| Ngôn ngữ lập trình | Python |
| Xử lý ảnh | OpenCV |
| Nhận diện landmark khuôn mặt | MediaPipe Face Mesh |
| Xử lý dữ liệu | NumPy, Pandas |
| Huấn luyện mô hình | TensorFlow / Keras |
| Mô hình học sâu | GRU |
| Chuẩn hóa đặc trưng | Scikit-learn StandardScaler |
| Triển khai edge AI | TensorFlow Lite |
| Cảnh báo âm thanh | Pygame hoặc module phát âm thanh |

---

## 5. Cấu trúc thư mục

```text
project/
├── data/
│   ├── normal/
│   │   ├── normal_01.mp4
│   │   ├── normal_02.mp4
│   │
│   └── drowsy/
│       ├── drowsy_01.mp4
│       ├── drowsy_02.mp4
│
├── data/processed/
│   ├── normal.csv
│   └── drowsy.csv
│
├── models/
│   ├── drowsiness_gru.keras
│   ├── drowsiness_gru.tflite
│   ├── drowsiness_gru_select_ops.tflite
│   ├── feature_scaler.joblib
│   ├── class_names.json
│   ├── training_history.csv
│   └── classification_report.txt
│
├── extract_to_csv_binary_fixed.py
├── train_model_binary_fixed.py
├── requirements.txt
└── README.md
```

---

## 6. Dữ liệu đầu vào

Dữ liệu được chia thành 2 lớp:

| Nhãn | Tên lớp | Mô tả |
|---|---|---|
| 0 | normal | Người lái tỉnh táo, mắt mở bình thường, không có dấu hiệu ngủ gật |
| 1 | drowsy | Người lái buồn ngủ, mắt lim dim, nhắm mắt lâu, ngáp hoặc gục đầu |

Cấu trúc dữ liệu:

```text
data/
├── normal/
│   ├── video_01.mp4
│   ├── video_02.mp4
│
└── drowsy/
    ├── video_01.mp4
    ├── video_02.mp4
```

---

## 7. Trích xuất đặc trưng ra CSV

Chạy file extract:

```bash
python extract_to_csv_binary_fixed.py --data-dir data --out-dir data/processed
```

Sau khi chạy xong, chương trình sẽ tạo các file CSV:

```text
data/processed/normal.csv
data/processed/drowsy.csv
```

Mỗi dòng trong CSV là một sequence gồm nhiều frame liên tiếp.

---

## 8. Đặc trưng sử dụng

Mỗi frame được trích xuất các đặc trưng chính:

| Đặc trưng | Ý nghĩa |
|---|---|
| `ear_L` | Độ mở mắt trái |
| `ear_R` | Độ mở mắt phải |
| `mar` | Độ mở miệng |
| `pitch` | Góc cúi/ngẩng đầu |
| `yaw` | Góc quay đầu trái/phải |

Một sequence gồm nhiều frame liên tiếp sẽ được đưa vào mô hình GRU để học diễn biến theo thời gian.

---

## 9. Huấn luyện mô hình

Chạy file train:

```bash
python train_model_binary_fixed.py --data-dir data/processed --num-classes 2
```

Có thể thay đổi số epoch:

```bash
python train_model_binary_fixed.py --data-dir data/processed --num-classes 2 --epochs 150
```

Sau khi train xong, các file mô hình sẽ được lưu trong thư mục:

```text
models/
```

---

## 10. Kết quả đầu ra sau khi train

| File | Ý nghĩa |
|---|---|
| `drowsiness_gru.keras` | Mô hình Keras tốt nhất |
| `drowsiness_gru.tflite` | Mô hình TensorFlow Lite |
| `drowsiness_gru_select_ops.tflite` | Bản TFLite dùng Select TF Ops nếu convert thường bị lỗi |
| `feature_scaler.joblib` | Bộ chuẩn hóa đặc trưng |
| `class_names.json` | Danh sách tên lớp |
| `training_history.csv` | Lịch sử huấn luyện |
| `classification_report.txt` | Báo cáo đánh giá mô hình |

---

## 11. Nguyên lý hoạt động tổng quát

Quy trình hệ thống:

```text
Camera
  ↓
OpenCV đọc frame
  ↓
MediaPipe Face Mesh phát hiện landmark khuôn mặt
  ↓
Tính EAR, MAR, pitch, yaw
  ↓
Gom nhiều frame thành sequence
  ↓
GRU dự đoán normal / drowsy
  ↓
Bộ xử lý cảnh báo đánh giá mức nguy hiểm
  ↓
Phát âm thanh cảnh báo theo 3 mức
```

---

## 12. Logic cảnh báo gợi ý

Khi chạy thực tế, không nên cảnh báo ngay chỉ sau 1 frame.  
Nên dựa trên nhiều frame liên tiếp để tránh báo sai.

Gợi ý:

```text
Mức 1:
- Model dự đoán drowsy với xác suất trung bình.
- Hoặc mắt nhắm/ngáp xuất hiện trong thời gian ngắn.

Mức 2:
- Model dự đoán drowsy nhiều frame liên tiếp.
- Hoặc EAR thấp kéo dài.

Mức 3:
- Drowsy kéo dài trong nhiều giây.
- Hoặc phát hiện mắt nhắm lâu kèm gục đầu.
```

Ví dụ ngưỡng:

| Mức | Điều kiện gợi ý |
|---|---|
| Mức 1 | Drowsy liên tục từ 1–2 giây |
| Mức 2 | Drowsy liên tục từ 2–4 giây |
| Mức 3 | Drowsy trên 4 giây hoặc gục đầu rõ ràng |

---

## 13. Cài đặt thư viện

Tạo môi trường ảo:

```bash
python -m venv .venv
```

Kích hoạt môi trường ảo trên Windows:

```bash
.venv\Scripts\activate
```

Cài thư viện:

```bash
pip install -r requirements.txt
```

Nếu chưa có `requirements.txt`, có thể cài trực tiếp:

```bash
pip install opencv-python mediapipe numpy pandas scikit-learn tensorflow joblib pygame
```

---

## 14. Gợi ý file `requirements.txt`

```text
opencv-python
mediapipe
numpy
pandas
scikit-learn
tensorflow
joblib
pygame
```

Nếu chỉ train model và chưa chạy cảnh báo âm thanh, có thể chưa cần `pygame`.

---

## 15. Lưu ý khi thu thập dữ liệu

Để mô hình đạt độ chính xác tốt hơn, dữ liệu nên có:

- Nhiều người khác nhau.
- Nhiều điều kiện ánh sáng.
- Nhiều góc quay camera.
- Video đủ dài, tối thiểu 20–60 giây mỗi video.
- Dữ liệu `normal` và `drowsy` tương đối cân bằng.
- Không trộn dữ liệu mất nhãn hoặc nhãn không rõ ràng.

Không nên đưa video người lái chỉ quay ngang, nhìn điện thoại hoặc mất tập trung vào lớp `drowsy` nếu người đó không thật sự buồn ngủ.

---

## 16. Hướng phát triển tiếp theo

- Tối ưu mô hình để chạy trên Raspberry Pi hoặc laptop cũ.
- Thêm giao diện hiển thị trạng thái người lái.
- Thêm module cảnh báo âm thanh 3 mức.
- Tự động lưu log khi phát hiện nguy hiểm.
- Thêm thống kê thời gian người lái ở trạng thái mệt mỏi.
- Tối ưu TFLite để chạy nhanh hơn trên thiết bị edge.
- Kết hợp thêm camera hồng ngoại để hoạt động tốt hơn ban đêm.

---

## 17. Trạng thái dự án

Dự án hiện hỗ trợ:

- Trích xuất đặc trưng từ video.
- Huấn luyện mô hình GRU.
- Phân loại 2 trạng thái: `normal` và `drowsy`.
- Xuất mô hình phục vụ triển khai thực tế.
- Định hướng cảnh báo âm thanh theo 3 mức.

---

## 18. Tác giả / đơn vị thực hiện

Dự án phục vụ nghiên cứu, thử nghiệm hệ thống phát hiện ngủ gật sử dụng AI và edge AI.

---

## 19. Ghi chú

Đây là dự án thử nghiệm, kết quả phụ thuộc nhiều vào chất lượng dữ liệu, góc quay camera và điều kiện ánh sáng thực tế.  
Khi triển khai trên phương tiện, cần kiểm thử kỹ trước khi sử dụng trong môi trường thật.
