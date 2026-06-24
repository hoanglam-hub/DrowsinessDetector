# Driver Drowsiness & Distraction Detection System

> Hệ thống phát hiện buồn ngủ và mất tập trung khi lái xe sử dụng Computer Vision, AI nhẹ và các luật cảnh báo theo thời gian thực.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Realtime%20Vision-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Face%20Mesh-orange)
![TensorFlow Lite](https://img.shields.io/badge/TensorFlow%20Lite-Edge%20AI-red)
![Status](https://img.shields.io/badge/Status-Prototype%20Ready-brightgreen)

---

## 1. Giới thiệu dự án

**Driver Drowsiness & Distraction Detection System** là dự án phát hiện trạng thái nguy hiểm của tài xế trong quá trình lái xe, bao gồm:

- Buồn ngủ, ngủ gật, nhắm mắt kéo dài.
- Ngáp liên tục hoặc có dấu hiệu mệt mỏi.
- Mất tập trung khi lái xe.
- Cúi đầu xuống dưới, có khả năng đang nhìn điện thoại hoặc thao tác với điện thoại.
- Không nhìn thẳng về phía trước trong thời gian dài.

Dự án hướng tới triển khai thực tế trên ô tô với thiết bị nhỏ gọn gồm **camera, máy tính xử lý AI, loa cảnh báo và bộ chuyển đổi nguồn từ tẩu ô tô sang nguồn DC phù hợp**.

Mục tiêu chính của hệ thống là cảnh báo sớm cho tài xế trước khi tình trạng buồn ngủ hoặc mất tập trung gây nguy hiểm khi tham gia giao thông.

---

## 2. Bài toán đặt ra

Trong quá trình lái xe, tài xế có thể rơi vào các trạng thái nguy hiểm như mệt mỏi, buồn ngủ, mất tập trung hoặc cúi xuống sử dụng điện thoại. Những hành vi này thường diễn ra trong thời gian ngắn nhưng có thể dẫn đến hậu quả nghiêm trọng.

Hệ thống được xây dựng để theo dõi khuôn mặt tài xế theo thời gian thực, phân tích các đặc trưng sinh học và hành vi, sau đó đưa ra cảnh báo theo 3 mức độ khác nhau.

Dự án không chỉ phát hiện “ngủ gật” đơn thuần, mà còn mở rộng sang nhóm hành vi **mất tập trung khi lái xe**, đặc biệt là tình huống tài xế **cúi xuống nhìn điện thoại**.

---

## 3. Tính năng chính

### Phát hiện buồn ngủ và ngủ gật

Hệ thống theo dõi mắt, miệng và trạng thái khuôn mặt để phát hiện các dấu hiệu như:

- Mắt nhắm trong thời gian dài.
- Tần suất chớp mắt bất thường.
- Ngáp nhiều hoặc mở miệng lâu.
- Gương mặt có xu hướng mệt mỏi.
- Model AI dự đoán trạng thái drowsy/fatigue với độ tin cậy cao.

### Phát hiện mất tập trung khi lái xe

Ngoài buồn ngủ, hệ thống có thể cảnh báo khi tài xế có dấu hiệu mất tập trung:

- Cúi đầu xuống dưới trong thời gian dài.
- Hướng nhìn lệch khỏi phía trước.
- Gương mặt biến mất khỏi khung hình.
- Có hành vi nghi vấn như nhìn xuống điện thoại hoặc thao tác bên dưới vô lăng.

Đây là phần quan trọng khi triển khai trên ô tô, vì không phải tình huống nguy hiểm nào cũng đến từ buồn ngủ. Việc cúi xuống nhìn điện thoại trong vài giây cũng có thể tạo ra rủi ro lớn khi xe đang di chuyển.

### Cảnh báo 3 mức độ

Hệ thống sử dụng 3 mức cảnh báo để tránh báo động quá sớm hoặc quá muộn:

| Mức cảnh báo | Trạng thái | Ý nghĩa |
|---|---|---|
| Level 1 | Cảnh báo nhẹ | Tài xế có dấu hiệu mệt mỏi, chớp mắt chậm, mất tập trung ngắn |
| Level 2 | Cảnh báo trung bình | Mắt nhắm lâu hơn, ngáp nhiều, cúi đầu hoặc nhìn lệch kéo dài |
| Level 3 | Cảnh báo nguy hiểm | Ngủ gật rõ ràng, mất tập trung nghiêm trọng, cúi xuống lâu hoặc không quan sát đường |

Mỗi mức cảnh báo có thể sử dụng một âm thanh riêng:

- `alert_level1.wav`: nhắc nhở nhẹ.
- `alert_level2.wav`: cảnh báo rõ ràng hơn.
- `alert_level3.wav`: cảnh báo khẩn cấp.

---

## 4. Công nghệ sử dụng

Dự án sử dụng các công nghệ phù hợp cho bài toán xử lý thời gian thực và triển khai trên thiết bị biên:

| Thành phần | Công nghệ |
|---|---|
| Xử lý hình ảnh | OpenCV |
| Nhận diện khuôn mặt | MediaPipe Face Mesh |
| Trích xuất đặc trưng | EAR, MAR, Head Pose, Face Direction |
| Mô hình học sâu | GRU / LSTM sequence model |
| Runtime nhẹ | TensorFlow Lite / TFLite Runtime |
| Phát âm thanh cảnh báo | Pygame |
| Ngôn ngữ phát triển | Python |
| Môi trường triển khai | Laptop cũ, mini PC, Raspberry Pi, thiết bị Edge AI |

---

## 5. Nguyên lý hoạt động

Hệ thống hoạt động theo quy trình sau:

1. Camera ghi hình khuôn mặt tài xế theo thời gian thực.
2. OpenCV đọc từng khung hình từ camera.
3. MediaPipe Face Mesh xác định các điểm đặc trưng trên khuôn mặt.
4. Hệ thống tính toán các chỉ số quan trọng như:
   - EAR: độ mở của mắt.
   - MAR: độ mở của miệng.
   - Head Pose: hướng đầu.
   - Face Direction: hướng nhìn tương đối.
   - Thời gian mắt nhắm.
   - Thời gian cúi đầu hoặc nhìn lệch.
5. Các đặc trưng được đưa vào model AI để dự đoán trạng thái tài xế.
6. Kết quả AI được kết hợp với các luật thực tế để quyết định mức cảnh báo.
7. Hệ thống phát âm thanh cảnh báo tương ứng với Level 1, Level 2 hoặc Level 3.

Cách tiếp cận này giúp hệ thống ổn định hơn so với việc chỉ dùng AI hoặc chỉ dùng ngưỡng thủ công.

---

## 6. Logic phát hiện và cảnh báo

Dự án kết hợp giữa **model AI** và **rule-based detection**.

### 6.1. Phát hiện bằng chỉ số mắt

Chỉ số EAR được dùng để đánh giá mắt đang mở hay nhắm. Khi EAR thấp hơn ngưỡng trong nhiều khung hình liên tiếp, hệ thống có thể xác định tài xế đang nhắm mắt lâu.

Ví dụ:

- EAR thấp trong thời gian ngắn: cảnh báo Level 1.
- EAR thấp kéo dài: cảnh báo Level 2.
- EAR rất thấp hoặc mắt nhắm quá lâu: cảnh báo Level 3.

### 6.2. Phát hiện bằng chỉ số miệng

Chỉ số MAR được dùng để phát hiện hành vi ngáp. Nếu MAR cao trong một khoảng thời gian nhất định, hệ thống có thể ghi nhận tài xế đang ngáp.

Nếu hành vi ngáp xuất hiện nhiều lần trong thời gian ngắn, hệ thống tăng mức cảnh báo vì đây là dấu hiệu mệt mỏi rõ ràng.

### 6.3. Phát hiện cúi đầu, nhìn xuống điện thoại

Hệ thống sử dụng hướng đầu và vị trí khuôn mặt để phát hiện tài xế cúi xuống dưới. Đây là tình huống thường gặp khi tài xế nhìn điện thoại, nhặt đồ hoặc thao tác bên dưới vô lăng.

Các dấu hiệu có thể dùng để nhận biết:

- Góc cúi đầu vượt ngưỡng cho phép.
- Mắt và khuôn mặt hướng xuống dưới trong nhiều khung hình liên tiếp.
- Khuôn mặt bị lệch khỏi vùng quan sát chính.
- Không quay lại nhìn đường sau một khoảng thời gian ngắn.

Khi tài xế cúi xuống trong thời gian ngắn, hệ thống có thể cảnh báo Level 1. Nếu cúi xuống kéo dài hoặc lặp lại nhiều lần, cảnh báo sẽ tăng lên Level 2 hoặc Level 3.

### 6.4. Phát hiện bằng model AI

Model GRU/LSTM phân tích chuỗi đặc trưng theo thời gian thay vì chỉ nhìn vào một khung hình đơn lẻ. Điều này giúp hệ thống hiểu được diễn biến hành vi của tài xế, ví dụ:

- Mắt đang dần nhắm lâu hơn.
- Tần suất ngáp tăng lên.
- Gương mặt mệt mỏi kéo dài.
- Trạng thái chuyển từ bình thường sang fatigue hoặc drowsy.

Model AI đưa ra xác suất dự đoán, sau đó hệ thống kết hợp với các luật thực tế để tránh cảnh báo sai.

---

## 7. Lý do kết hợp AI và rule-based

Nếu chỉ dùng ngưỡng EAR/MAR, hệ thống dễ báo sai trong các trường hợp như tài xế chớp mắt lâu, quay mặt tạm thời hoặc điều kiện ánh sáng kém.

Nếu chỉ dùng model AI, hệ thống có thể khó kiểm soát trong tình huống thực tế, nhất là khi dữ liệu train chưa đủ đa dạng.

Vì vậy, dự án kết hợp cả hai hướng:

- **AI model** giúp nhận diện trạng thái tổng thể theo chuỗi thời gian.
- **Rule-based detection** giúp kiểm soát các tình huống rõ ràng như mắt nhắm lâu, ngáp, cúi đầu, mất mặt khỏi camera.
- **Cơ chế cảnh báo 3 mức** giúp phản ứng phù hợp với độ nguy hiểm.

Đây là hướng phù hợp cho một hệ thống cần chạy ổn định trong môi trường thật như ô tô.

---

## 8. Cài đặt môi trường

Khuyến nghị sử dụng Python 3.10 để đảm bảo tương thích tốt với OpenCV, MediaPipe, TensorFlow/TFLite và các thư viện âm thanh.

```bash
python -m venv .venv
```

Kích hoạt môi trường ảo trên Windows:

```bash
.venv\Scripts\activate
```

Cài đặt thư viện:

```bash
pip install -r requirements.txt
```

Một bộ thư viện tối thiểu thường gồm:

```txt
opencv-python
mediapipe
numpy
pygame
scikit-learn
pandas
tensorflow-cpu
```

Khi chỉ chạy runtime nhẹ với model `.tflite`, có thể dùng `tflite-runtime` thay cho TensorFlow nếu thiết bị hỗ trợ.

---

## 9. Huấn luyện model

Quy trình huấn luyện thường gồm 3 bước:

1. Thu thập video hoặc ảnh khuôn mặt tài xế ở nhiều trạng thái.
2. Trích xuất đặc trưng khuôn mặt thành dữ liệu chuỗi thời gian.
3. Huấn luyện model GRU/LSTM để phân loại trạng thái.

Các nhóm trạng thái có thể sử dụng:

| Nhãn | Ý nghĩa |
|---|---|
| normal | Tài xế tỉnh táo, nhìn đường bình thường |
| distracted | Tài xế mất tập trung, nhìn lệch, cúi đầu hoặc thao tác khác |
| fatigue | Tài xế mệt mỏi, ngáp, chớp mắt chậm |
| drowsy | Tài xế buồn ngủ/ngủ gật rõ ràng |

Sau khi train xong, model nên được chuyển sang TensorFlow Lite để chạy nhẹ hơn trên thiết bị thật.

```bash
python train_model.py
```

Model sau khi export có thể có dạng:

```txt
drowsiness_model.tflite
```

---

## 10. Chạy hệ thống runtime

Khi đã có model `.tflite`, có thể chạy hệ thống phát hiện thời gian thực bằng lệnh:

```bash
python drowsiness_detector_tflite_runtime.py
```

Trên Windows, có thể tạo file `.bat` để chạy nhanh:

```bat
@echo off
cd /d D:\DrowsineDetector
call .venv_runtime\Scripts\activate.bat
python drowsiness_detector_tflite_runtime.py
pause
```

Khi triển khai thực tế, có thể cấu hình để chương trình tự chạy sau khi máy tính khởi động khoảng 1–2 phút, giúp hệ thống ổn định hơn so với chạy ngay lập tức khi vừa bật nguồn.

---

## 11. Triển khai thực tế trên ô tô

Khi đưa hệ thống lên ô tô, cần quan tâm đến 4 phần chính:

### 11.1. Thiết bị xử lý

Có thể sử dụng một trong các lựa chọn sau:

- Laptop cũ đã bỏ màn hình.
- Mini PC tiết kiệm điện.
- Raspberry Pi 4/5.
- Thiết bị Edge AI phù hợp với TensorFlow Lite.

Với dự án nhỏ, laptop cũ hoặc mini PC thường dễ triển khai hơn vì tương thích tốt với Python, OpenCV, MediaPipe và camera USB.

### 11.2. Camera

Camera nên đặt ở vị trí nhìn rõ mặt tài xế, thường là:

- Gần gương chiếu hậu trong xe.
- Trên taplo, hướng về mặt tài xế.
- Không bị che bởi vô lăng hoặc kính.

Camera cần đủ góc nhìn để nhận diện được mắt, miệng và hướng đầu. Nếu sử dụng ban đêm, nên cân nhắc camera có hỗ trợ ánh sáng yếu hoặc hồng ngoại.

### 11.3. Âm thanh cảnh báo

Loa cảnh báo nên đủ lớn để tài xế nghe rõ trong môi trường có tiếng động cơ, tiếng đường và âm thanh trong xe.

Nên dùng 3 âm báo khác nhau cho 3 cấp độ để tài xế dễ phân biệt mức nguy hiểm.

### 11.4. Nguồn điện trên ô tô

Khi triển khai trên ô tô, không nên cấp nguồn trực tiếp một cách tùy tiện. Cần dùng **bộ chuyển đổi nguồn từ tẩu ô tô sang DC** phù hợp với thiết bị xử lý.

Một số phương án nguồn:

| Thiết bị | Nguồn thường dùng | Gợi ý cấp nguồn trên ô tô |
|---|---|---|
| Laptop cũ | 19V DC | Bộ chuyển tẩu ô tô 12V/24V sang 19V DC |
| Mini PC | 12V hoặc 19V DC | Bộ chuyển tẩu ô tô sang đúng điện áp của mini PC |
| Raspberry Pi | 5V USB-C | Tẩu sạc ô tô USB-C chất lượng cao, đủ dòng |
| Camera USB | 5V USB | Lấy nguồn trực tiếp từ thiết bị xử lý |

Khi chọn bộ chuyển nguồn, cần chú ý:

- Điện áp đầu ra phải đúng với thiết bị.
- Dòng điện/công suất phải đủ tải.
- Nên có bảo vệ quá áp, quá dòng, quá nhiệt.
- Nên có cầu chì hoặc mạch bảo vệ để an toàn khi dùng trên xe.
- Dây nguồn cần chắc chắn, không lỏng khi xe rung.

Ví dụ: nếu dùng laptop cũ yêu cầu sạc 19V, cần dùng bộ chuyển đổi **tẩu ô tô 12V sang DC 19V** có công suất phù hợp, thay vì dùng sạc USB thông thường.

---

## 12. Khuyến nghị cấu hình triển khai

### Phương án tiết kiệm

- Laptop cũ hoặc mini PC cũ.
- Webcam USB.
- Loa USB hoặc loa 3.5mm.
- Bộ chuyển tẩu ô tô sang DC đúng điện áp.
- Chạy model TensorFlow Lite.

Phương án này phù hợp để thử nghiệm thực tế, demo sản phẩm và triển khai nguyên mẫu.

### Phương án gọn nhẹ

- Raspberry Pi 4/5 hoặc thiết bị Edge AI.
- Camera Pi hoặc camera USB.
- Loa nhỏ.
- Nguồn USB-C hoặc DC ổn định từ tẩu ô tô.
- Tối ưu model `.tflite` để giảm tải CPU.

Phương án này nhỏ gọn hơn nhưng cần tối ưu kỹ hơn về hiệu năng.

---

## 13. Ưu điểm của dự án

- Có thể chạy thời gian thực bằng camera phổ thông.
- Kết hợp AI và luật thực tế để tăng độ tin cậy.
- Hỗ trợ 3 mức cảnh báo rõ ràng.
- Có thể phát hiện cả buồn ngủ và mất tập trung.
- Có khả năng nhận diện tình huống tài xế cúi xuống nhìn điện thoại.
- Phù hợp để triển khai trên laptop cũ, mini PC hoặc thiết bị Edge AI.
- Có thể mở rộng để ghi log, chụp ảnh sự kiện hoặc gửi cảnh báo về trung tâm.

---

## 14. Giới hạn hiện tại

Hệ thống vẫn có thể bị ảnh hưởng bởi một số yếu tố thực tế:

- Ánh sáng quá yếu hoặc quá chói.
- Camera đặt sai góc.
- Tài xế đeo kính đen hoặc khẩu trang che nhiều khuôn mặt.
- Dữ liệu train chưa đủ đa dạng.
- Xe rung mạnh làm hình ảnh bị mờ.
- Một số hành vi cúi đầu không phải lúc nào cũng là dùng điện thoại.

Vì vậy, khi triển khai thật, cần hiệu chỉnh ngưỡng, vị trí camera và dữ liệu train theo môi trường sử dụng thực tế.

---

## 15. Hướng phát triển tiếp theo

Các hướng có thể mở rộng trong tương lai:

- Bổ sung nhận diện điện thoại trên tay bằng object detection.
- Lưu lịch sử cảnh báo theo thời gian.
- Gửi cảnh báo qua mạng nội bộ hoặc 4G.
- Thêm giao diện dashboard theo dõi trạng thái tài xế.
- Tối ưu model để chạy mượt hơn trên Raspberry Pi hoặc mini PC.
- Thêm chế độ tự khởi động sau khi xe cấp nguồn.
- Hỗ trợ camera hồng ngoại để chạy tốt vào ban đêm.
- Kết hợp GPS hoặc tốc độ xe để điều chỉnh mức cảnh báo.

---

## 16. Mục tiêu ứng dụng

Dự án có thể được ứng dụng trong:

- Xe cá nhân.
- Xe khách.
- Xe tải đường dài.
- Xe công vụ.
- Hệ thống giám sát an toàn lái xe.
- Mô hình nghiên cứu Edge AI trong giao thông.

Với chi phí triển khai hợp lý, hệ thống có thể trở thành một giải pháp hỗ trợ an toàn lái xe, giúp giảm nguy cơ tai nạn do buồn ngủ hoặc mất tập trung.

---

## 17. Tuyên bố an toàn

Hệ thống này là công cụ hỗ trợ cảnh báo, không thay thế trách nhiệm quan sát và điều khiển phương tiện của tài xế.

Trong triển khai thực tế, cần kiểm thử kỹ trong nhiều điều kiện khác nhau trước khi sử dụng lâu dài trên xe.

---

## 18. Tác giả / Đơn vị phát triển

Tác giả: Hoàng Lâm

---

## 19. License

Dự án có thể sử dụng cho mục đích học tập, nghiên cứu, thử nghiệm và phát triển nguyên mẫu. Khi triển khai thương mại hoặc đưa vào hệ thống vận hành thực tế, cần kiểm thử, đánh giá độ tin cậy và tuân thủ các yêu cầu an toàn liên quan.
