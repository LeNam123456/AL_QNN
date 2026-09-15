# GÓI DEMO TRỰC QUAN HÓA MÔ HÌNH LƯỢNG TỬ VỚI LIT (LEARNING INTERPRETABILITY TOOL)

Thư mục này đóng gói toàn bộ mã nguồn, dữ liệu dự đoán từ mô hình **AL-QNN (Adaptive Layered Quantum Neural Network)** với **tên biến cụ thể, khoảng giá trị min-max và giá trị trung bình tự động gợi ý**.

---

## 📂 Cấu trúc Thư mục Đóng gói

```
lit_demo_package/
├── lit_demo.py                            # Web server LIT (Hiển thị khoảng Min-Max & gợi ý điền mẫu)
├── export_predictions.py                  # Script trích xuất dự đoán từ QNN
├── predictions_predictive_maintenance.csv # Dataset 6-qubit với tên biến cơ khí + khoảng giá trị
├── predictions_heart.csv                  # Dataset 8-qubit với tên biến tim mạch + khoảng giá trị
├── run_predictive_maintenance.bat         # Khởi chạy nhanh dataset Cơ khí (Port 7001)
├── run_heart.bat                          # Khởi chạy nhanh dataset Tim mạch (Port 7002)
├── run_demo.ps1                           # Script chạy trên PowerShell
└── README.md                              # Tài liệu hướng dẫn chi tiết
```

---

## 🎯 Bảng Ánh Xạ Tên Biến & Khoảng Giá Trị Gợi Ý

### 1. Bài toán AI4I Predictive Maintenance (Mạch 6 Qubits)
| Tên biến hiển thị trên LIT | Khoảng hợp lệ $[Min - Max]$ | Giá trị trung bình gợi ý | Ý nghĩa nghiệp vụ |
| :--- | :---: | :---: | :--- |
| `air_temperature [0.0 - 3.1]` | $0.0 \rightarrow 3.14\text{ rad}$ | `2.06` | Nhiệt độ môi trường không khí xung quanh máy |
| `process_temperature [0.0 - 3.1]` | $0.0 \rightarrow 3.14\text{ rad}$ | `1.97` | Nhiệt độ phát sinh trong quá trình gia công |
| `rotational_speed [0.0 - 3.1]` | $0.0 \rightarrow 3.14\text{ rad}$ | `1.54` | Tốc độ quay của trục chính máy cắt gọt |
| `torque [0.0 - 3.2]` | $0.0 \rightarrow 3.23\text{ rad}$ | `0.90` | Mô-men xoắn tác dụng lên đầu trục dao |
| `tool_wear [0.0 - 3.1]` | $0.0 \rightarrow 3.14\text{ rad}$ | `1.59` | Tổng thời gian dao đã cắt gọt (mức độ mòn) |
| `machine_type_encoded [0.0 - 3.1]` | $0.0 \rightarrow 3.14\text{ rad}$ | `1.18` | Loại máy phân cấp công suất (L / M / H) |

* **Nhãn phân loại:** `ok` (Bình thường - Lớp 0) vs `failure` (Sự cố hỏng máy - Lớp 1).
* **Cổng truy cập:** **[http://localhost:7001](http://localhost:7001)**

---

### 2. Bài toán Heart Disease Cleveland (Mạch 8 Qubits)
| Tên biến hiển thị trên LIT | Khoảng hợp lệ $[Min - Max]$ | Giá trị trung bình gợi ý | Ý nghĩa nghiệp vụ |
| :--- | :---: | :---: | :--- |
| `age [0.0 - 3.1]` | $0.0 \rightarrow 3.14\text{ rad}$ | `1.80` | Tuổi của bệnh nhân |
| `sex [0.0 - 3.1]` | $0.0 \rightarrow 3.14\text{ rad}$ | `1.38` | Giới tính (0: Nữ, 1: Nam) |
| `chest_pain_type [0.0 - 3.1]` | $0.0 \rightarrow 3.14\text{ rad}$ | `1.92` | Loại đau ngực (Điển hình / Không điển hình / ...) |
| `resting_blood_pressure [0.0 - 3.1]` | $0.0 \rightarrow 3.14\text{ rad}$ | `1.58` | Huyết áp tâm thu đo lúc nghỉ ngơi |
| `serum_cholesterol [0.0 - 3.5]` | $0.0 \rightarrow 3.51\text{ rad}$ | `1.56` | Nồng độ Cholesterol trong huyết thanh |
| `fasting_blood_sugar [0.0 - 3.1]` | $0.0 \rightarrow 3.14\text{ rad}$ | `1.34` | Đường huyết lúc đói (> 120 mg/dl: 1, ngược lại: 0) |
| `resting_ecg [0.0 - 3.1]` | $0.0 \rightarrow 3.14\text{ rad}$ | `1.31` | Kết quả đo điện tâm đồ lúc nghỉ |
| `max_heart_rate [0.0 - 3.3]` | $0.0 \rightarrow 3.28\text{ rad}$ | `2.02` | Nhịp tim tối đa đo được khi gắng sức |

* **Nhãn phân loại:** `healthy` (Khỏe mạnh - Lớp 0) vs `disease` (Có bệnh tim - Lớp 1).
* **Cổng truy cập:** **[http://localhost:7002](http://localhost:7002)**

---

## 🚀 Cách Chạy Demo

### Cách 1: Click đúp vào file `.bat` (Khuyên dùng trên Windows)
* Click đúp `run_predictive_maintenance.bat` $\rightarrow$ Mở trình duyệt vào `http://localhost:7001`.
* Click đúp `run_heart.bat` $\rightarrow$ Mở trình duyệt vào `http://localhost:7002`.

### Cách 2: Chạy từ dòng lệnh Terminal
```bash
cd d:\al_qnn\al_qnn_project\lit_demo_package

# Chạy cho Predictive Maintenance
python lit_demo.py predictive_maintenance

# Hoặc chạy cho Heart Disease
python lit_demo.py heart
```
