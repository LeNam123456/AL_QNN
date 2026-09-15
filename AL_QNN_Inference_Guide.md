# Hướng Dẫn Inference AL-QNN: Tái Sử Dụng Mô Hình Không Cần Huấn Luyện Lại

Tài liệu này trình bày quy trình đóng gói mô hình Adaptive Layer Quantum Neural Network (AL-QNN) và thực hiện dự đoán (Inference) tốc độ cao trên dữ liệu mới mà **hoàn toàn không cần phải huấn luyện lại**. 

Quy trình này đáp ứng hai tiêu chí quan trọng trong nghiên cứu Học Máy Lượng Tử:
1. **Tính khách quan khoa học**: Đánh giá hiệu năng trên tập kiểm thử (Test Set) độc lập, chưa từng xuất hiện trong quá trình tối ưu hóa.
2. **Triển khai thực tế**: Lưu trữ các tham số lượng tử tối ưu (độ sâu, góc quay qubit) để tải lại siêu tốc.

---

## 1. Cơ Chế Đóng Gói Mô Hình (Checkpointing)

AL-QNN không sử dụng độ sâu cố định mà dùng tác tử học tăng cường (RL Scheduler) hoặc luật (Rule-based) để tự động quyết định thêm/bớt lớp (layer). Do đó, cấu trúc mạch của mô hình thay đổi liên tục.

Để đóng gói, **Parameter Registry** của hệ thống sẽ theo dõi các thay đổi và thu thập một **Snapshot** toàn diện của mạch khi đạt hiệu năng tốt nhất trên tập Validation.

Một snapshot (được lưu dưới dạng `.npz` bằng Numpy) bao gồm:
- `depth`: Độ sâu tối ưu của mạch ($D_{opt}$).
- `entangler_strength`: Trọng số của cổng vướng víu.
- `theta`: Ma trận các góc xoay (shape: `[depth, n_qubits, 3]`).
- `masks`: Ma trận cờ quyết định lớp nào được kích hoạt hay bị cắt tỉa.
- `n_qubits`: Số lượng Qubits của mạch.

---

## 2. Quy Trình 3-Way Split (Train / Val / Test)

Dữ liệu đầu vào được tự động chia làm 3 phần độc lập:
- **Tập Train (70%)**: Để Gradient Descent tối ưu hóa các tham số `theta`.
- **Tập Val (15%)**: Để Scheduler quyết định thêm/bớt lớp nhằm tránh Barren Plateaus và Overfitting.
- **Tập Test (15%)**: Bị "khóa" hoàn toàn trong suốt quá trình trên. Tập này được lưu tách biệt ra tệp `test_data.npz` để dành riêng cho đánh giá cuối cùng.

---

## 3. Cách Thức Hoạt Động Của Inference (Không Huấn Luyện)

Thay vì phải chạy lại chu trình huấn luyện tốn kém, quá trình Inference diễn ra chỉ trong vài mili-giây với các bước sau:

### Bước 1: Khởi tạo mạch lượng tử "trống"
Mô hình khởi tạo một đối tượng `PennyLaneBackend` hoàn toàn không có dữ liệu Train, chỉ cần số `n_qubits` tương ứng với Checkpoint.

### Bước 2: Nạp tham số (Restore Checkpoint)
Gọi hàm `backend.restore(snapshot)` để hệ thống tự động:
- Cấu hình lại mạch lượng tử có độ sâu đúng bằng `depth` đã lưu.
- Khôi phục nguyên vẹn các góc quay `theta` và `masks`.

### Bước 3: Dự đoán
Dữ liệu từ tập Test (hoặc dữ liệu thực tế mới) được đưa thẳng vào mạch. Vì không cần tính Gradient (Backpropagation), mạch chỉ thực hiện truyền thẳng (Forward Pass), mang lại tốc độ phản hồi tính bằng mili-giây trên mỗi mẫu.

---

## 4. Hướng Dẫn Chạy Script Thực Tế

Bạn có thể chạy thử nghiệm quy trình này thông qua hai script đã được thiết kế sẵn trong thư mục `scripts/`:

### A. Sinh Checkpoint và Data
Chạy lệnh sau để huấn luyện mô hình, lưu lại checkpoint và tập Test:
```bash
python scripts/08_save_qnn_checkpoint.py
```
*Kết quả sinh ra:* 
- `checkpoints/qnn_best.npz` (Trọng số mô hình)
- `checkpoints/test_data.npz` (Dữ liệu Test)

### B. Chạy Inference Siêu Tốc
Chạy lệnh sau để tải thẳng tệp `.npz` và đo hiệu năng dự đoán trên tập Test:
```bash
python scripts/09_demo_quantum_inference.py
```
*Bạn sẽ nhận thấy:* Quá trình khởi tạo mạch và nạp trọng số diễn ra gần như tức thời. Kết quả độ chính xác (Accuracy), Loss, và ROC AUC sẽ được in ra dựa trên tập Test độc lập.

---

## 5. Kết Luận
Việc tách biệt môi trường Train và Inference giúp AL-QNN sẵn sàng để có thể tích hợp vào các pipeline lớn hơn, hoặc thậm chí triển khai trực tiếp lên các **Máy tính Lượng tử vật lý thật (Real Quantum Hardware)** trong tương lai (khi đó chỉ cần gửi cấu trúc mạch và `theta` đã được pre-train lên máy thật để chạy dự đoán).
