# ĐÁNH GIÁ HIỆU NĂNG CÁC CHIẾN LƯỢC TỐI ƯU CẤU TRÚC MẠCH LƯỢNG TỬ (AL-QNN) DƯỚI TÁC ĐỘNG CỦA BARREN PLATEAUS

## 1. Thiết lập Thí nghiệm (Experimental Setup)
Thí nghiệm được tiến hành nhằm so sánh 4 chiến lược lập lịch (Schedulers) trong việc thiết kế kiến trúc động cho Mạng nơ-ron lượng tử (QNN): 2 chiến lược Heuristic (Rule-based, Greedy) và 2 chiến lược Học tăng cường (PPO, DQN). 
*   **Môi trường huấn luyện RL:** Các tác tử PPO và DQN được huấn luyện trên môi trường Surrogate giả lập để tối ưu chi phí.
*   **Môi trường triển khai (Deploy):** Mô phỏng vật lý trên `PennyLane` (tăng tốc bằng GPU).
*   **Siêu tham số (Hyperparameters):** Tập dữ liệu ung thư vú (BCW), hàm mục tiêu `local`, chiều sâu ban đầu $D_{start} = 2$, tối đa $D_{max} = 12$.

## 2. Kết quả trên hệ 6-Qubit: Lợi thế của chiến lược "Tham lam" (Greedy)
Trên hệ 6-qubit, không gian Hilbert còn tương đối nhỏ, hiện tượng Barren Plateaus (Sập gradient) chưa thực sự nghiêm trọng. 
*   **Số liệu:** Chiến lược **Greedy** đạt độ chính xác (Accuracy) cao nhất với **72.51%**, vượt qua Rule-based (66.08%) và DQN (65.5%). PPO gặp hiện tượng overfit ở môi trường giả lập nên bị sụp đổ (34.5%).
*   **Phân tích kiến trúc:** Agent Greedy đạt được hiệu năng này bằng cách liên tục thêm lớp (`act_add_layer` = 6), đẩy chiều sâu mạch lên $D_{end} = 8$. Vì hiện tượng sập gradient ở 6-qubit chưa đủ mạnh để triệt tiêu việc cập nhật tham số, Greedy đã thành công dùng "bạo lực" (brute-force) để tăng độ biểu diễn (expressivity) của mạch, từ đó fit dữ liệu tốt hơn.

## 3. Kết quả trên hệ 8-Qubit: Sự toả sáng của Học tăng cường (RL)
Khi mở rộng hệ thống lên 8-qubit, không gian trạng thái phình to gấp 4 lần. Lúc này, Barren Plateaus trở thành "nút thắt cổ chai" chí mạng.
*   **Số liệu:** Trái ngược với kịch bản 6-qubit, **PPO** đã vươn lên trở thành chiến lược xuất sắc nhất với độ chính xác **70.76%**. Greedy tụt xuống chỉ còn 64.33% và Rule-based đạt 63.74%. DQN bị sụp đổ ở mức 32.16%.
*   **Phân tích kiến trúc:** Điểm mấu chốt tạo nên chiến thắng của PPO nằm ở kiến trúc mà nó sinh ra. Trong khi Greedy mù quáng nhồi thêm lớp (đạt $D_{end} = 8$) và lao thẳng vào "vùng chết" của Barren Plateaus khiến model không thể học tiếp, thì **PPO lại đưa ra quyết định cực kỳ khôn ngoan**. Nó chỉ thêm đúng 1 lớp (`act_add_layer` = 1), giữ nguyên mạch ở trạng thái siêu nông ($D_{end} = 3$), sau đó liên tục chọn hành động giữ nguyên (`act_keep`). 

## 4. Đánh giá tác động của Thời gian huấn luyện và Số vòng lặp tối ưu
Nghiên cứu đã tiến hành so sánh đối chiếu giữa hai phiên bản cấu hình:
*   **Cấu hình A (Thử nghiệm nhanh):** Huấn luyện RL 15 episodes trên Surrogate -> Triển khai thực tế 60 epochs.
*   **Cấu hình B (Huấn luyện sâu):** Huấn luyện RL 1000 episodes trên Surrogate -> Triển khai thực tế 40 epochs.

### 4.1. Sự "giác ngộ" của PPO (Impact of RL Episodes)
*   Ở **Cấu hình A (15 eps)**, do thời gian học quá ngắn, các tác tử RL gần như hành động theo bản năng (random) hoặc bắt chước. Điển hình ở bài toán 8-qubit, PPO sao chép y hệt hành vi của Greedy (cùng vươn lên $Depth = 8$ và mắc kẹt ở độ chính xác $64.91\%$). Lúc này, RL chưa chứng minh được sự vượt trội.
*   Sang **Cấu hình B (1000 eps)**, sự khác biệt đã xảy ra. PPO "học" được quy luật tàn khốc của Barren Plateaus trong môi trường 8-qubit giả lập. Khi mang sang môi trường thật, thay vì bắt chước Greedy, PPO chủ động kìm hãm mạch ở mức $Depth = 3$. Nhờ chiến lược xuất sắc này, nó đã lách qua được vùng sập đạo hàm và lật ngược tình thế để vươn lên top 1 ($70.76\%$). Điều này chứng minh: RL cần đủ thời gian cọ xát để hình thành được Policy kháng lại Barren Plateaus.

### 4.2. Rủi ro "Quá khớp" (Sim-to-Real Overfitting)
Bên cạnh mặt tích cực, việc train tới 1000 eps cũng làm lộ ra nhược điểm kinh điển của RL: **Khoảng cách giữa Giả lập và Thực tế (Sim-to-Real Gap).** 
*   Trong **Cấu hình B (1000 eps)**, PPO đột ngột sụp đổ ở bài toán 6-qubit (Accuracy rớt thảm hại xuống $34.5\%$, thực hiện tỉa layer $5$ lần). Tương tự, DQN cũng sụp đổ ở bài toán 8-qubit ($32.16\%$). 
*   **Nguyên nhân:** Do mô trường Surrogate là một xấp xỉ toán học chứ không phải mô phỏng lượng tử vật lý hoàn hảo. Việc train quá lâu khiến RL bị "quá khớp" (overfit) vào những đặc tính giả của Surrogate. Khi sang môi trường thực tế, Agent tự tin ra quyết định sai lầm. Điều này mở ra hướng nghiên cứu tiếp theo về việc hiệu chuẩn môi trường Surrogate sao cho sát với vật lý thực tế hơn.

### 4.3. Điểm cân bằng của Tối ưu hoá Gradient (Deploy Epochs)
*   Ở Cấu hình A (60 epochs), thuật toán tối ưu (Adam) có nhiều thời gian cập nhật trọng số hơn, giúp độ chính xác nhỉnh hơn vài phần trăm so với 40 epochs. Tuy nhiên, các đường biên giới về mặt cấu trúc mạng (Depth) và xu hướng hành vi của 4 Agents ở 40 epochs không hề thay đổi so với 60 epochs.
*   Việc giảm số vòng lặp xuống 40 Epochs là một quyết định đánh đổi (Trade-off) khôn ngoan, giúp tiết kiệm tới 33% thời gian chạy GPU mà vẫn bảo toàn hoàn toàn tính toàn vẹn của kết luận đánh giá kiến trúc lượng tử.

## 5. Kết luận (Conclusion)
Kết quả đối sánh giữa hệ 6-qubit và 8-qubit đã chứng minh ba luận điểm quan trọng:
1.  **Sự thất bại của Heuristic ở Scale lớn:** Chiến lược dựa trên luật cứng hoặc tối ưu cục bộ (Greedy) có xu hướng lạm dụng việc thêm tham số. Khi ứng dụng vào hệ lượng tử lớn, nó kích hoạt Barren Plateaus và làm hỏng mô hình.
2.  **Khả năng điều hướng của Policy-based RL:** Tác tử PPO đã tự phát hiện ra ranh giới giữa khả năng biểu diễn (Expressivity) và khả năng huấn luyện (Trainability) để tự động chọn cấu trúc tối ưu.
3.  **Tính chuyển giao (Transferability):** Việc huấn luyện RL trên không gian Surrogate, sau đó Deploy thành công trên môi trường vật lý PennyLane GPU mở ra hướng đi tiết kiệm tài nguyên khổng lồ cho bài toán Quantum Architecture Search (QAS).
