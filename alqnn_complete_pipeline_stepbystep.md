# Pipeline Hoàn Chỉnh AL-QNN: Từng Bước Chi Tiết

> Tài liệu này mô tả **TOÀN BỘ quy trình từ A đến Z** của dự án AL-QNN (Adaptive Layered Quantum Neural Network), tích hợp **Môi trường Synthetic Teacher-Student (Fixed-Qubit & Ground-Truth Depth)**, **Hiệu chuẩn Reward (Reward Calibration)** và **Đánh giá Đa nền tảng (Sim-to-Real Deployment)**.

---

## Sơ Đồ Tổng Quan Pipeline (Full System Overview)

```mermaid
graph TB
    subgraph "STAGE 0: Chuẩn Bị Dữ Liệu & Synthetic Data Engine"
        S0_Data[Nạp Dataset Thực: BCW, Parkinsons, German Credit] --> S0_PCA[PCA Giảm Chiều<br/>leakage-safe]
        S0_TS[Teacher-Student Generator<br/>synthetic_ts<br/>Fixed D_teacher & N_qubits] --> S0_Split
        S0_PCA --> S0_Split[Chia Train / Val / Test<br/>70 / 15 / 15<br/>Balanced Sampling]
        S0_Split --> S0_Enc[Mã hóa Lượng tử<br/>Angle Embedding]
    end

    subgraph "STAGE 1A: Phân Tích BP Không Dữ Liệu"
        S0_Enc -.-> S1A
        S1A[Random BP Engine<br/>Tham số ngẫu nhiên θ<br/>Không nạp X] -->|Sweep: n_qubits<br/>depth, cost_type<br/>topology| S1A_Out[CSV Artifacts<br/>grad_variance<br/>near_zero_ratio<br/>theo depth]
    end

    subgraph "STAGE 1B: Phân Tích BP Có Dữ Liệu"
        S0_Enc --> S1B[Task-Aware Engine<br/>Nạp X & y<br/>Tính BCE Loss] -->|So sánh với 1A<br/>Bóc tách nguồn BP| S1B_Out[CSV Artifacts<br/>Data-induced<br/>variance delta]
    end

    subgraph "STAGE 1C: Xây Dựng Bảng Hiệu Chuẩn"
        S1A_Out --> S1C[Calibration Builder<br/>Welford Algorithm]
        S1B_Out --> S1C
        S1C -->|τ = β√(Nθ·Var)| S1C_Out[(calibration_rules.json<br/>+ threshold_tables.csv<br/>Bảng Rủi ro theo<br/>n_qubits, depth, cost_type)]
    end

    subgraph "STAGE 2: Bộ Lập Lịch Scheduler & Calibrated RL"
        S1C_Out --> S2{Scheduler<br/>Hot-swappable}
        S2 --> S2_Rule[Rule-Based<br/>Luật tĩnh tra τ]
        S2 --> S2_Greedy[Greedy Gradient<br/>ADAPT-VQE style]
        S2 --> S2_DQN[RL-DQN<br/>Value-based]
        S2 --> S2_PPO[RL-PPO<br/>Policy-gradient]
    end

    subgraph "STAGE 3: Huấn Luyện Thích Ứng An Toàn"
        S2_Rule & S2_Greedy & S2_DQN & S2_PPO -->|MutationRequest| S3_Mut[Mutation Engine<br/>Soft-Masking<br/>Cosine Annealing]
        S3_Mut <--> S3_Reg[Parameter Registry<br/>Snapshot / Rollback]
        S3_Mut --> S3_Back{Circuit Backend}
        S3_Back --> S3_Sim[Surrogate Backend<br/>NumPy heuristics]
        S3_Back --> S3_PL[PennyLane Backend<br/>Real QNode]
        S3_PL --> S3_Train[Adaptive Trainer<br/>Vòng lặp Epoch]
        S3_Train -->|Telemetry Metrics| S2
    end

    subgraph "STAGE 4: Đánh Giá Ground-Truth & Triển Khai Thực Tế"
        S3_Train --> S4_Synth[Script 07: Ground-Truth Benchmark<br/>Môi trường Teacher-Student<br/>Đo Error = |D_end - D_teacher|]
        S3_Train --> S4_Real[Script 08: Deploy Dữ Liệu Thực<br/>BCW, German Credit, Parkinsons<br/>Đánh giá Accuracy & Stability]
    end
```

---

## Bước 0: Chuẩn Bị Dữ Liệu (Stage 0 — Data Preparation & Synthetic Engine)

**Mục tiêu:** Cung cấp nguồn dữ liệu chuẩn hóa cho mạch lượng tử, bao gồm cả **Dữ liệu Thực tế** và **Dữ liệu Lượng tử Tổng hợp (Teacher-Student)**.

**Quy trình chi tiết:**
1. **Dữ liệu Thực tế (Real Datasets):** 
   - Hỗ trợ các bộ dữ liệu benchmark: Breast Cancer (BCW), Predictive Maintenance, Pima Diabetes, Parkinsons, German Credit.
   - PCA giảm chiều xuống đúng số qubit $N$ (fit trên tập Train, transform trên Val/Test để chống leakage).
   - Angle Embedding: Mã hóa góc quay $RX(x_i)$ trong khoảng $[0, \pi]$.
2. **Dữ liệu Lượng tử Tổng hợp Teacher-Student (`synthetic_ts`):**
   - Triển khai trong module `src/synthetic_bp/synthetic_data.py`.
   - Sinh nhãn $y$ từ một mạch Teacher cố định $U_{\text{teacher}}$ với số qubit $N$ và độ sâu $D_{\text{teacher}}$ biết trước.
   - Đảm bảo cân bằng nhãn nhị phân chuẩn 50/50 qua phân ngưỡng trung vị $E(x) > \text{median}(E)$.
   - Cung cấp "Môi trường có đáp án Ground-Truth" để đánh giá khả năng hội tụ độ sâu của RL Agents.

**File liên quan:** `src/synthetic_bp/synthetic_data.py`, `scripts/dataset.py`, `scripts/00_bootstrap_project.py`

---

## Bước 1A & 1B: Phân Tích Barren Plateaus (Stage 1 — BP Benchmarking)

**Mục tiêu:** Bóc tách phương sai gradient thành 2 nguồn: **Cấu trúc Mạch (Data-free)** và **Dữ liệu Đầu vào (Task-aware)**.

**Quy trình chi tiết:**
1. **Stage 1A (Data-free BP):** Quét ngẫu nhiên góc $\theta \sim [-\pi, \pi]$ trên các cấu hình $(N, \text{Depth}, \text{Cost}, \text{Topology})$. Tính phương sai gradient online bằng thuật toán **Welford** để tiết kiệm bộ nhớ RAM.
2. **Stage 1B (Task-aware BP):** Nạp dữ liệu $X, y$ thực tế, đo phương sai gradient của hàm loss BCE. So sánh delta phương sai với Stage 1A để xác định tác động của phân phối dữ liệu lên ranh giới Barren Plateaus.

**File liên quan:** `src/synthetic_bp/stage1a/random_bp_engine.py`, `src/synthetic_bp/stage1b/task_aware_bp_engine.py`

---

## Bước 1C: Xây Dựng Bảng Tra Cứu Hiệu Chuẩn (Stage 1C — Calibration Tables)

**Mục tiêu:** Tạo bảng rủi ro `calibration_rules.json` chứa ngưỡng rủi ro $\tau_{\text{empirical}} = \beta \sqrt{N_\theta \cdot \text{Var}}$.

**File liên quan:** `src/synthetic_bp/stage1c/calibration_builder.py`, `src/synthetic_bp/stage1c/thresholds.py`

---

## Bước 2 & 3: Lập Lịch Thích Ứng & Huấn Luyện RL (Stage 2 & 3 — Adaptive Training & RL Env)

**Mục tiêu:** Cho phép 4 Schedulers (**Rule-Based, Greedy, DQN, PPO**) điều khiển động cấu trúc mạng QNN (gắn/tỉa layer, khóa tăng trưởng).

**Cơ chế cốt lõi:**
- **Hot-swappable Schedulers:** Mọi bộ lập lịch kế thừa từ `SchedulerBase` và phát lệnh `MutationRequest`.
- **Soft-Masking Engine (`MutationEngine`):** Khi thêm layer mới, áp dụng Cosine Annealing để chuyển tiếp êm từ weight 0 ➔ weight cập nhật, tránh làm sụp đổ các tham số đã học trước đó.
- **Parameter Registry (`ParameterRegistry`):** Snapshot và Rollback trọng số nếu mutation gây hại cho mô hình.
- **Gym RL Environment (`Stage3RLEnv`):** Môi trường huấn luyện tác tử RL hỗ trợ cả `SimulateBackend` (NumPy siêu nhanh) lẫn `PennyLaneBackend` (Mô phỏng vật lý thật).

**File liên quan:** `src/synthetic_bp/stage2/*`, `src/synthetic_bp/stage3/*`

---

## Bước 4: Đánh Giá Ground-Truth & Triển Khai Thực Tế (Stage 4 — Deployment & Benchmark)

**Mục tiêu:** Đánh giá hiệu năng toàn diện trên 2 kịch bản chính:

### 1. Ground-Truth Benchmark (Script 07):
- Chạy đối sánh cả 4 Schedulers trên môi trường Synthetic Teacher-Student với $D_{\text{teacher}} \in \{2, 4, 6\}$.
- Đo lường chỉ số **Ground-Truth Convergence Error**: $\text{Error} = |D_{\text{end}} - D_{\text{teacher}}|$.
- **Phát hiện quan trọng:** Kiểm chứng tác tử PPO/DQN dừng đúng ở $D_{\text{end}} \approx D_{\text{teacher}}$, trong khi Greedy bị kẹt do lạm dụng thêm lớp vào vùng Barren Plateaus.

### 2. Triển khai Dữ liệu Thực tế đã Hiệu chuẩn (Script 08):
- Train RL Agent trên môi trường Multi-Teacher với bộ trọng số Reward đã hiệu chuẩn (Reward Calibration).
- Deploy Agent sang các dữ liệu thực tế (BCW, Parkinsons, German Credit, Pima).
- Đánh giá chỉ số Accuracy, F1-score và độ ổn định cấu trúc $D_{\text{end}}$.

**File liên quan:** `scripts/07_run_synthetic_teacher_experiment.py`, `scripts/08_calibrate_reward_and_evaluate_real.py`, `scripts/05_compare_schedulers.py`
