"""
scripts/09_demo_quantum_inference.py

Load checkpoint đã lưu và thực hiện Inference siêu tốc trên tập Test độc lập.
Minh chứng tính khả dụng của mô hình AL-QNN sau khi huấn luyện.
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(Path(__file__).resolve().parent))

from src.synthetic_bp.stage3.circuit_backend import PennyLaneBackend

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/qnn_best.npz")
    ap.add_argument("--test-data", default="checkpoints/test_data.npz")
    args = ap.parse_args()

    ckpt_path = ROOT / args.ckpt
    test_path = ROOT / args.test_data

    if not ckpt_path.exists() or not test_path.exists():
        print(f"Lỗi: Không tìm thấy checkpoint hoặc test_data. Hãy chạy script 08_save_qnn_checkpoint.py trước.")
        return

    print("=== TẢI CHECKPOINT VÀ DỮ LIỆU TEST ===")
    ckpt = np.load(ckpt_path, allow_pickle=True)
    test_data = np.load(test_path, allow_pickle=True)

    X_test = test_data["X_test"]
    y_test = test_data["y_test"]
    
    n_qubits = int(ckpt["n_qubits"])
    
    # Tái tạo snapshot
    snapshot = {
        "depth": int(ckpt["depth"]),
        "entangler_strength": float(ckpt["entangler_strength"]),
        "theta": ckpt["theta"],
        "masks": ckpt["masks"]
    }

    print(f"Checkpoint info: Qubits={n_qubits}, Depth={snapshot['depth']}, Entangler Strength={snapshot['entangler_strength']:.2f}")
    print(f"Test size: {len(X_test)} samples")

    print("\n=== KHỞI TẠO MẠCH LƯỢNG TỬ ĐỂ INFERENCE ===")
    # Chỉ truyền X_test, y_test vào làm val để dùng hàm evaluate()
    backend = PennyLaneBackend(
        X_train=np.empty((0, n_qubits)), y_train=np.empty((0,)),
        X_val=X_test, y_val=y_test,
        n_qubits=n_qubits, initial_depth=2, max_depth=16,
        device_name="default.qubit"
    )

    backend.restore(snapshot)
    print("[OK] Đã nạp thành công trọng số vào mạch.")

    print("\n=== THỰC HIỆN QUANTUM INFERENCE ===")
    start_time = time.time()
    
    metrics = backend.evaluate()
    
    inf_time = time.time() - start_time
    
    print(f"Thời gian Inference cho {len(X_test)} mẫu: {inf_time:.4f} giây")
    if len(X_test) > 0:
        print(f"Tốc độ trung bình: {(inf_time/len(X_test))*1000:.2f} ms/mẫu")

    print("\n=== KẾT QUẢ TRÊN TẬP TEST ĐỘC LẬP ===")
    print(f"- Loss: {metrics['validation_loss']:.4f}")
    print(f"- Accuracy: {metrics['accuracy']:.4f}")
    print(f"- F1-Score: {metrics['f1']:.4f}")
    print(f"- ROC AUC: {metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
