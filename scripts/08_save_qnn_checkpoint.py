"""
scripts/08_save_qnn_checkpoint.py

Huấn luyện mô hình AL-QNN trên tập dữ liệu với 3-Way Split (Train/Val/Test).
Sau khi huấn luyện xong, lưu trạng thái mô hình (snapshot) vào thư mục checkpoints.
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(Path(__file__).resolve().parent))

import numpy as np

from dataset import load_binary_dataset
from src.synthetic_bp.stage2.rule_based_scheduler import RuleBasedScheduler
from src.synthetic_bp.stage3.trainer import AdaptiveTrainer
from src.synthetic_bp.stage3.circuit_backend import PennyLaneBackend

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="synthetic_ts")
    ap.add_argument("--n-qubits", type=int, default=4)
    ap.add_argument("--teacher-depth", type=int, default=4)
    ap.add_argument("--n-samples", type=int, default=150)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--out-ckpt", default="checkpoints/qnn_best.npz")
    args = ap.parse_args()

    out_ckpt = ROOT / args.out_ckpt
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)

    print(f"=== ĐANG TẢI DỮ LIỆU ({args.dataset}) VỚI 3-WAY SPLIT ===")
    data = load_binary_dataset(
        name=args.dataset,
        n_qubits=args.n_qubits,
        teacher_depth=args.teacher_depth,
        max_samples=args.n_samples,
        val_frac=0.15,
        test_frac=0.15,
        seed=42
    )
    print(f"Train: {data['n_train']}, Val: {data['n_val']}, Test: {data['n_test']}")

    # Lưu tập test để script 09 dùng
    test_data_path = out_ckpt.parent / "test_data.npz"
    np.savez(test_data_path, X_test=data["X_test"], y_test=data["y_test"])
    print(f"Đã lưu tập test độc lập vào {test_data_path}")

    device_name = "default.qubit"
    
    print(f"\n=== KHỞI TẠO PENNYLANE BACKEND ({device_name}) ===")
    backend = PennyLaneBackend(
        X_train=data["X_train"], y_train=data["y_train"],
        X_val=data["X_val"], y_val=data["y_val"],
        n_qubits=args.n_qubits, initial_depth=2, max_depth=8,
        device_name=device_name
    )

    scheduler = RuleBasedScheduler()
    trainer = AdaptiveTrainer(backend, scheduler, n_qubits=args.n_qubits, epochs=args.epochs)
    
    print("\n=== BẮT ĐẦU HUẤN LUYỆN AL-QNN ===")
    start_time = time.time()
    df = trainer.run()
    train_time = time.time() - start_time
    
    print(f"\nHuấn luyện xong trong {train_time:.2f} giây.")
    last_row = df.iloc[-1]
    print(f"Epoch {args.epochs}: Acc={last_row['accuracy']:.4f}, Depth={last_row['depth']}")

    # Lấy snapshot checkpoint và lưu lại
    snapshot = backend.snapshot()
    
    # Save bằng np.savez
    np.savez(
        out_ckpt,
        depth=snapshot["depth"],
        entangler_strength=snapshot["entangler_strength"],
        theta=snapshot["theta"],
        masks=snapshot["masks"],
        n_qubits=args.n_qubits
    )
    print(f"\n[OK] Đã lưu Checkpoint mô hình vào: {out_ckpt}")


if __name__ == "__main__":
    main()
