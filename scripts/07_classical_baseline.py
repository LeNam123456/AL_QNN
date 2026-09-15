"""
07_classical_baseline.py — Đánh giá các mô hình học máy cổ điển (Classical ML Baselines)
để so sánh đối chứng trực tiếp với Mô hình Mạch Lượng tử (VQC / AL-QNN).

Các mô hình cổ điển:
  - Logistic Regression (Tuyến tính)
  - Random Forest (Cây phi tuyến / Ensemble)
  - Support Vector Machine - SVM RBF (Không gian Hilbert cổ điển)

Tính năng:
  - Sử dụng CÙNG pipeline nạp dữ liệu (dataset.py) và cùng số chiều PCA (n_qubits).
  - Sử dụng CÙNG bộ metrics: Accuracy, F1-Score (zero_division=0), ROC-AUC.
  - Tương thích 100% định dạng cột với comparison.csv của VQC để tự động gộp bảng.

Cách chạy:
  # Chạy 1 dataset (vd: heart, 8 qubits)
  python scripts/07_classical_baseline.py --dataset heart --n-qubits 8

  # Chạy nhiều dataset và tự động ghép với file kết quả VQC
  python scripts/07_classical_baseline.py --dataset heart,cancer,pima,predictive_maintenance --n-qubits 8 --vqc-comparison-csv outputs/comparison/heart_8q_simulate_rl/comparison.csv
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

# Fix Windows console encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

MODELS = {
    "logreg": lambda seed: __import__(
        "sklearn.linear_model", fromlist=["LogisticRegression"]
    ).LogisticRegression(max_iter=2000, random_state=seed),
    "random_forest": lambda seed: __import__(
        "sklearn.ensemble", fromlist=["RandomForestClassifier"]
    ).RandomForestClassifier(n_estimators=200, random_state=seed),
    "svm": lambda seed: __import__(
        "sklearn.svm", fromlist=["SVC"]
    ).SVC(probability=True, random_state=seed),
}


def load_data(dataset: str, n_qubits: int, max_samples: int | None, seed: int):
    from dataset import load_binary_dataset
    d = load_binary_dataset(dataset, n_qubits=n_qubits, max_samples=max_samples, seed=seed)
    n_test = d.get("n_test", 0)
    print(f"[data] {d['name']}: train={d['n_train']} val={d['n_val']} test={n_test} n_features={n_qubits}")
    return d


def evaluate(model, X_eval, y_eval):
    from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
    preds = model.predict(X_eval)
    probs = model.predict_proba(X_eval)[:, 1]
    try:
        auc = float(roc_auc_score(y_eval, probs))
    except Exception:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_eval, preds)),
        "f1": float(f1_score(y_eval, preds, zero_division=0)),
        "roc_auc": auc,
    }


def run_one(model_name: str, dataset: str, data: dict, seed: int, eval_split: str = "val"):
    model = MODELS[model_name](seed)
    t0 = time.perf_counter()
    model.fit(data["X_train"], data["y_train"])
    fit_time = time.perf_counter() - t0

    X_eval = data["X_test"] if (eval_split == "test" and "X_test" in data and len(data["X_test"]) > 0) else data["X_val"]
    y_eval = data["y_test"] if (eval_split == "test" and "y_test" in data and len(data["y_test"]) > 0) else data["y_val"]

    m = evaluate(model, X_eval, y_eval)
    return {
        "scheduler": f"classical_{model_name}",
        "backend": "classical",
        "dataset": dataset,
        "seed": seed,
        "epochs": None,
        "acc_start": float("nan"),
        "acc_end": round(m["accuracy"], 4),
        "acc_delta": float("nan"),
        "f1_end": round(m["f1"], 4),
        "roc_auc_end": round(m["roc_auc"], 4),
        "depth_start": 0,
        "depth_end": 0,
        "n_commits": 0,
        "n_rollbacks": 0,
        "fit_time_s": round(fit_time, 4),
    }


def plot(rows: list[dict], vqc_rows: list[dict], out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    all_rows = list(vqc_rows) + list(rows)
    names = [r["scheduler"] for r in all_rows]
    acc = [r["acc_end"] for r in all_rows]
    colors = ["#5b8def" if r.get("backend") == "classical" else "#e07a3f" for r in all_rows]

    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(names)), 4.5))
    ax.bar(names, acc, color=colors)
    ax.set(title="VQC (Quantum) vs Classical ML Baselines",
           ylabel="Accuracy (Validation)", ylim=(0, 1.05))
    ax.tick_params(axis="x", rotation=30)
    for i, a in enumerate(acc):
        ax.text(i, a + 0.015, f"{a:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Chạy Classical ML Baselines đối chứng với VQC")
    ap.add_argument("--dataset", default="heart",
                    help="1 hoặc nhiều dataset cách nhau bởi dấu phẩy (vd: 'heart,cancer,pima,predictive_maintenance')")
    ap.add_argument("--n-qubits", type=int, default=8,
                    help="Số feature sau PCA (phải khớp với số qubit n_qubits của VQC để so sánh công bằng)")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--models", default="logreg,random_forest,svm")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--eval-split", choices=["val", "test"], default="val",
                    help="Tập dữ liệu dùng để đánh giá: 'val' hoặc 'test'")
    ap.add_argument("--vqc-comparison-csv", default=None,
                    help="Đường dẫn file comparison.csv của VQC để tự động gộp bảng và vẽ biểu đồ so sánh")
    ap.add_argument("--out", default=None, help="Thư mục xuất kết quả CSV")
    args = ap.parse_args()

    datasets = [d.strip() for d in args.dataset.split(",") if d.strip()]
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    out_dir = Path(args.out or ROOT / "outputs" / "comparison" / "classical")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print("        CHẠY THỰC NGHIỆM CLASSICAL BASELINES (ĐỐI CHỨNG VQC)     ")
    print("=================================================================")
    print(f"  Datasets: {', '.join(datasets)}")
    print(f"  Models: {', '.join(model_names)}")
    print(f"  Features (n_qubits): {args.n_qubits}")
    print(f"  Seed: {args.seed} | Split: {args.eval_split}")
    print("=================================================================\n")

    rows = []
    for dataset in datasets:
        data = load_data(dataset, args.n_qubits, args.max_samples, args.seed)
        for model_name in model_names:
            print(f"  -> Fitting {model_name:15s} trên [{dataset}]...")
            res = run_one(model_name, dataset, data, args.seed, args.eval_split)
            rows.append(res)

    comp = pd.DataFrame(rows)
    csv_out = out_dir / "comparison_classical.csv"
    comp.to_csv(csv_out, index=False)

    print("\n-----------------------------------------------------------------")
    print("KẾT QUẢ CÁC MÔ HÌNH HỌC MÁY CỔ ĐIỂN (CLASSICAL ML):")
    print("-----------------------------------------------------------------")
    cols = ["scheduler", "dataset", "acc_end", "f1_end", "roc_auc_end", "fit_time_s"]
    print(comp[cols].to_string(index=False))
    print(f"\n[OK] Đã lưu kết quả tại: {csv_out}")

    vqc_rows = []
    if args.vqc_comparison_csv and Path(args.vqc_comparison_csv).exists():
        vqc_path = Path(args.vqc_comparison_csv)
        vqc_df = pd.read_csv(vqc_path)
        if "backend" not in vqc_df.columns:
            vqc_df["backend"] = "pennylane"
        vqc_rows = vqc_df.to_dict("records")
        merged = pd.concat([vqc_df, comp], ignore_index=True, sort=False)
        merged_csv = out_dir / "comparison_quantum_vs_classical.csv"
        merged.to_csv(merged_csv, index=False)
        print(f"\n[OK] Đã gộp thành công với kết quả VQC -> {merged_csv}")
        
        plot_path = out_dir / "comparison_quantum_vs_classical.png"
        plot(rows, vqc_rows, plot_path)
        print(f"[OK] Đã vẽ biểu đồ so sánh -> {plot_path}")
    else:
        print("\n(Gợi ý: Truyền thêm --vqc-comparison-csv <path> để tự động ghép bảng và vẽ biểu đồ Quantum vs Classical)")


if __name__ == "__main__":
    main()
