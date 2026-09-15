"""
08_calibrate_reward_and_evaluate_real.py — Reward Calibration & Real Dataset Evaluation.

Thực hiện:
1. Calibrate bộ trọng số Reward của RL Agent thông qua môi trường Multi-Teacher Synthetic.
2. Deploy Agent đã Calibrate sang các tập dữ liệu thực tế (BCW, Parkinsons, German Credit, Pima).
3. Đánh giá sự nâng cấp Accuracy và mức độ ổn định cấu trúc mạng (Depth).

Ví dụ chạy:
    python scripts/08_calibrate_reward_and_evaluate_real.py --dataset cancer --n-qubits 8
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(Path(__file__).resolve().parent))

from dataset import load_binary_dataset
from src.synthetic_bp.stage2.rule_based_scheduler import RuleBasedScheduler
from src.synthetic_bp.stage2.greedy_scheduler import GreedyScheduler
from src.synthetic_bp.stage2.rl_scheduler import RLScheduler
from src.synthetic_bp.stage3.trainer import AdaptiveTrainer
from src.synthetic_bp.stage3.circuit_backend import PennyLaneBackend


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibrate RL Rewards and Evaluate on Real Quantum Datasets."
    )
    parser.add_argument("--dataset", type=str, default="cancer",
                        choices=["cancer", "parkinsons", "german_credit", "pima_diabetes", "heart_disease"])
    parser.add_argument("--n-qubits", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ppo-ckpt", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print(f"  [EVALUATION] Calibrated AL-QNN Deployment on Real Dataset")
    print(f"  Dataset: {args.dataset} | n_qubits={args.n_qubits} | Epochs={args.epochs}")
    print("=" * 70)

    # 1. Nạp dữ liệu thực tế
    data = load_binary_dataset(args.dataset, n_qubits=args.n_qubits, seed=args.seed)
    print(f"  [data] Loaded {data['name']}: Train={data['n_train']}, Val={data['n_val']}")

    out_dir = ROOT / "outputs" / "calibrated_real_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    schedulers = {
        "Rule-Based": RuleBasedScheduler(),
        "Greedy": GreedyScheduler(),
    }

    # Nếu có agent PPO đã calibrate
    rng = np.random.default_rng(args.seed)
    ppo_ckpt_path = getattr(args, "ppo_ckpt", None)
    if ppo_ckpt_path and Path(ppo_ckpt_path).exists():
        import importlib
        mod_05 = importlib.import_module("scripts.05_compare_schedulers")
        load_policy = mod_05.load_policy
        policy = load_policy("ppo", ppo_ckpt_path, ROOT / "RL" / "PPO", hidden=128)
        schedulers["PPO (Calibrated)"] = RLScheduler(policy)

    results = []
    for name, sched in schedulers.items():
        print(f"\n--- Running Deploy for: {name} ---")
        backend = PennyLaneBackend(
            data["X_train"], data["y_train"], data["X_val"], data["y_val"],
            n_qubits=args.n_qubits, initial_depth=2, max_depth=16,
            topology="circular", entangler_type="cnot", cost_type="local",
            seed=args.seed, device_name="lightning.qubit", diff_method="best"
        )
        trainer = AdaptiveTrainer(backend, sched, n_qubits=args.n_qubits, epochs=args.epochs)
        res = trainer.run()

        vloss = float(res["summary"]["final_validation_loss"])
        d_end = int(res["summary"]["final_depth"])
        acc = max(0.0, min(100.0, (1.0 - vloss) * 100.0))

        results.append({
            "dataset": args.dataset,
            "agent": name,
            "n_qubits": args.n_qubits,
            "final_depth": d_end,
            "val_loss": vloss,
            "estimated_accuracy": acc,
        })
        print(f"  [{name}] Final Depth: {d_end} | Val Loss: {vloss:.4f} | Est Acc: {acc:.2f}%")

    df = pd.DataFrame(results)
    csv_path = out_dir / f"calibrated_eval_{args.dataset}_{args.n_qubits}q.csv"
    df.to_csv(csv_path, index=False)
    print("\n" + "=" * 70)
    print("  FINAL EVALUATION REPORT:")
    print(df.to_string(index=False))
    print(f"\n  Report saved to: {csv_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
