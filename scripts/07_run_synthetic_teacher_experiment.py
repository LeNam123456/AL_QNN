"""
scripts/07_run_synthetic_teacher_experiment.py

Thực nghiệm đối sánh Môi trường Teacher-Student cho AL-QNN.
So sánh 4 Schedulers (Rule-Based, Greedy, DQN, PPO) trên các tập dữ liệu tổng hợp
được sinh từ Teacher QNN Circuit có độ sâu cố định D_teacher (ví dụ: D_teacher = 3, 4, 6).
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from dataset import load_binary_dataset
from src.synthetic_bp.stage2.rule_based_scheduler import RuleBasedScheduler
from src.synthetic_bp.stage2.greedy_scheduler import GreedyScheduler
from src.synthetic_bp.stage2.rl_scheduler import RLScheduler
from src.synthetic_bp.stage3.trainer import AdaptiveTrainer
from src.synthetic_bp.stage3.circuit_backend import SimulateBackend, PennyLaneBackend

MACROS = ["keep", "add_layer", "stop_growth",
          "propose_prune_layer", "propose_reduce_entanglement"]


def build_backend(args, data, seed):
    if args.backend == "pennylane":
        return PennyLaneBackend(
            data["X_train"], data["y_train"], data["X_val"], data["y_val"],
            n_qubits=args.n_qubits, initial_depth=args.initial_depth,
            max_depth=args.max_depth, topology=args.topology,
            entangler_type=args.entangler, cost_type=args.cost_type, seed=seed,
            device_name=getattr(args, 'device_name', 'lightning.gpu'), diff_method="best")
    return SimulateBackend(n_qubits=args.n_qubits, initial_depth=args.initial_depth,
                           max_depth=args.max_depth, cost_type=args.cost_type,
                           topology=args.topology, seed=seed)


class GreedyPolicy:
    def __init__(self, net, torch, kind):
        self.net = net.eval()
        self.torch = torch
        self.kind = kind

    def act(self, state, mask):
        t = self.torch
        s = t.as_tensor(np.asarray(state, dtype=np.float32)).unsqueeze(0)
        mask = np.asarray(mask, dtype=bool)
        with t.no_grad():
            if self.kind == "ppo":
                logits, _ = self.net(s, t.as_tensor(mask).unsqueeze(0))
                return int(logits.argmax(dim=1).item())
            q = self.net(s).squeeze(0).cpu().numpy()
            q[~mask] = -1e9
            return int(np.argmax(q))


def load_policy(kind, ckpt, rl_dir, hidden):
    import torch
    sys.path.append(str(rl_dir))
    if kind == "ppo":
        from agent_ppo_torch import ActorCritic
        net = ActorCritic(hidden=hidden)
    else:
        from agent_dqn_torch import QNet
        net = QNet(hidden=hidden)
    net.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return GreedyPolicy(net, torch, kind)


def deploy(name, scheduler, args, data):
    b = build_backend(args, data, seed=args.seed)
    tr = AdaptiveTrainer(b, scheduler, n_qubits=args.n_qubits,
                         cost_type=args.cost_type, topology=args.topology,
                         epochs=args.epochs,
                         log_telemetry=getattr(args, "log_telemetry", False),
                         telemetry_guard=getattr(args, "telemetry_guard", False))
    return tr.run()


def summarize(name, df, args, teacher_depth):
    first, last = df.iloc[0], df.iloc[-1]
    counts = df["scheduler_action"].value_counts().to_dict()
    row = {
        "scheduler": name,
        "teacher_depth": teacher_depth,
        "backend": args.backend,
        "seed": args.seed,
        "epochs": args.epochs,
        "acc_start": round(float(first["accuracy"]), 4),
        "acc_end": round(float(last["accuracy"]), 4),
        "acc_delta": round(float(last["accuracy"]) - float(first["accuracy"]), 4),
        "f1_end": round(float(last.get("f1", float("nan"))), 4),
        "depth_start": int(first["depth"]),
        "depth_end": int(last["depth"]),
        "depth_error": int(last["depth"]) - teacher_depth,
        "n_commits": int(last["n_commits"]),
        "n_rollbacks": int(last["n_rollbacks"]),
    }
    for a in MACROS:
        row[f"act_{a}"] = int(counts.get(a, 0))
    return row


def plot(results, teacher_depth, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    fig, ((ax_acc, ax_depth), (ax_bar, ax_err)) = plt.subplots(2, 2, figsize=(12, 8))
    for name, df in results:
        ax_acc.plot(df["epoch"], df["accuracy"], label=name, linewidth=1.8)
        ax_depth.step(df["epoch"], df["depth"], where="post", label=name, linewidth=1.8)
    
    ax_depth.axhline(teacher_depth, color="red", linestyle="--", label=f"Teacher Depth ({teacher_depth})")

    ax_acc.set(title=f"Accuracy (Teacher Depth={teacher_depth})", xlabel="epoch", ylabel="accuracy")
    ax_depth.set(title=f"Depth Trajectory vs Teacher Depth", xlabel="epoch", ylabel="depth")
    for ax in (ax_acc, ax_depth):
        ax.grid(alpha=0.3); ax.legend()

    names = [n for n, _ in results]
    acc_end = [float(df.iloc[-1]["accuracy"]) for _, df in results]
    ax_bar.bar(names, acc_end, color="skyblue")
    ax_bar.set(title="Final Accuracy", ylabel="accuracy", ylim=(0, 1))
    for i, v in enumerate(acc_end):
        ax_bar.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax_bar.grid(alpha=0.3, axis="y")

    depth_errors = [abs(int(df.iloc[-1]["depth"]) - teacher_depth) for _, df in results]
    ax_err.bar(names, depth_errors, color="coral")
    ax_err.set(title="|Student Depth - Teacher Depth| Error", ylabel="absolute error")
    for i, v in enumerate(depth_errors):
        ax_err.text(i, v + 0.05, f"{v}", ha="center", fontsize=9)
    ax_err.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def collect_schedulers(args):
    cal = args.calibration if Path(args.calibration).exists() else None
    scheds = [("rule", RuleBasedScheduler(calibration=cal))]
    if not args.no_greedy:
        scheds.append(("greedy", GreedyScheduler(calibration=cal)))

    for kind, ckpt in (("ppo", args.ppo_ckpt), ("dqn", args.dqn_ckpt)):
        if not ckpt or not Path(ckpt).exists():
            continue
        try:
            rl_dir = args.rl_root / kind.upper()
            hidden = args.ppo_hidden if kind == "ppo" \
                else tuple(int(x) for x in str(args.dqn_hidden).split(","))
            pol = load_policy(kind, ckpt, rl_dir, hidden)
            scheds.append((kind, RLScheduler(pol, tau=0.1, total_epochs=args.epochs,
                                             warmup_epochs=args.warmup, calibration=cal)))
        except Exception as e:
            print(f"[compare] bỏ qua {kind}: {e}")
    return scheds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["simulate", "pennylane"], default="pennylane")
    ap.add_argument("--device-name", default="lightning.gpu")
    ap.add_argument("--teacher-depth", type=int, default=4, help="Độ sâu mạch Teacher (3, 4, 6, ...)")
    ap.add_argument("--n-samples", type=int, default=500)
    ap.add_argument("--n-qubits", type=int, default=8)
    ap.add_argument("--initial-depth", type=int, default=2)
    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument("--cost-type", default="local")
    ap.add_argument("--topology", default="circular")
    ap.add_argument("--entangler", choices=["cnot", "cz"], default="cnot")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--calibration", default="outputs/stage1c/calibration_rules.json")
    ap.add_argument("--no-greedy", action="store_true")
    ap.add_argument("--ppo-ckpt", default=None)
    ap.add_argument("--dqn-ckpt", default=None)
    ap.add_argument("--rl-root", default=None)
    ap.add_argument("--ppo-hidden", type=int, default=128)
    ap.add_argument("--dqn-hidden", default="128,128")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    args.rl_root = Path(args.rl_root) if args.rl_root else ROOT / "RL"
    out_dir = Path(args.out or ROOT / "outputs" / "synthetic_teacher" / f"teacher_d{args.teacher_depth}")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_binary_dataset("synthetic_ts", n_qubits=args.n_qubits,
                               teacher_depth=args.teacher_depth,
                               max_samples=args.n_samples, seed=args.seed,
                               topology=args.topology)

    scheds = collect_schedulers(args)
    print(f"=== SYNTHETIC TEACHER-STUDENT EXPERIMENT ===")
    print(f"Teacher Depth: {args.teacher_depth} | N_Qubits: {args.n_qubits} | Backend: {args.backend}")
    print(f"Schedulers: {[s[0] for s in scheds]}")

    results, rows = [], []
    for name, sched in scheds:
        print(f"\n---> Running {name} scheduler...")
        df = deploy(name, sched, args, data)
        df.to_csv(out_dir / f"deploy_log_{name}.csv", index=False)
        results.append((name, df))
        rows.append(summarize(name, df, args, args.teacher_depth))

    comp = pd.DataFrame(rows)
    comp.to_csv(out_dir / "comparison.csv", index=False)
    plot(results, args.teacher_depth, out_dir / "comparison.png")

    cols = ["scheduler", "teacher_depth", "acc_start", "acc_end", "acc_delta",
            "f1_end", "depth_start", "depth_end", "depth_error", "n_commits", "n_rollbacks"]
    print("\n" + "="*80)
    print(comp[cols].to_string(index=False))
    print("="*80)
    print(f"Lưu kết quả tại -> {out_dir}")


if __name__ == "__main__":
    main()
