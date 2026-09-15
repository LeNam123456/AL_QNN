import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

ROOT = Path(".")

count = 0
for p in ROOT.rglob("learning_curve.csv"):
    if ".venv" in p.parts:
        continue
    try:
        df = pd.read_csv(p)
        plt.figure(figsize=(8, 4))
        plt.plot(df["episode"], df["return"], alpha=0.3, label="return")
        if "return_ma100" in df.columns:
            plt.plot(df["episode"], df["return_ma100"], label="moving avg (100)")
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.legend()
        plt.title("RL Scheduler - Learning Curve")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        out_img = p.parent / "learning_curve.png"
        plt.savefig(out_img, dpi=120)
        plt.close()
        count += 1
        print(f"[{count}] Fixed & re-rendered plot: {out_img}")
    except Exception as e:
        print(f"Error rendering {p}: {e}")

print(f"Done! Total plots re-rendered: {count}")
