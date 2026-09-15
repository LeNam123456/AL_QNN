"""
lit_demo.py — Trực quan hóa và diễn giải kết quả mô hình AL-QNN với LIT (Learning Interpretability Tool).

Dữ liệu đầu vào: GIÁ TRỊ THỰC TẾ RAW (Tuổi thực, Nhiệt độ K, Huyết áp mmHg, rpm, Nm...)
"""
from __future__ import annotations
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
import lit_nlp
from lit_nlp import dev_server
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types

ROOT = Path(__file__).resolve().parent

DATASETS_CFG = {
    "predictive_maintenance": dict(
        key="Predictive_Maintenance_Raw",
        csv="predictions_predictive_maintenance.csv",
        friendly="AI4I Predictive Maintenance (Dữ liệu cơ khí thực tế)",
        pos="failure", neg="ok", port=7001),
    "heart": dict(
        key="Heart_Disease_Raw",
        csv="predictions_heart.csv",
        friendly="Heart Disease Cleveland (Dữ liệu y khoa thực tế)",
        pos="disease", neg="healthy", port=7002),
}

EXCLUDE_COLS = {"true_label", "proba_class1", "pred_label"}


class PredictionsDataset(lit_dataset.Dataset):
    """Bọc file predictions_<name>.csv thành LIT Dataset với giá trị RAW thực tế."""

    def __init__(self, df: pd.DataFrame, feature_cols: list[str], pos: str, neg: str, col_specs: dict):
        self.feature_cols = feature_cols
        self.pos, self.neg = pos, neg
        self.col_specs = col_specs
        self._examples = [
            {**{c: float(row[c]) for c in feature_cols},
             "label": pos if int(row["true_label"]) == 1 else neg}
            for _, row in df.iterrows()
        ]

    def spec(self) -> lit_types.Spec:
        d = {}
        for c in self.feature_cols:
            spec_info = self.col_specs.get(c, {})
            d[c] = lit_types.Scalar(
                min_val=spec_info.get("min_val", 0.0),
                max_val=spec_info.get("max_val", 100.0),
                default=spec_info.get("default", 50.0),
                step=spec_info.get("step", 1.0)
            )
        d["label"] = lit_types.CategoryLabel(vocab=[self.neg, self.pos])
        return d


class ReplayModel(lit_model.BatchedModel):
    """Mô hình phản hồi kết quả suy luận AL-QNN theo thời gian thực (Inference Replay)."""

    def __init__(self, df: pd.DataFrame, feature_cols: list[str], pos: str, neg: str, col_specs: dict):
        self._df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.pos, self.neg = pos, neg
        self.col_specs = col_specs
        
        # Chuẩn hóa ma trận tham chiếu để tính khoảng cách Euclidean trên không gian chuẩn hóa
        X_mat = self._df[self.feature_cols].to_numpy(dtype=float)
        mins = X_mat.min(axis=0)
        ranges = np.where((X_mat.max(axis=0) - mins) == 0, 1.0, (X_mat.max(axis=0) - mins))
        self._X_norm = (X_mat - mins) / ranges
        self._mins = mins
        self._ranges = ranges

    def input_spec(self) -> lit_types.Spec:
        d = {}
        for c in self.feature_cols:
            spec_info = self.col_specs.get(c, {})
            d[c] = lit_types.Scalar(
                min_val=spec_info.get("min_val", 0.0),
                max_val=spec_info.get("max_val", 100.0),
                default=spec_info.get("default", 50.0),
                step=spec_info.get("step", 1.0)
            )
        return d

    def output_spec(self) -> lit_types.Spec:
        return {"probas": lit_types.MulticlassPreds(
            vocab=[self.neg, self.pos], null_idx=0, parent="label", threshold=0.5)}

    def predict_minibatch(self, inputs):
        outputs = []
        for ex in inputs:
            # Chuẩn hóa vector input theo cùng scale để tìm mẫu gần nhất
            x_vec = np.array([float(ex[c]) for c in self.feature_cols])
            x_norm = (x_vec - self._mins) / self._ranges
            
            dists = np.sum((self._X_norm - x_norm) ** 2, axis=1)
            best_i = int(np.argmin(dists))
            p1 = float(self._df.iloc[best_i]["proba_class1"])
            outputs.append({"probas": [round(1 - p1, 4), round(p1, 4)]})
        return outputs


def main():
    available = {k: v for k, v in DATASETS_CFG.items() if (ROOT / v["csv"]).exists()}
    if not available:
        raise SystemExit("Lỗi: Không tìm thấy predictions_*.csv. Hãy chạy export_predictions.py trước.")

    requested = sys.argv[1] if len(sys.argv) > 1 else None
    if requested is None:
        requested = "predictive_maintenance" if "predictive_maintenance" in available else next(iter(available))

    if requested not in available:
        raise SystemExit(f"Dataset '{requested}' không hợp lệ. Chọn một trong: {', '.join(available)}")

    cfg = available[requested]
    csv_path = ROOT / cfg["csv"]
    df = pd.read_csv(csv_path)
    
    # Lấy các cột feature thực tế (loại bỏ cột nhãn)
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    model_key = cfg.get("key", requested)
    
    # Tính toán min, max, mean thực tế cho từng ô nhập liệu
    col_specs = {}
    for c in feature_cols:
        col_data = df[c]
        min_v = float(col_data.min())
        max_v = float(col_data.max())
        mean_v = float(col_data.mean())
        is_int = (col_data % 1 == 0).all()
        col_specs[c] = {
            "min_val": round(min_v, 1 if not is_int else 0),
            "max_val": round(max_v, 1 if not is_int else 0),
            "default": round(mean_v, 1 if not is_int else 0),
            "step": 1.0 if is_int else 0.1
        }

    datasets = {model_key: PredictionsDataset(df, feature_cols, cfg["pos"], cfg["neg"], col_specs)}
    models = {model_key: ReplayModel(df, feature_cols, cfg["pos"], cfg["neg"], col_specs)}
    
    print("\n=======================================================")
    print(f"  [LIT DEMO] Đang tải: {cfg['friendly']} ({model_key})")
    print(f"  [Dữ liệu] {len(df)} mẫu thực tế | {len(feature_cols)} biến số:")
    for col in feature_cols:
        spec = col_specs[col]
        print(f"    - {col} (Min: {spec['min_val']}, Max: {spec['max_val']}, Gợi ý: {spec['default']})")
    print(f"  [Máy chủ Web] Đang chạy tại -> http://localhost:{cfg['port']}")
    print("=======================================================\n")

    CLIENT_ROOT = str(Path(lit_nlp.__file__).parent / "client" / "build" / "default")
    lit_demo = dev_server.Server(
        models, datasets,
        port=cfg["port"],
        client_root=CLIENT_ROOT,
        default_layout="simple"
    )
    lit_demo.serve()


if __name__ == "__main__":
    main()
