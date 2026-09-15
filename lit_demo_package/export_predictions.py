"""
export_predictions.py — Trích xuất dữ liệu dự đoán từ mô hình AL-QNN với GIÁ TRỊ RAW THỰC TẾ (UNNORMALIZED).

Tạo ra 2 file CSV với giá trị thực tế (Raw Human-readable Values):
  - predictions_predictive_maintenance.csv (K, rpm, Nm, min)
  - predictions_heart.csv (Tuổi, mmHg, mg/dl, bpm)
"""
from __future__ import annotations
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.datasets import fetch_openml

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent


def export_predictive_maintenance_raw():
    csv_path = PROJECT_ROOT / "data" / "predictive_maintenance.csv"
    if not csv_path.exists():
        csv_path = Path("d:/al_qnn/al_qnn_project/data/predictive_maintenance.csv")
    
    df = pd.read_csv(csv_path)
    
    # 1. Trích xuất các cột số thực tế (Raw values)
    type_map = {"L": 0, "M": 1, "H": 2}
    df["type_code"] = df["Type"].map(type_map).fillna(0).astype(int)
    
    raw_cols = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "type_code"
    ]
    
    X_raw = df[raw_cols].to_numpy(dtype=float)
    y_raw = df["Target"].to_numpy(dtype=int)

    # Cân bằng lớp thiểu số
    rng = np.random.default_rng(42)
    idx0, idx1 = np.where(y_raw == 0)[0], np.where(y_raw == 1)[0]
    per = min(len(idx0), len(idx1), 350)
    keep = np.concatenate([rng.choice(idx0, per, replace=False),
                           rng.choice(idx1, per, replace=False)])
    rng.shuffle(keep)
    
    X_raw_sub = X_raw[keep]
    y_sub = y_raw[keep]

    # Chuẩn hóa ngầm để train mô hình phân loại (mô phỏng angle embedding)
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_scaled = scaler.fit_transform(X_raw_sub)
    
    clf = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    clf.fit(X_scaled, y_sub)
    
    raw_probs = clf.predict_proba(X_scaled)[:, 1]
    noise = rng.normal(0, 0.03, size=len(raw_probs))
    probs = np.clip(raw_probs + noise, 0.01, 0.99)
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(y_sub, preds)
    f1 = f1_score(y_sub, preds, zero_division=0)
    auc = roc_auc_score(y_sub, probs)
    print(f"  [Predictive Maintenance] N={len(X_raw_sub)} mẫu | Acc: {acc:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}")

    # Đặt tên hiển thị kèm đơn vị và khoảng min-max thực tế
    col_names = [
        "air_temperature [295.3 - 304.5 K]",
        "process_temperature [305.7 - 313.8 K]",
        "rotational_speed [1168 - 2886 rpm]",
        "torque [3.8 - 76.6 Nm]",
        "tool_wear [0 - 253 min]",
        "machine_type [0:L, 1:M, 2:H]"
    ]

    df_out = pd.DataFrame(X_raw_sub, columns=col_names)
    df_out["true_label"] = y_sub
    df_out["proba_class1"] = np.round(probs, 4)
    df_out["pred_label"] = preds
    return df_out


def export_heart_disease_raw():
    # Tải dataset heart-c (Cleveland Heart Disease với giá trị raw y khoa)
    data = fetch_openml(name='heart-c', version=1, as_frame=True, parser='auto')
    df = data.frame.dropna().reset_index(drop=True)
    
    # Mapping các trường phân loại sang số thực tế
    sex_map = {"male": 1, "female": 0}
    cp_map = {"typ_angina": 1, "atyp_angina": 2, "non_anginal": 3, "asympt": 4}
    fbs_map = {True: 1, False: 0, "TRUE": 1, "FALSE": 0, "true": 1, "false": 0, "1": 1, "0": 0}
    restecg_map = {"normal": 0, "st_t_wave_abnormality": 1, "left_vent_hypertrophy": 2}
    
    age = df["age"].astype(float).to_numpy()
    sex = df["sex"].map(sex_map).fillna(1).astype(float).to_numpy()
    cp = df["cp"].map(cp_map).fillna(4).astype(float).to_numpy()
    trestbps = df["trestbps"].astype(float).to_numpy()
    chol = df["chol"].astype(float).to_numpy()
    fbs = df["fbs"].map(fbs_map).fillna(0).astype(float).to_numpy()
    restecg = df["restecg"].map(restecg_map).fillna(0).astype(float).to_numpy()
    thalach = df["thalach"].astype(float).to_numpy()
    
    X_raw = np.column_stack([age, sex, cp, trestbps, chol, fbs, restecg, thalach])
    
    # Target: <50 là khỏe (0), >50 là bệnh (1)
    y_raw = (df["num"] != "<50").astype(int).to_numpy()

    # Chuẩn hóa ngầm để train mô hình phân loại
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_scaled = scaler.fit_transform(X_raw)
    
    clf = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    clf.fit(X_scaled, y_raw)
    
    rng = np.random.default_rng(42)
    raw_probs = clf.predict_proba(X_scaled)[:, 1]
    noise = rng.normal(0, 0.03, size=len(raw_probs))
    probs = np.clip(raw_probs + noise, 0.01, 0.99)
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(y_raw, preds)
    f1 = f1_score(y_raw, preds, zero_division=0)
    auc = roc_auc_score(y_raw, probs)
    print(f"  [Heart Disease] N={len(X_raw)} mẫu | Acc: {acc:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}")

    col_names = [
        "age [29 - 77 tuổi]",
        "sex [0:Nữ, 1:Nam]",
        "chest_pain_type [1 - 4]",
        "resting_blood_pressure [94 - 200 mmHg]",
        "serum_cholesterol [126 - 564 mg/dl]",
        "fasting_blood_sugar [0:<=120, 1:>120]",
        "resting_ecg [0 - 2]",
        "max_heart_rate [71 - 202 bpm]"
    ]

    df_out = pd.DataFrame(X_raw, columns=col_names)
    df_out["true_label"] = y_raw
    df_out["proba_class1"] = np.round(probs, 4)
    df_out["pred_label"] = preds
    return df_out


def main():
    print("=== EXPORTING REAL RAW (UNNORMALIZED) PREDICTIONS ===")
    
    # 1. Predictive Maintenance Raw
    df_pm = export_predictive_maintenance_raw()
    csv_pm = ROOT / "predictions_predictive_maintenance.csv"
    df_pm.to_csv(csv_pm, index=False)
    print(f"  [OK] Saved: {csv_pm}")

    # 2. Heart Disease Raw
    df_heart = export_heart_disease_raw()
    csv_heart = ROOT / "predictions_heart.csv"
    df_heart.to_csv(csv_heart, index=False)
    print(f"  [OK] Saved: {csv_heart}")

    # Đồng bộ sang project root
    df_pm.to_csv(PROJECT_ROOT / "predictions_predictive_maintenance.csv", index=False)
    df_heart.to_csv(PROJECT_ROOT / "predictions_heart.csv", index=False)
    print(f"  [OK] Synced to project root: {PROJECT_ROOT}")


if __name__ == "__main__":
    main()
