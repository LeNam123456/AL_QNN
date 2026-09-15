"""
src/synthetic_bp/synthetic_data.py — Bộ sinh dữ liệu tổng hợp Teacher-Student cho AL-QNN.

Tạo dữ liệu nhị phân dựa trên một mạch Teacher QNN có độ sâu biết trước D_teacher.
Sử dụng PennyLane với thiết kế mạch Z-Y-Z Euler decomposition & Entanglement block tương thích hoàn toàn với Stage 1A ansatz.
"""

from __future__ import annotations
import numpy as np
import pennylane as qml
from sklearn.model_selection import train_test_split
from src.synthetic_bp.stage1b.encoding import apply_data_encoding
from src.synthetic_bp.stage1a.ansatz import apply_hea

class TeacherStudentGenerator:
    """
    Tạo dữ liệu nhị phân thông qua Teacher QNN Circuit.
    
    Args:
        n_qubits: Số qubit (mặc định 8)
        teacher_depth: Độ sâu D_teacher của Teacher Circuit (mặc định 4)
        topology: Kiến trúc vướng víu ('linear', 'circular', 'full', ...)
        entangler: Loại cổng vướng víu ('cnot', 'cz')
        noise_std: Độ nhiễu Gauss bổ sung (nếu có, mặc định 0.0)
        seed: Seed khởi tạo ngẫu nhiên
    """
    def __init__(self, n_qubits: int = 8, teacher_depth: int = 4,
                 topology: str = "circular", entangler: str = "cnot",
                 noise_std: float = 0.0, seed: int = 42):
        self.n_qubits = n_qubits
        self.teacher_depth = teacher_depth
        self.topology = topology
        self.entangler = entangler
        self.noise_std = noise_std
        self.seed = seed
        
        rng = np.random.default_rng(seed)
        # Khởi tạo trọng số ngẫu nhiên cố định cho Teacher
        self.teacher_params = rng.uniform(0, 2 * np.pi, size=(teacher_depth, n_qubits, 3))
        
        # PennyLane Simulator device cho Teacher
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        @qml.qnode(self.dev)
        def _teacher_circuit(x, params):
            apply_data_encoding(x, n_qubits, encoding_type="angle")
            apply_hea(params, n_qubits, teacher_depth,
                      entanglement_topology=topology, entangler_type=entangler)
            # Đo kỳ vọng PauliZ trên qubit 0
            return qml.expval(qml.PauliZ(0))
            
        self._circuit = _teacher_circuit

    def generate(self, n_samples: int = 500) -> tuple[np.ndarray, np.ndarray]:
        """
        Sinh n_samples mẫu (X, y).
        
        Returns:
            X: Ma trận đặc trưng shape (n_samples, n_qubits) nằm trong [0, pi]
            y: Nhãn nhị phân {0, 1}
        """
        rng = np.random.default_rng(self.seed)
        # Sinh X ngẫu nhiên đều trong [0, pi]
        X = rng.uniform(0, np.pi, size=(n_samples, self.n_qubits))
        
        expvals = np.zeros(n_samples)
        for i in range(n_samples):
            expvals[i] = self._circuit(X[i], self.teacher_params)
            
        if self.noise_std > 0:
            expvals += rng.normal(0, self.noise_std, size=n_samples)
            
        # Threshold tại 0: expval > 0 -> 1, expval <= 0 -> 0
        y = np.where(expvals > 0, 1, 0).astype(int)
        
        # Đảm bảo cân bằng nhãn nếu số lượng quá lệch
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)
        print(f"[TeacherStudentGenerator] Sinh {n_samples} mẫu (D_teacher={self.teacher_depth}): "
              f"Lớp 1={n_pos}, Lớp 0={n_neg}")
        
        return X, y


def generate_teacher_student_dataset(n_qubits: int = 8, teacher_depth: int = 4,
                                     n_samples: int = 500, topology: str = "circular",
                                     val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 42) -> dict:
    """
    Hàm adapter được gọi trực tiếp bởi dataset.py load_binary_dataset().
    """
    gen = TeacherStudentGenerator(n_qubits=n_qubits, teacher_depth=teacher_depth,
                                  topology=topology, seed=seed)
    X, y = gen.generate(n_samples=n_samples)
    
    if test_frac > 0:
        val_test_frac = val_frac + test_frac
        Xtr, X_temp, ytr, y_temp = train_test_split(
            X, y, test_size=val_test_frac, random_state=seed, stratify=y
        )
        rel_test_frac = test_frac / val_test_frac
        Xva, Xte, yva, yte = train_test_split(
            X_temp, y_temp, test_size=rel_test_frac, random_state=seed, stratify=y_temp
        )
    else:
        Xtr, Xva, ytr, yva = train_test_split(
            X, y, test_size=val_frac, random_state=seed, stratify=y
        )
        Xte, yte = np.empty((0, X.shape[1])), np.empty((0,))
    
    res = {
        "X_train": Xtr.astype(float), "y_train": ytr.astype(float),
        "X_val": Xva.astype(float), "y_val": yva.astype(float),
        "name": f"synthetic_ts_d{teacher_depth}", "n_qubits": n_qubits,
        "teacher_depth": teacher_depth,
        "n_train": int(len(Xtr)), "n_val": int(len(Xva)),
    }
    
    if test_frac > 0:
        res["X_test"] = Xte.astype(float)
        res["y_test"] = yte.astype(float)
        res["n_test"] = int(len(Xte))
        
    return res
