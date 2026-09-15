$ErrorActionPreference = "Stop"
$PYTHON = "D:\Test\env_prj\Scripts\python.exe"

Write-Host "================= PIMA DIABETES ================="
Write-Host "1. Running Stage 1B..."
& $PYTHON .\scripts\01b_run_task_aware_bp.py --config .\configs\stage1b_pima_8q.yaml
Write-Host "2. Running Stage 1C..."
& $PYTHON .\scripts\01c_build_calibration.py --config .\configs\stage1c_pima_8q.yaml
Write-Host "3. Running Stage 5 (Simulate - 800 Epochs)..."
& $PYTHON .\scripts\05_compare_schedulers.py --backend simulate --dataset pima_diabetes --n-qubits 8 --max-depth 8 --topology linear --epochs 800 --calibration outputs/stage1c/calibration_rules_pima_8q.json --out outputs/comparison/pima_8q_simulate --log-telemetry
Write-Host "4. Running Stage 5 (Pennylane CPU - 40 Epochs)..."
& $PYTHON .\scripts\05_compare_schedulers.py --backend pennylane --device-name lightning.qubit --dataset pima_diabetes --n-qubits 8 --max-depth 8 --topology linear --epochs 40 --calibration outputs/stage1c/calibration_rules_pima_8q.json --out outputs/comparison/pima_8q_pennylane --log-telemetry

Write-Host "`n================= HEART DISEASE ================="
Write-Host "1. Running Stage 1B..."
& $PYTHON .\scripts\01b_run_task_aware_bp.py --config .\configs\stage1b_heart_8q.yaml
Write-Host "2. Running Stage 1C..."
& $PYTHON .\scripts\01c_build_calibration.py --config .\configs\stage1c_heart_8q.yaml
Write-Host "3. Running Stage 5 (Simulate - 800 Epochs)..."
& $PYTHON .\scripts\05_compare_schedulers.py --backend simulate --dataset heart_disease --n-qubits 8 --max-depth 16 --topology circular --epochs 800 --calibration outputs/stage1c/calibration_rules_heart_8q.json --out outputs/comparison/heart_8q_simulate --log-telemetry
Write-Host "4. Running Stage 5 (Pennylane CPU - 40 Epochs)..."
& $PYTHON .\scripts\05_compare_schedulers.py --backend pennylane --device-name lightning.qubit --dataset heart_disease --n-qubits 8 --max-depth 16 --topology circular --epochs 40 --calibration outputs/stage1c/calibration_rules_heart_8q.json --out outputs/comparison/heart_8q_pennylane --log-telemetry

Write-Host "`nDONE!"
