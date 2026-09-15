$ErrorActionPreference = "Stop"

$PYTHON = "D:\Test\env_prj\Scripts\python.exe"

Write-Host "Running Stage 1B (Task-Aware BP) in parallel..."
Start-Process -FilePath $PYTHON -ArgumentList ".\scripts\01b_run_task_aware_bp.py --config .\configs\stage1b_pima_8q.yaml"
Start-Process -FilePath $PYTHON -ArgumentList ".\scripts\01b_run_task_aware_bp.py --config .\configs\stage1b_heart_8q.yaml"

# We must wait for them to finish, but Start-Process without -Wait runs asynchronously.
# Actually, using Start-Job is better for waiting in powershell without blocking entirely, 
# or we can just run sequentially to avoid overloading the Windows machine.
# Since it's fast enough, let's just run them sequentially to guarantee stable telemetry logs.

Write-Host "Running PIMA pipeline..."
& $PYTHON .\scripts\01b_run_task_aware_bp.py --config .\configs\stage1b_pima_8q.yaml
& $PYTHON .\scripts\01c_build_calibration.py --config .\configs\stage1c_pima_8q.yaml
& $PYTHON .\scripts\05_compare_schedulers.py --backend pennylane --device-name lightning.qubit --dataset pima_diabetes --n-qubits 8 --max-depth 8 --epochs 60 --calibration outputs/stage1c/calibration_rules_pima_8q.json --out outputs/comparison/pima_8q_pennylane --log-telemetry

Write-Host "Running HEART DISEASE pipeline..."
& $PYTHON .\scripts\01b_run_task_aware_bp.py --config .\configs\stage1b_heart_8q.yaml
& $PYTHON .\scripts\01c_build_calibration.py --config .\configs\stage1c_heart_8q.yaml
& $PYTHON .\scripts\05_compare_schedulers.py --backend pennylane --device-name lightning.qubit --dataset heart_disease --n-qubits 8 --max-depth 8 --epochs 60 --calibration outputs/stage1c/calibration_rules_heart_8q.json --out outputs/comparison/heart_8q_pennylane --log-telemetry

Write-Host "All datasets processed successfully."
