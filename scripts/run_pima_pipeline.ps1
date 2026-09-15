$ErrorActionPreference = "Stop"

$PYTHON = "D:\Test\env_prj\Scripts\python.exe"

Write-Host "============================================================"
Write-Host "    AL-QNN Pipeline Runner - Pima Indians Diabetes 8-Qubit"
Write-Host "============================================================"
Write-Host "Note: Stage 1A (Random BP) is dataset-independent, so we reuse the results from bcw_8q."

Write-Host "`n[1/3] Running Stage 1B (Task-Aware BP)..."
& $PYTHON .\scripts\01b_run_task_aware_bp.py --config .\configs\stage1b_pima_8q.yaml

Write-Host "`n[2/3] Running Stage 1C (Calibration Builder)..."
& $PYTHON .\scripts\01c_build_calibration.py --config .\configs\stage1c_pima_8q.yaml

Write-Host "`n[3/3] Running Stage 5 (Comparison - AL-QNN vs Greedy)..."
# We run comparison with Pennylane backend to get the actual results
& $PYTHON .\scripts\05_compare_schedulers.py `
    --backend pennylane `
    --dataset pima_diabetes `
    --n-qubits 8 `
    --max-depth 8 `
    --epochs 60 `
    --calibration outputs/stage1c/calibration_rules_pima_8q.json `
    --out outputs/comparison/pima_8q_pennylane `
    --log-telemetry

Write-Host "`n============================================================"
Write-Host "Pipeline Completed Successfully!"
Write-Host "Check outputs/comparison/pima_8q_pennylane for the final comparison logs."
Write-Host "============================================================"
