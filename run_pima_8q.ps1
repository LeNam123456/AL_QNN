$ErrorActionPreference = "Stop"
$python = "D:\Test\env_prj\Scripts\python.exe"

Write-Host "Creating Stage 1A Config for Pima..."
(Get-Content configs\stage1a_bcw_8q.yaml) -replace "bcw", "pima" | Set-Content configs\stage1a_pima_8q.yaml

Write-Host "Running Stage 1A..."
& $python scripts/01a_run_random_bp.py --config configs/stage1a_pima_8q.yaml
if ($LASTEXITCODE -ne 0) { throw "Stage 1A failed" }

Write-Host "Running Stage 1B..."
& $python scripts/01b_run_task_aware_bp.py --config configs/stage1b_pima_8q.yaml
if ($LASTEXITCODE -ne 0) { throw "Stage 1B failed" }

Write-Host "Running Stage 1C..."
& $python scripts/01c_build_calibration.py --config configs/stage1c_pima_8q.yaml
if ($LASTEXITCODE -ne 0) { throw "Stage 1C failed" }

Write-Host "Running Surrogate Calibration..."
& $python scripts/06_calibrate_surrogate.py --n-qubits 8 --cost-type local --dataset pima_diabetes --depths 1,2,4,6,8,10,12,14,16 --out outputs/surrogate/pima_8q
if ($LASTEXITCODE -ne 0) { throw "Stage 6 failed" }

Write-Host "Running RL Scheduler Training (PPO)..."
& $python scripts/04_train_rl_scheduler.py --agent ppo --episodes 800 --backend simulate --use-telemetry --deploy-backend pennylane --dataset pima_diabetes --n-qubits 8 --cost-type local --initial-depth 2 --max-depth 16 --surrogate outputs/surrogate/pima_8q/surrogate_fit.json --deploy-epochs 40 --calibration outputs/stage1c/calibration_rules_pima_8q.json --results-dir outputs/rl_results/pima_8q/ppo --seeds 0
if ($LASTEXITCODE -ne 0) { throw "Stage 2 PPO failed" }

Write-Host "Running Schedulers Comparison..."
& $python scripts/05_compare_schedulers.py --backend pennylane --dataset pima_diabetes --n-qubits 8 --max-depth 16 --cost-type local --epochs 40 --ppo-ckpt outputs/rl_results/pima_8q/ppo/seed_0/agent.pt --out outputs/comparison/pima_8q_pennylane --log-telemetry --telemetry-guard
if ($LASTEXITCODE -ne 0) { throw "Stage 4/5 Compare failed" }

Write-Host "Pipeline completed successfully!"
