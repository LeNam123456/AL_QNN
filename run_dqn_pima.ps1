$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$python = "D:\Test\env_prj\Scripts\python.exe"

Write-Host "Running RL Scheduler Training (DQN)..."
& $python scripts/04_train_rl_scheduler.py --rl-path RL\DQN --agent dqn --episodes 800 --backend simulate --use-telemetry --deploy-backend pennylane --dataset pima_diabetes --n-qubits 8 --cost-type local --initial-depth 2 --max-depth 16 --surrogate outputs/surrogate/pima_8q/surrogate_fit.json --deploy-epochs 40 --calibration outputs/stage1c/calibration_rules_pima_8q.json --results-dir outputs/rl_results/pima_8q/dqn --seeds 0
if ($LASTEXITCODE -ne 0) { throw "Stage 2 DQN failed" }

Write-Host "Running Schedulers Comparison (All 4)..."
& $python scripts/05_compare_schedulers.py --backend pennylane --dataset pima_diabetes --n-qubits 8 --max-depth 16 --cost-type local --epochs 40 --ppo-ckpt outputs/rl_results/pima_8q/ppo/agent.pt --dqn-ckpt outputs/rl_results/pima_8q/dqn/agent.pt --out outputs/comparison/pima_8q_pennylane --log-telemetry --telemetry-guard
if ($LASTEXITCODE -ne 0) { throw "Stage 4/5 Compare failed" }
