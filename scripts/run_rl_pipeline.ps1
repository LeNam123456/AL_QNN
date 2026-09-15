$ErrorActionPreference = "Stop"
$PYTHON = "D:\Test\env_prj\Scripts\python.exe"

$ppo_pima_dir = (Get-ChildItem -Path "outputs\rl_results\simulate" -Filter "ppo_*" | Sort-Object LastWriteTime -Descending | Select-Object -Skip 1 -First 1).FullName
$dqn_pima_dir = (Get-ChildItem -Path "outputs\rl_results\simulate" -Filter "dqn_*" | Sort-Object LastWriteTime -Descending | Select-Object -Skip 1 -First 1).FullName

# Wait, the latest ones might be Heart Disease, and the ones before that are Pima.
# Since it ran Pima then Heart, the Heart ones are the absolute latest. Pima ones are the 2nd latest.
# Let's just find them by filtering files in those directories if we can't be sure, OR we can train them again for a few seconds. Actually 600 episodes takes ~1 min, maybe I'll just find them correctly.

$ppo_dirs = Get-ChildItem -Path "outputs\rl_results\simulate" -Filter "ppo_*" | Sort-Object LastWriteTime -Descending
$dqn_dirs = Get-ChildItem -Path "outputs\rl_results\simulate" -Filter "dqn_*" | Sort-Object LastWriteTime -Descending

# Pima was trained first, Heart was trained second.
$ppo_heart_dir = $ppo_dirs[0].FullName
$ppo_pima_dir = $ppo_dirs[1].FullName

$dqn_heart_dir = $dqn_dirs[0].FullName
$dqn_pima_dir = $dqn_dirs[1].FullName

$ppo_pima_ckpt = Join-Path $ppo_pima_dir "agent.pt"
$dqn_pima_ckpt = Join-Path $dqn_pima_dir "agent.pt"
$ppo_heart_ckpt = Join-Path $ppo_heart_dir "agent.pt"
$dqn_heart_ckpt = Join-Path $dqn_heart_dir "agent.pt"

Write-Host "3. Running Stage 5 Compare Schedulers for PIMA (including RL)..."
& $PYTHON .\scripts\05_compare_schedulers.py --backend simulate --dataset pima_diabetes --n-qubits 8 --max-depth 8 --topology linear --epochs 800 --calibration outputs/stage1c/calibration_rules_pima_8q.json --out outputs/comparison/pima_8q_simulate_rl --ppo-ckpt $ppo_pima_ckpt --dqn-ckpt $dqn_pima_ckpt --rl-root .\RL --log-telemetry

Write-Host "6. Running Stage 5 Compare Schedulers for HEART DISEASE (including RL)..."
& $PYTHON .\scripts\05_compare_schedulers.py --backend simulate --dataset heart_disease --n-qubits 8 --max-depth 16 --topology circular --epochs 800 --calibration outputs/stage1c/calibration_rules_heart_8q.json --out outputs/comparison/heart_8q_simulate_rl --ppo-ckpt $ppo_heart_ckpt --dqn-ckpt $dqn_heart_ckpt --rl-root .\RL --log-telemetry

Write-Host "DONE RL COMPARISON!"
