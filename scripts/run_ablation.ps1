# ============================================================
#  run_ablation.ps1 — so sánh CÓ / KHÔNG telemetry (patch C)
#
#  Chạy cùng cấu hình 2 lần: baseline (không telemetry) vs telemetry.
#  Chỉ local n=12 (nơi có depth-BP, telemetry có đất diễn nhất).
#
#  Chạy:  powershell -ExecutionPolicy Bypass -File .\scripts\run_ablation.ps1
#
#  ĐIỀU KIỆN: đã áp patch_telemetry_C.py vào rl_env.py và thêm --use-telemetry
#  vào 04_train_rl_scheduler.py. Chạy test khớp trước (xem cuối file).
# ============================================================
param(
  [int]$Nqubits    = 12,
  [string]$Cost    = "local",
  [int]$MaxDepth   = 12,
  [string]$Surrogate = "outputs\surrogate\q12_local_d16\surrogate_fit.json",
  [string]$Seeds   = "0,1,2",
  [int]$DeployEpochs = 40
)
$ErrorActionPreference = "Stop"
$log = "run_ablation_q${Nqubits}_${Cost}.log"
function Log($m){ $l="[{0}] {1}" -f (Get-Date -Format HH:mm:ss),$m; Write-Host $l -ForegroundColor Cyan; Add-Content $log $l }

Log "===== ABLATION telemetry: q$Nqubits $Cost ====="

# ---- nhánh A: KHÔNG telemetry (baseline) ----
Log "1/2: baseline (khong telemetry)"
python scripts\04_train_rl_scheduler.py --rl-path RL\PPO --agent ppo --episodes 600 `
  --backend simulate --deploy-backend pennylane --dataset predictive_maintenance `
  --max-samples 200 --n-qubits $Nqubits --cost-type $Cost --initial-depth 2 --max-depth $MaxDepth `
  --surrogate $Surrogate --deploy-epochs $DeployEpochs --seeds $Seeds `
  --results-dir "outputs\rl_results\simulate\ppo_q${Nqubits}_${Cost}_baseline" 2>&1 | Tee-Object -Append $log

# ---- nhánh B: CÓ telemetry ----
Log "2/2: telemetry ON"
python scripts\04_train_rl_scheduler.py --rl-path RL\PPO --agent ppo --episodes 600 `
  --backend simulate --deploy-backend pennylane --dataset predictive_maintenance `
  --max-samples 200 --n-qubits $Nqubits --cost-type $Cost --initial-depth 2 --max-depth $MaxDepth `
  --surrogate $Surrogate --deploy-epochs $DeployEpochs --seeds $Seeds --use-telemetry `
  --results-dir "outputs\rl_results\simulate\ppo_q${Nqubits}_${Cost}_telemetry" 2>&1 | Tee-Object -Append $log

Log "===== XONG. So 2 aggregate.json: ====="
Log "  baseline : outputs\rl_results\simulate\ppo_q${Nqubits}_${Cost}_baseline\aggregate.json"
Log "  telemetry: outputs\rl_results\simulate\ppo_q${Nqubits}_${Cost}_telemetry\aggregate.json"
Log "So acc_end, f1_end, depth_end (std). Neu telemetry giup: depth on dinh hon / it overshoot hon."

# ============================================================
# TEST KHỚP TRƯỚC (BẮT BUỘC chạy trước ablation):
#   Chạy baseline với --use-telemetry TẮT, so 1 seed với kết quả cũ.
#   Phải khớp tới chữ số. Nếu lệch -> patch sai, DỪNG.
#
#   python scripts\04_train_rl_scheduler.py ... --n-qubits 6 --cost-type local `
#     --deploy-epochs 5 --seeds 123 --results-dir outputs\check_notele
#   (so với comparison_6qbit_local.csv seed 123)
# ============================================================
