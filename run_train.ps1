# run_train.ps1
param(
  [string]$RunName = "ppo_dualarm_v22",
  [int]$NEnv = 4,
  [int]$Total = 3000000
)
$env:PYTHONPATH = "."
cd g1_dual_arm_grasping_rl
python scripts/train_ppo_v22.py `
  --xml models/g1_dual_arm.xml `
  --run-name $RunName `
  --n-env $NEnv `
  --rollout 1024 `
  --max-steps 400 `
  --total-steps $Total `
  --device cuda
