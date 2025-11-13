# run_eval.ps1
param(
  [string]$RunName = "ppo_dualarm_v22",
  [string]$Ckpt    = "ppo_latest.zip",
  [switch]$Deterministic = $true
)

$env:PYTHONPATH = "."
$env:MUJOCO_GL = "glfw"
cd g1_dual_arm_grasping_rl

$det = $Deterministic.IsPresent ? "--deterministic" : ""
python scripts/eval_policy_v22.py `
  --xml models/g1_dual_arm.xml `
  --ckpt ("runs/{0}/{1}" -f $RunName,$Ckpt) `
  --venv-stats ("runs/{0}/vecnorm_latest.pkl" -f $RunName) `
  --episodes 3 `
  --max-steps 800 `
  --render $det
