# run_demo_pd.ps1
$env:PYTHONPATH = "."
$env:MUJOCO_GL = "glfw"
cd g1_dual_arm_grasping_rl
python scripts/demo_pd_close_hand.py --xml models/g1_dual_arm.xml --secs 8 --fist
