# QUICKSTART — G1 Dual‑Arm Grasping RL (v22)

> Hướng dẫn chạy **đánh giá (inference)**, **train**, và **demo PD** cho bộ source bạn đã gửi.
> Hệ điều hành mẫu: **Windows 10/11 + PowerShell**, Python **3.10/3.11**.

## 0) Cài môi trường

```powershell
# 0.1. Tạo venv/conda (tuỳ bạn)
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # (hoặc conda activate g1hands)

# 0.2. Cài thư viện
pip install -r g1_dual_arm_grasping_rl/requirements.txt
# nếu gặp lỗi GPU, có thể dùng CPU: pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**`requirements.txt`** (rút gọn): `mujoco>=3.2.5`, `gymnasium>=0.29.1`, `stable-baselines3>=2.3.2`, `torch>=2.3`, `matplotlib>=3.8`

> **Lưu ý MuJoCo Viewer:** đặt biến môi trường để mở viewer:
```powershell
$env:MUJOCO_GL = "glfw"   # (Windows)
$env:PYTHONPATH = "."      # để import envs/*
```

## 1) Cấu trúc & file quan trọng
- `envs/g1_dual_arm_env_v22.py` : môi trường RL (MuJoCo), có `make_env(...)`, `DualArmGraspEnv`.
- `models/g1_dual_arm.xml`      : mô hình MJCF dual‑arm + bàn tay.
- `scripts/train_ppo_v22.py`    : train PPO (VecNormalize).
- `scripts/eval_policy_v22.py`  : đánh giá checkpoint (cần cả `vecnorm`).
- `scripts/demo_pd_close_hand.py`: **demo PD** (baseline), *cần sửa 1 dòng import* (xem mục 4).

Mặc định **VecNormalize** được dùng trong train → sẽ lưu **2 file** trong thư mục `runs/<run-name>/`:
- `ppo_latest.zip` (hoặc các `ppo_ckpt_*.zip`)
- `vecnorm_latest.pkl`           (thống kê chuẩn hoá quan sát/Reward)

## 2) Đánh giá (dùng model đã train xong)
Giả sử bạn có thư mục `runs/<run-name>/` (ví dụ `runs/ppo_dualarm_v22`) chứa:
- `ppo_latest.zip`
- `vecnorm_latest.pkl`

Chạy lệnh:

```powershell
cd g1_dual_arm_grasping_rl
$env:PYTHONPATH = "."
$env:MUJOCO_GL = "glfw"

python scripts/eval_policy_v22.py `
  --xml models/g1_dual_arm.xml `
  --ckpt runs/<run-name>/ppo_latest.zip `
  --venv-stats runs/<run-name>/vecnorm_latest.pkl `
  --episodes 5 `
  --max-steps 800 `
  --render `
  --deterministic
```
Ghi chú:
- Nếu bạn muốn dùng checkpoint khác: đổi `--ckpt` sang `runs/<run-name>/ppo_ckpt_***.zip`.
- Nếu không mở được viewer, bỏ `--render` hoặc kiểm tra `$env:MUJOCO_GL`.

## 3) Train (từ đầu hoặc resume)
```powershell
cd g1_dual_arm_grasping_rl
$env:PYTHONPATH = "."

python scripts/train_ppo_v22.py `
  --xml models/g1_dual_arm.xml `
  --run-name ppo_dualarm_v22 `
  --n-env 4 `
  --rollout 1024 `
  --total-steps 3000000 `
  --max-steps 400 `
  --device cuda      # hoặc cpu
```
- **Resume** (load checkpoint cũ, tuỳ chọn reset value‑fn):
```powershell
python scripts/train_ppo_v22.py `
  --xml models/g1_dual_arm.xml `
  --run-name ppo_dualarm_v22_resume `
  --resume runs/ppo_dualarm_v22/ppo_latest.zip `
  --reset-vf
```

## 4) Demo PD baseline (đóng nắm tay)
File `scripts/demo_pd_close_hand.py` đang `import envs.g1_dual_arm_env` nhưng repo chỉ có `g1_dual_arm_env_v22.py`.
**Sửa 1 dòng** trong `scripts/demo_pd_close_hand.py`:
```diff
- from envs.g1_dual_arm_env import DualArmGraspEnv
+ from envs.g1_dual_arm_env_v22 import DualArmGraspEnv
```
Chạy demo:
```powershell
cd g1_dual_arm_grasping_rl
$env:PYTHONPATH = "."
$env:MUJOCO_GL = "glfw"
python scripts/demo_pd_close_hand.py --xml models/g1_dual_arm.xml --secs 8 --fist
```

## 5) Tham số thường dùng (train/eval)
- `--xml`        : đường dẫn mô hình MJCF.
- `--run-name`   : tên thư mục log/ckpt lưu ở `runs/<run-name>/`.
- `--n-env`      : số env song song (>=2 sẽ dùng `SubprocVecEnv`).
- `--rollout`    : số bước/worker mỗi lần thu thập.
- `--max-steps`  : số bước tối đa/episode (env sẽ `truncate` sau ngưỡng).
- `--use-priv`   : dùng quan sát “đặc quyền” (GT contact/poses) **trong sim** để critic/obs ổn định hơn.
- `--device`     : `cuda`/`cpu`.
- `--save-every` : tần suất lưu ckpt (theo timesteps).
- **Eval** cần thêm: `--ckpt`, `--venv-stats`, `--deterministic`, `--render`.

## 6) Lỗi hay gặp & cách xử lý nhanh
- **GLFW/Viewer lỗi**: đặt `$env:MUJOCO_GL="glfw"` (Windows) hoặc `"osmesa"` (headless).
- **Không load được venv stats**: kiểm tra đường dẫn `--venv-stats` đúng file `vecnorm_latest.pkl` tương ứng cùng run.
- **SB3 version mismatch**: đảm bảo `stable-baselines3>=2.3.2`. Nếu ckpt tạo bằng bản khác, nâng/hạ version tương ứng.
- **CUDA lỗi**: chạy `--device cpu` hoặc cài đúng `torch` phù hợp GPU/driver.

## 7) Gợi ý kiểm thử nhanh (sanity checks)
1. Chạy **demo PD** (mục 4) để chắc viewer + MJCF ok.
2. `eval_policy_v22.py` với ckpt bạn đã train (mục 2).
3. `--episodes 1 --max-steps 200` để xem log/obs hoạt động trước khi chạy dài.

---

**Made for you on 2025-11-12T14:52:28**.
