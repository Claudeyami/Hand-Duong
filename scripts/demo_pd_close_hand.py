
import argparse, time, numpy as np
from envs.g1_dual_arm_env import DualArmGraspEnv

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", type=str, default="models/g1_dual_arm.xml")
    ap.add_argument("--secs", type=float, default=6.0)
    ap.add_argument("--fist", action="store_true")
    args = ap.parse_args()

    env = DualArmGraspEnv(xml_path=args.xml, render_mode="human")
    obs, _ = env.reset()

    a = np.zeros(env.action_space.shape[0], dtype=np.float32)
    if args.fist:
        a[-2:] = 1.0  # left_fist, right_fist

    t_end = time.time() + float(args.secs)
    while time.time() < t_end:
        obs, rew, term, trunc, info = env.step(a)
        if term or trunc:
            env.reset()
    env.close()
