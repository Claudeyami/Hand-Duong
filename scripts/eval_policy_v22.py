# scripts/eval_policy_v22.py  (FIXED)
import argparse, numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.g1_dual_arm_env_v22 import make_env

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--xml", type=str, default="models/g1_dual_arm.xml")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--venv-stats", type=str, required=True)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--use-priv", action="store_true")
    p.add_argument("--render", action="store_true")
    p.add_argument("--deterministic", action="store_true")
    return p.parse_args()

def main():
    args = parse()

    def _mk():
        return make_env(xml_path=args.xml,
                        render=args.render,
                        use_privileged=args.use_priv,
                        max_steps=args.max_steps)

    # SB3 VecEnv API → 4-tuple (obs, reward, done, infos)
    venv_raw = DummyVecEnv([_mk])
    venv = VecNormalize.load(args.venv_stats, venv_raw)
    venv.training = False
    venv.norm_reward = False

    model = PPO.load(args.ckpt)

    rets, succs, holds = [], [], []
    for ep in range(args.episodes):
        obs = venv.reset()
        ep_ret, steps = 0.0, 0

        while True:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, rew, done, infos = venv.step(action)      # ← 4 outputs
            # với 1 env: rew là mảng (1,), done là mảng (1,), infos là list[dict]
            ep_ret += float(rew[0])
            steps  += 1

            info = infos[0] if isinstance(infos, (list, tuple)) else infos
            if bool(done[0]):  # episode kết thúc (terminated hoặc truncated)
                s = int(info.get("success", 0))
                h = int(info.get("hold", 0))
                print(f"[EP {ep:02d}] ret={ep_ret:.2f} | len={steps} | success={s} | hold={h} | curr_level={info.get('curr_level',0):.2f}")
                rets.append(ep_ret); succs.append(s); holds.append(h)
                break

    print("\n========== EVAL SUMMARY ==========")
    print(f"Episodes           : {len(rets)}")
    print(f"Mean reward        : {np.mean(rets):.2f} ± {np.std(rets):.2f}")
    print(f"Success rate       : {100.0*sum(succs)/len(succs):.1f}%")
    print(f"Hold (stable) rate : {100.0*sum(holds)/len(holds):.1f}%")
    print(f"VecNormalize used  : True")

if __name__ == "__main__":
    main()
