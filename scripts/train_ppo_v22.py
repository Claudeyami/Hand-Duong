# scripts/train_ppo_v22.py
import os, argparse, torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from envs.g1_dual_arm_env_v22 import make_env

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--xml", type=str, default="models/g1_dual_arm.xml")
    p.add_argument("--run-name", type=str, required=True)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--total-steps", type=int, default=300000)
    p.add_argument("--n-env", type=int, default=3)
    p.add_argument("--rollout", type=int, default=1024)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--use-priv", action="store_true")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--save-every", type=int, default=100000)
    p.add_argument("--eval-every", type=int, default=100000)
    p.add_argument("--reset-vf", action="store_true", help="reinit value net on resume")
    return p.parse_args()

def make_vec(n_env, xml, use_priv, max_steps):
    def _thunk(seed):
        return lambda: make_env(xml_path=xml, render=False, use_privileged=use_priv, max_steps=max_steps, seed=seed)
    env_fns = [_thunk(1234+i) for i in range(max(1, n_env))]
    venv_raw = DummyVecEnv(env_fns) if n_env==1 else SubprocVecEnv(env_fns)
    venv = VecNormalize(venv_raw, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.995)
    return venv

def reinit_value_net(model: PPO):
    with torch.no_grad():
        for m in model.policy.mlp_extractor.value_net.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight, gain=1.0)
                torch.nn.init.zeros_(m.bias)

def main():
    args = parse()
    os.makedirs(f"runs/{args.run_name}", exist_ok=True)
    logdir = f"runs/{args.run_name}"

    venv = make_vec(args.n_env, args.xml, args.use_priv, args.max_steps)

    policy_kwargs = dict(activation_fn=torch.nn.Tanh, net_arch=[dict(pi=[256,256], vf=[256,256])])
    model = PPO("MlpPolicy", venv, n_steps=args.rollout, batch_size= args.rollout*max(1,args.n_env)//4,
                learning_rate=2.5e-4, gamma=0.995, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0,
                verbose=1, device=args.device, policy_kwargs=policy_kwargs)

    if args.resume:
        print(f"[RESUME] Loading PPO from {args.resume}")
        model = PPO.load(args.resume, env=venv, device=args.device, print_system_info=False)
        if args.reset_vf:
            print("[RESUME] Reset value network")
            reinit_value_net(model)

    ckpt_cb = CheckpointCallback(
        save_freq=max(1, args.save_every)//max(1,args.n_env),
        save_path=logdir, name_prefix="ppo_ckpt",
        save_replay_buffer=False, save_vecnormalize=False
    )

    model.learn(total_timesteps=args.total_steps, callback=ckpt_cb)
    model.save(os.path.join(logdir, "ppo_latest.zip"))
    venv.save(os.path.join(logdir, "vecnorm_latest.pkl"))
    venv.close()

if __name__ == "__main__":
    main()
