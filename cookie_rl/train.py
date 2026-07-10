"""Train MaskablePPO on the headless Cookie Clicker env.

Episode return == log10(final total baked), so `rollout/ep_rew_mean` in
TensorBoard is directly comparable with eval/compare.py numbers.

Curriculum (run sequentially, each loading the previous checkpoint):
  uv run python -m cookie_rl.train --horizon-days 0.5 --timesteps 1000000 --run-name h05
  uv run python -m cookie_rl.train --horizon-days 1 --timesteps 1000000 --run-name h1 --load checkpoints/h05_final.zip
  uv run python -m cookie_rl.train --horizon-days 2 --timesteps 1000000 --run-name h2 --load checkpoints/h1_final.zip
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from cookie_rl.gym_env import CookieClickerEnv, mask_fn


def make_env(rank: int, seed: int, horizon_days: float, step_seconds: float):
    def _init():
        env = CookieClickerEnv(
            horizon_days=horizon_days,
            step_seconds=step_seconds,
            seed=seed + rank * 100_000,
        )
        return ActionMasker(env, mask_fn)

    return _init


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--horizon-days", type=float, default=0.5)
    ap.add_argument("--step-seconds", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--load", type=str, default=None, help="checkpoint .zip to resume from")
    ap.add_argument("--ent-coef", type=float, default=0.02, help="entropy bonus (lower => sharper argmax)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--run-name", type=str, default="ppo")
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    # leave CPU cores for the browser processes (each env = chromium + node + python)
    torch.set_num_threads(2)

    Path("checkpoints").mkdir(exist_ok=True)

    vec = SubprocVecEnv(
        [make_env(i, args.seed, args.horizon_days, args.step_seconds) for i in range(args.envs)],
        start_method="spawn",
    )
    vec = VecMonitor(vec)

    if args.load:
        model = MaskablePPO.load(args.load, env=vec, device=args.device, tensorboard_log="runs")
        model.ent_coef = args.ent_coef  # allow sharpening the argmax when fine-tuning
        model.learning_rate = args.lr
        print(f"resumed from {args.load} (ent_coef={args.ent_coef}, lr={args.lr})")
    else:
        model = MaskablePPO(
            "MlpPolicy",
            vec,
            n_steps=256,
            batch_size=1024,
            gamma=0.999,
            gae_lambda=0.98,
            ent_coef=args.ent_coef,
            learning_rate=args.lr,
            policy_kwargs=dict(net_arch=[256, 256]),
            tensorboard_log="runs",
            seed=args.seed,
            verbose=1,
            device=args.device,
        )

    ckpt = CheckpointCallback(
        save_freq=max(100_000 // args.envs, 1000),
        save_path="checkpoints",
        name_prefix=args.run_name,
    )
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=ckpt,
            tb_log_name=args.run_name,
            reset_num_timesteps=not bool(args.load),
        )
    finally:
        model.save(f"checkpoints/{args.run_name}_final")
        vec.close()
    print(f"saved checkpoints/{args.run_name}_final.zip")


if __name__ == "__main__":
    main()
