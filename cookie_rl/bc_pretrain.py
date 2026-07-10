"""Behavior-cloning warm-start: imitate the greedy payback-period policy.

PPO from scratch collapses to always-noop: at short horizons undirected buying
scores below the autoclick income floor, so exploration teaches "buying is bad".
We instead initialize the policy by supervised imitation of the greedy agent
(which reaches log10~11-13), then fine-tune with PPO (train.py --load).

Steps:
  1. roll out greedy over several seeds, record (obs, action_mask, greedy_action)
  2. supervised cross-entropy: maximize log pi(greedy_action | obs) under masks
  3. save a MaskablePPO checkpoint PPO can resume from
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from cookie_rl.agents.greedy import greedy_policy
from cookie_rl.gym_env import CookieClickerEnv, mask_fn


def collect(env: CookieClickerEnv, episodes: int, base_seed: int):
    obs_buf, mask_buf, act_buf = [], [], []
    for e in range(episodes):
        obs, _ = env.reset(seed=base_seed + e)
        done = False
        while not done:
            mask = env.action_masks()
            a = greedy_policy(env.raw_obs, mask)
            obs_buf.append(obs)
            mask_buf.append(mask)
            act_buf.append(a)
            obs, _, term, trunc, info = env.step(a)
            done = term or trunc
        print(f"  ep {e + 1}/{episodes}: log10_baked={info['log10_baked']:.2f} samples={len(obs_buf)}")
    return (
        np.asarray(obs_buf, dtype=np.float32),
        np.asarray(mask_buf, dtype=bool),
        np.asarray(act_buf, dtype=np.int64),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--horizon-days", type=float, default=0.25)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=str, default="checkpoints/bc_init")
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    Path("checkpoints").mkdir(exist_ok=True)
    torch.set_num_threads(max(1, __import__("os").cpu_count() - 1))

    env = CookieClickerEnv(horizon_days=args.horizon_days, step_seconds=30, seed=5000)
    print(f"collecting {args.episodes} greedy episodes...")
    obs, masks, acts = collect(env, args.episodes, base_seed=5000)

    # a MaskablePPO instance whose policy we will pretrain, then hand to train.py
    masked_env = ActionMasker(env, mask_fn)
    model = MaskablePPO(
        "MlpPolicy",
        masked_env,
        policy_kwargs=dict(net_arch=[256, 256]),
        device=args.device,
        seed=0,
    )
    policy = model.policy
    policy.set_training_mode(True)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    n = len(obs)
    obs_t = torch.as_tensor(obs, device=args.device)
    mask_t = torch.as_tensor(masks, device=args.device)
    act_t = torch.as_tensor(acts, device=args.device)

    print(f"\nBC training on {n} samples, action histogram:")
    uniq, cnt = np.unique(acts, return_counts=True)
    print("  " + ", ".join(f"a{u}:{c}" for u, c in zip(uniq, cnt)))

    for epoch in range(args.epochs):
        perm = torch.randperm(n, device=args.device)
        total_loss, total_acc, nb = 0.0, 0.0, 0
        for i in range(0, n, args.batch_size):
            idx = perm[i : i + args.batch_size]
            dist = policy.get_distribution(obs_t[idx], action_masks=mask_t[idx].cpu().numpy())
            logp = dist.log_prob(act_t[idx])
            loss = -logp.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                pred = dist.distribution.probs.argmax(dim=-1)
                total_acc += (pred == act_t[idx]).float().sum().item()
            total_loss += loss.item() * len(idx)
            nb += len(idx)
        print(f"epoch {epoch + 1:>2}: loss={total_loss / nb:.4f} imitation_acc={total_acc / nb:.3f}")

    model.save(args.out)
    print(f"\nsaved {args.out}.zip")
    env.close()


if __name__ == "__main__":
    main()
