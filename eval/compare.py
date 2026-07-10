"""Compare agents (random / greedy / trained PPO) on seed-matched episodes.

Usage:
  uv run python eval/compare.py --horizon-days 0.5 --seeds 3
  uv run python eval/compare.py --horizon-days 2 --seeds 3 --model checkpoints/ppo_latest.zip
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cookie_rl.agents.greedy import greedy_policy, random_policy
from cookie_rl.gym_env import CookieClickerEnv


def run_episode(env: CookieClickerEnv, policy, seed: int) -> dict:
    obs, _ = env.reset(seed=seed)
    done = False
    curve = []  # (game_seconds, log10 baked)
    while not done:
        mask = env.action_masks()
        action = policy(obs, env.raw_obs, mask)
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
        if env._steps % 20 == 0 or done:
            curve.append((env._steps * env.step_frames / 30, info["log10_baked"]))
    return {
        "final_baked": info["total_baked"],
        "log10_baked": info["log10_baked"],
        "ascensions": info["ascensions"],
        "curve": curve,
    }


def make_policies(models: dict[str, str]) -> dict:
    rng = np.random.default_rng(0)
    policies = {
        "random": lambda obs, raw, mask: random_policy(raw, mask, rng),
        "greedy": lambda obs, raw, mask: greedy_policy(raw, mask),
    }
    from sb3_contrib import MaskablePPO

    for name, path in models.items():
        model = MaskablePPO.load(path, device="cpu")

        def ppo_policy(obs, raw, mask, _m=model):
            action, _ = _m.predict(obs, action_masks=mask, deterministic=True)
            return int(action)

        policies[name] = ppo_policy
    return policies


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon-days", type=float, default=0.5)
    ap.add_argument("--step-seconds", type=float, default=30.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed-base", type=int, default=2000, help="held-out eval seeds (train used 0-3, BC 5000+)")
    ap.add_argument("--models", type=str, default="", help="comma list name=path.zip")
    ap.add_argument("--out", type=str, default="results")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    models = dict(kv.split("=", 1) for kv in args.models.split(",") if kv.strip())
    policies = make_policies(models)

    results: dict[str, list[dict]] = {name: [] for name in policies}
    env = CookieClickerEnv(horizon_days=args.horizon_days, step_seconds=args.step_seconds)
    try:
        for name, policy in policies.items():
            for s in range(args.seeds):
                t0 = time.perf_counter()
                r = run_episode(env, policy, seed=args.seed_base + s)
                dt = time.perf_counter() - t0
                results[name].append(r)
                print(
                    f"{name:10s} seed={args.seed_base + s}: baked=1e{r['log10_baked']:.2f} "
                    f"ascensions={r['ascensions']} ({dt:.0f}s wall)"
                )
    finally:
        env.close()

    summary = {
        name: {
            "mean_log10_baked": float(np.mean([r["log10_baked"] for r in rs])),
            "std_log10_baked": float(np.std([r["log10_baked"] for r in rs])),
            "mean_ascensions": float(np.mean([r["ascensions"] for r in rs])),
        }
        for name, rs in results.items()
    }
    print("\n=== summary (log10 total baked, higher is better) ===")
    for name, s in summary.items():
        print(f"  {name:8s}: 1e{s['mean_log10_baked']:.2f} ± {s['std_log10_baked']:.2f} "
              f"(ascensions {s['mean_ascensions']:.1f})")

    stamp = f"h{args.horizon_days}d"
    (out_dir / f"compare_{stamp}.json").write_text(json.dumps({"summary": summary, "results": results}, indent=1))

    # progression plot
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, rs in results.items():
        for i, r in enumerate(rs):
            xs = [c[0] / 3600 for c in r["curve"]]
            ys = [c[1] for c in r["curve"]]
            ax.plot(xs, ys, label=name if i == 0 else None, alpha=0.7)
    ax.set_xlabel("game time (hours)")
    ax.set_ylabel("log10(total cookies baked)")
    ax.set_title(f"Cookie Clicker agents, horizon={args.horizon_days} game-days")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"compare_{stamp}.png", dpi=120)
    print(f"\nwrote {out_dir}/compare_{stamp}.json and .png")


if __name__ == "__main__":
    main()
