"""Deterministically evaluate one PPO checkpoint (true objective, not shaped).

Prints one line: <steps> log10_baked=.. cps=.. asc=.. | top actions
Used both standalone and by the checkpoint watcher during training.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cookie_rl.gym_env import A_ASCEND, A_LUMP, A_UPGRADE0, CookieClickerEnv


def action_name(a: int) -> str:
    if a == 0:
        return "noop"
    if a < A_UPGRADE0:
        return f"bld{a - 1}"
    if a < A_ASCEND:
        return f"upg{a - A_UPGRADE0}"
    if a == A_ASCEND:
        return "ascend"
    return "lump"


def evaluate(model_path: str, horizon_days: float, seed: int) -> dict:
    from sb3_contrib import MaskablePPO

    env = CookieClickerEnv(horizon_days=horizon_days, step_seconds=30, seed=seed)
    model = MaskablePPO.load(model_path, device="cpu")
    acts: Counter = Counter()
    obs, _ = env.reset(seed=seed)
    done = False
    info: dict = {}
    while not done:
        mask = env.action_masks()
        a, _ = model.predict(obs, action_masks=mask, deterministic=True)
        acts[int(a)] += 1
        obs, _, term, trunc, info = env.step(int(a))
        done = term or trunc
    env.close()
    m = re.search(r"_(\d+)_steps", model_path)
    steps = int(m.group(1)) if m else -1
    return {
        "steps": steps,
        "log10_baked": info["log10_baked"],
        "cps": info["cps"],
        "ascensions": info["ascensions"],
        "actions": [(action_name(a), c) for a, c in acts.most_common(6)],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--horizon-days", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=2000)
    args = ap.parse_args()
    r = evaluate(args.model, args.horizon_days, args.seed)
    print(
        f"{r['steps']:>9} log10_baked={r['log10_baked']:.2f} "
        f"cps={r['cps']:.3g} asc={r['ascensions']} | {r['actions']}"
    )


if __name__ == "__main__":
    main()
