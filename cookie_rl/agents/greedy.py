"""Greedy baseline: CookieMonster-style payback-period policy.

Per step (one action allowed, mirroring the RL agent's constraints):
  1. turn on autoclick / golden-cookie auto-pop if off
  2. harvest ripe sugar lump
  3. ascend when the prestige gain is large enough
  4. buy the cheapest affordable upgrade (minus grandmapocalypse triggers)
  5. buy the building with the lowest payback period
       PP = max(cost - bank, 0)/CpS + cost/dCpS
     if the argmin building is unaffordable, wait (noop) for it.
"""

from __future__ import annotations

import numpy as np

from cookie_rl.gym_env import (
    A_ASCEND,
    A_BUILDING0,
    A_LUMP,
    A_NOOP,
    A_UPGRADE0,
    N_UPGRADE_SLOTS,
)

# leave the grandmapocalypse switches to the RL agent; keep the baseline clean
UPGRADE_BLACKLIST = {
    "One mind",
    "Communal brainsweep",
    "Elder Pact",
    "Elder Pledge",
    "Elder Covenant",
    "Revoke Elder Covenant",
}


def greedy_policy(raw_obs: dict, mask: np.ndarray, ascend_gain_factor: float = 1.0,
                  ascend_min_chips: float = 200.0) -> int:
    o = raw_obs

    # autoclick + golden-cookie popping are automatic (harness); nothing to toggle
    if mask[A_LUMP]:
        return A_LUMP

    gain = o["prestigePotential"] - o["prestige"]
    if mask[A_ASCEND] and gain >= max(ascend_min_chips, ascend_gain_factor * o["prestige"]):
        return A_ASCEND

    bank = o["cookies"]
    cps = o["cookiesPs"]

    # cheapest non-blacklisted upgrade that the mask allows (== affordable)
    best_upg, best_upg_price = -1, float("inf")
    for k, u in enumerate(o["upgrades"][:N_UPGRADE_SLOTS]):
        if u["name"] in UPGRADE_BLACKLIST or not mask[A_UPGRADE0 + k]:
            continue
        if u["price"] < best_upg_price:
            best_upg, best_upg_price = k, u["price"]
    if best_upg >= 0:
        return A_UPGRADE0 + best_upg

    # lowest payback period over all buildings. `locked` is ignored: Object.buy()
    # only checks affordability and the headless store never flips locked to 0.
    # Buy the argmin if affordable; else noop to save up for it.
    best_pp, best_id, best_price = float("inf"), -1, 0.0
    for b in o["buildings"]:
        dcps = max(b["unitCps"], 1e-12)
        wait = max(b["price"] - bank, 0.0) / cps if cps > 0 else (0.0 if b["price"] <= bank else float("inf"))
        pp = wait + b["price"] / dcps
        if pp < best_pp:
            best_pp, best_id, best_price = pp, b["id"], b["price"]
    if best_id >= 0 and best_price <= bank and mask[A_BUILDING0 + best_id]:
        return A_BUILDING0 + best_id
    return A_NOOP  # saving up for the argmin building


def random_policy(raw_obs: dict, mask: np.ndarray, rng: np.random.Generator) -> int:
    return int(rng.choice(np.flatnonzero(mask)))
