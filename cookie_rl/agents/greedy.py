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
    A_TOGGLE_CLICK,
    A_TOGGLE_POP,
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

    if o["clicksPerSec"] == 0:
        return A_TOGGLE_CLICK
    if not o["autoPop"]:
        return A_TOGGLE_POP
    if mask[A_LUMP]:
        return A_LUMP

    gain = o["prestigePotential"] - o["prestige"]
    if mask[A_ASCEND] and gain >= max(ascend_min_chips, ascend_gain_factor * o["prestige"]):
        return A_ASCEND

    bank = o["cookies"]
    cps = o["cookiesPs"]

    # cheapest affordable non-blacklisted upgrade
    best_upg, best_upg_price = -1, float("inf")
    for k, u in enumerate(o["upgrades"][:N_UPGRADE_SLOTS]):
        if u["name"] in UPGRADE_BLACKLIST:
            continue
        if u["price"] <= bank and u["price"] < best_upg_price:
            best_upg, best_upg_price = k, u["price"]
    if best_upg >= 0:
        return A_UPGRADE0 + best_upg

    # payback period over unlocked buildings (waiting for a better one is allowed)
    best_pp, best_id, best_affordable = float("inf"), -1, False
    for b in o["buildings"]:
        if b["locked"] and b["amount"] == 0 and b["price"] > bank * 10:
            continue
        dcps = max(b["unitCps"], 1e-12)
        wait = max(b["price"] - bank, 0.0) / cps if cps > 0 else (0.0 if b["price"] <= bank else float("inf"))
        pp = wait + b["price"] / dcps
        if pp < best_pp:
            best_pp, best_id, best_affordable = pp, b["id"], b["price"] <= bank
    if best_id >= 0 and best_affordable:
        return A_BUILDING0 + best_id
    return A_NOOP  # saving up for the argmin building


def random_policy(raw_obs: dict, mask: np.ndarray, rng: np.random.Generator) -> int:
    return int(rng.choice(np.flatnonzero(mask)))
