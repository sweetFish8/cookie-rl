"""Gymnasium environment wrapping the headless Cookie Clicker browser.

MDP design:
- One env step = (apply one discrete action) then advance `step_seconds` of game
  time (default 30 s = 900 logic frames, one page.evaluate round-trip).
- Episode = fixed game-time horizon (default 0.5 game-days), truncated at the end.
- Reward = delta of log10(1 + total cookies baked all-time) per step. Total baked
  (cookiesReset + cookiesEarned) is monotonic across ascensions, so the return is
  well-defined even when the agent resets its run.
- Invalid actions are masked (MaskablePPO / ActionMasker compatible via
  `action_masks()`).

Action space (Discrete(33)):
  0        noop (just let time pass)
  1..20    buy 1 of building id 0..19
  21..28   buy upgrade in store slot 0..7 (price-sorted)
  29       toggle big-cookie autoclick (10 clicks/s while on)
  30       toggle golden-cookie auto-pop
  31       ascend (skip animation, greedy cheapest-first heavenly buys, reincarnate)
  32       harvest ripe sugar lump
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from cookie_rl.browser_env import CookieBrowser

N_BUILDINGS = 20
N_UPGRADE_SLOTS = 8
N_ACTIONS = 1 + N_BUILDINGS + N_UPGRADE_SLOTS + 4  # 33

A_NOOP = 0
A_BUILDING0 = 1
A_UPGRADE0 = 1 + N_BUILDINGS
A_TOGGLE_CLICK = A_UPGRADE0 + N_UPGRADE_SLOTS  # 29
A_TOGGLE_POP = A_TOGGLE_CLICK + 1              # 30
A_ASCEND = A_TOGGLE_POP + 1                    # 31
A_LUMP = A_ASCEND + 1                          # 32

OBS_SCALARS = 22
OBS_DIM = OBS_SCALARS + N_BUILDINGS * 3 + N_UPGRADE_SLOTS * 2  # 98

CLICKS_PER_SEC = 10.0
GAME_FPS = 30


def _l(x: float) -> float:
    """log10 squash for quantities spanning many orders of magnitude."""
    return float(np.log10(1.0 + max(0.0, x)) / 30.0)


class CookieClickerEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        horizon_days: float = 0.5,
        step_seconds: float = 30.0,
        seed: int = 0,
        browser_reset_every: int = 100,
        headless: bool = True,
    ) -> None:
        super().__init__()
        self.horizon_steps = int(round(horizon_days * 86_400 / step_seconds))
        self.step_frames = int(round(step_seconds * GAME_FPS))
        self._base_seed = seed
        self._episode_count = 0
        self._browser_reset_every = browser_reset_every
        self._headless = headless
        self._browser: CookieBrowser | None = None

        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(OBS_DIM,), dtype=np.float32)

        self.raw_obs: dict = {}
        self._steps = 0
        self._last_log_baked = 0.0

    # ---- helpers --------------------------------------------------------------

    def _ensure_browser(self, seed: int) -> None:
        recycle = (
            self._browser is not None
            and self._episode_count % self._browser_reset_every == 0
            and self._episode_count > 0
        )
        if self._browser is None or recycle:
            if self._browser is not None:
                self._browser.close()
            self._browser = CookieBrowser(seed=seed, headless=self._headless)

    def _vectorize(self, o: dict) -> np.ndarray:
        v = np.zeros(OBS_DIM, dtype=np.float32)
        gain = max(0.0, o["prestigePotential"] - o["prestige"])
        buffs = o.get("buffs", [])
        mult = 1.0
        max_frac = 0.0
        for b in buffs:
            mult *= b["multCpS"] if b["multCpS"] > 0 else 1.0
            if b["maxTime"] > 0:
                max_frac = max(max_frac, b["time"] / b["maxTime"])
        ripeness = 0.0
        if o["lumpRipeAgeMs"] > 0 and o["lumpAgeMs"] >= 0:
            ripeness = min(1.5, o["lumpAgeMs"] / o["lumpRipeAgeMs"]) / 1.5

        v[0] = _l(o["cookies"])
        v[1] = _l(o["cookiesPs"])
        v[2] = _l(o["mouseCps"])
        v[3] = _l(o["totalBaked"])
        v[4] = _l(o["cookiesEarned"])
        v[5] = _l(o["prestige"])
        v[6] = _l(o["heavenlyChips"])
        v[7] = _l(gain)
        v[8] = float(np.clip(np.log2(mult) / 5.0, -1.0, 1.0))
        v[9] = max_frac
        v[10] = min(1.0, len(buffs) / 5.0)
        v[11] = 1.0 if o["shimmersGold"] > 0 else 0.0
        v[12] = 1.0 if o["shimmersWrath"] > 0 else 0.0
        v[13] = o["elderWrath"] / 3.0
        v[14] = min(1.0, o["pledges"] / 20.0)
        v[15] = float(o["canLumps"])
        v[16] = min(1.0, o["lumps"] / 100.0)
        v[17] = ripeness
        v[18] = 1.0 if o["clicksPerSec"] > 0 else 0.0
        v[19] = float(o["autoPop"])
        v[20] = self._steps / max(1, self.horizon_steps)
        v[21] = min(1.0, o["ascensions"] / 10.0)

        i = OBS_SCALARS
        for b in o["buildings"][:N_BUILDINGS]:
            v[i] = float(np.log10(1.0 + b["amount"]) / 3.0)
            v[i + 1] = _l(b["price"])
            v[i + 2] = _l(b["unitCps"])
            i += 3
        i = OBS_SCALARS + N_BUILDINGS * 3
        for k in range(N_UPGRADE_SLOTS):
            if k < len(o["upgrades"]):
                v[i] = 1.0
                v[i + 1] = _l(o["upgrades"][k]["price"])
            i += 2
        return v

    def action_masks(self) -> np.ndarray:
        o = self.raw_obs
        m = np.zeros(N_ACTIONS, dtype=bool)
        m[A_NOOP] = True
        bank = o["cookies"]
        for b in o["buildings"][:N_BUILDINGS]:
            m[A_BUILDING0 + b["id"]] = (not b["locked"]) and b["price"] <= bank
        for k in range(min(N_UPGRADE_SLOTS, len(o["upgrades"]))):
            m[A_UPGRADE0 + k] = o["upgrades"][k]["price"] <= bank
        m[A_TOGGLE_CLICK] = True
        m[A_TOGGLE_POP] = True
        m[A_ASCEND] = (o["prestigePotential"] - o["prestige"]) >= 1.0 and not o["onAscend"]
        m[A_LUMP] = bool(o["canLumps"]) and 0 <= o["lumpRipeAgeMs"] <= o["lumpAgeMs"]
        return m

    def _apply_action(self, action: int) -> None:
        assert self._browser is not None
        b = self._browser
        if action == A_NOOP:
            return
        if A_BUILDING0 <= action < A_UPGRADE0:
            b.buy_building(action - A_BUILDING0, 1)
        elif A_UPGRADE0 <= action < A_TOGGLE_CLICK:
            b.buy_upgrade_slot(action - A_UPGRADE0)
        elif action == A_TOGGLE_CLICK:
            on = self.raw_obs["clicksPerSec"] > 0
            b.set_state(clicks_per_sec=0.0 if on else CLICKS_PER_SEC)
        elif action == A_TOGGLE_POP:
            b.set_state(auto_pop=not bool(self.raw_obs["autoPop"]))
        elif action == A_ASCEND:
            b.ascend()
        elif action == A_LUMP:
            b.click_lump()

    # ---- gym API ----------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        ep_seed = (seed if seed is not None else self._base_seed) + self._episode_count * 7919
        self._ensure_browser(ep_seed)
        self._episode_count += 1
        assert self._browser is not None
        self.raw_obs = self._browser.reset_game(ep_seed)
        self._steps = 0
        self._last_log_baked = float(np.log10(1.0 + self.raw_obs["totalBaked"]))
        return self._vectorize(self.raw_obs), {}

    def step(self, action: int):
        assert self._browser is not None
        self._apply_action(int(action))
        self.raw_obs = self._browser.step(self.step_frames)
        self._steps += 1

        log_baked = float(np.log10(1.0 + self.raw_obs["totalBaked"]))
        reward = log_baked - self._last_log_baked
        self._last_log_baked = log_baked

        truncated = self._steps >= self.horizon_steps
        info: dict[str, Any] = {
            "total_baked": self.raw_obs["totalBaked"],
            "cps": self.raw_obs["cookiesPs"],
            "ascensions": self.raw_obs["ascensions"],
            "log10_baked": log_baked,
        }
        return self._vectorize(self.raw_obs), reward, False, truncated, info

    def close(self) -> None:
        if self._browser is not None:
            self._browser.close()
            self._browser = None


def mask_fn(env: CookieClickerEnv) -> np.ndarray:
    return env.action_masks()
