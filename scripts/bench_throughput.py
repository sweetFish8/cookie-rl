"""Phase 0 benchmark: how many Game.Logic() frames/sec can we run headless?

Measures three game states (throughput degrades as the save grows):
  1. fresh save
  2. midgame  (10 building types, ~50 each, some upgrades)
  3. lategame (all 20 building types, ~300 each)

Decision gate from the plan:
  >=5k fps  -> proceed as designed
  1k-5k fps -> stub DOM touches inside Game.Logic and re-measure
  <1k fps   -> shorten horizon / more parallel envs
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cookie_rl.browser_env import CookieBrowser

BATCH_FRAMES = 30_000  # 1000 game-seconds per evaluate call
ROUNDS = 3
GAME_DAY_FRAMES = 30 * 86_400  # 2.592M


def measure(env: CookieBrowser, label: str) -> float:
    # warmup
    env.step(3_000)
    best = 0.0
    for i in range(ROUNDS):
        t0 = time.perf_counter()
        env.step(BATCH_FRAMES)
        dt = time.perf_counter() - t0
        fps = BATCH_FRAMES / dt
        best = max(best, fps)
        print(f"  [{label}] round {i + 1}: {fps:,.0f} frames/s ({dt:.2f}s per {BATCH_FRAMES:,} frames)")
    speedup = best / 30
    day_sec = GAME_DAY_FRAMES / best
    print(f"  [{label}] best: {best:,.0f} f/s = {speedup:,.0f}x realtime, 1 game-day in {day_sec:,.0f}s")
    return best


def main() -> None:
    print("launching headless Cookie Clicker...")
    with CookieBrowser(seed=42) as env:
        obs = env.observe()
        print(f"Game ready. T={obs['t']}, cookies={obs['cookies']:.1f}, version ok\n")

        # round-trip overhead of a no-op evaluate
        t0 = time.perf_counter()
        for _ in range(50):
            env.evaluate("1")
        rt = (time.perf_counter() - t0) / 50 * 1000
        print(f"evaluate() round-trip: {rt:.2f} ms\n")

        results: dict[str, float] = {}

        print("--- state 1: fresh save ---")
        results["fresh"] = measure(env, "fresh")

        print("\n--- state 2: midgame (cheat-grant cookies, 10 buildings x50) ---")
        env.evaluate("Game.Earn(1e30)")
        env.evaluate(
            "for (let i=0;i<10;i++){ Game.ObjectsById[i].buy(50); }"
            "for (let k=0;k<15;k++){ if(Game.UpgradesInStore[0]) Game.UpgradesInStore[0].buy(1); }"
        )
        obs = env.observe()
        n_buildings = sum(b["amount"] for b in obs["buildings"])
        print(f"  buildings={n_buildings}, cps={obs['cookiesPs']:.3g}")
        results["midgame"] = measure(env, "midgame")

        print("\n--- state 3: lategame (all 20 buildings x300) ---")
        env.evaluate("Game.Earn(1e300)")
        env.evaluate("for (let i=0;i<Game.ObjectsById.length;i++){ Game.ObjectsById[i].buy(300); }")
        obs = env.observe()
        n_buildings = sum(b["amount"] for b in obs["buildings"])
        print(f"  buildings={n_buildings}, cps={obs['cookiesPs']:.3g}")
        results["lategame"] = measure(env, "lategame")

        print("\n=== summary ===")
        for k, v in results.items():
            print(f"  {k:10s}: {v:,.0f} frames/s ({v / 30:,.0f}x realtime)")
        worst = min(results.values())
        if worst >= 5_000:
            print("gate: PASS (>=5k f/s) — proceed as designed")
        elif worst >= 1_000:
            print("gate: MARGINAL (1k-5k f/s) — consider stubbing DOM touches in Game.Logic")
        else:
            print("gate: FAIL (<1k f/s) — shorten horizon / add parallel envs")


if __name__ == "__main__":
    main()
