"""Environment fidelity checks (plan Phase 1 verification).

1. CpS accounting: cookies earned over N frames == CpS * N/30
2. Virtual clock: in-page Date.now advances in lockstep with frames
3. Golden cookies: spawn at a plausible rate and can be popped (goldenClicks)
4. Ascension: prestige earned, heavenly upgrades bought, CpS multiplier applied
5. Reset: wipes everything
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cookie_rl.browser_env import CookieBrowser

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    status = "PASS" if cond else "FAIL"
    if cond:
        PASS += 1
    else:
        FAIL += 1
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))


def main() -> None:
    with CookieBrowser(seed=123) as env:
        # --- 1. CpS accounting -------------------------------------------------
        env.evaluate("Game.Earn(1e6)")
        env.evaluate("Game.ObjectsById[1].buy(10)")  # 10 grandmas
        env.step(30)  # settle recalc
        o0 = env.observe()
        frames = 9_000  # 300 game-seconds
        o1 = env.step(frames)
        expected = o0["cookiesPs"] * frames / 30
        earned = o1["cookiesEarned"] - o0["cookiesEarned"]
        rel_err = abs(earned - expected) / max(expected, 1e-9)
        check(
            "CpS accounting (300 game-s, 10 grandmas)",
            rel_err < 0.01,
            f"earned={earned:.2f} expected={expected:.2f} rel_err={rel_err:.2%}",
        )

        # --- 2. virtual clock lockstep -----------------------------------------
        t0 = env.evaluate("Date.now()")
        env.step(3_000)
        t1 = env.evaluate("Date.now()")
        drift = abs((t1 - t0) - 3_000 * 1000 / 30)
        check("virtual clock lockstep (3000 frames = 100s)", drift < 50, f"drift={drift:.0f}ms")

        # --- 3. golden cookies over 3 game-hours -------------------------------
        env.set_state(auto_pop=True)
        gc0 = env.evaluate("Game.goldenClicks")
        env.step(30 * 3600 * 3)  # 3 game-hours
        gc1 = env.evaluate("Game.goldenClicks")
        pops = gc1 - gc0
        # spawn interval 5-15 min => expect roughly 12-30 pops in 3h
        check("golden cookies spawn & pop (3 game-h)", 5 <= pops <= 60, f"pops={pops}")
        env.set_state(auto_pop=False)

        # --- 4. ascension -------------------------------------------------------
        env.evaluate("Game.Earn(1e18)")  # ~100 heavenly chips potential
        o = env.observe()
        check("prestige potential before ascend", o["prestigePotential"] >= 99, f"potential={o['prestigePotential']:.1f}")
        mult_before = o["globalCpsMult"]
        ok = env.ascend()
        o = env.step(30)
        check("ascend executed", bool(ok) and o["ascensions"] == 1, f"ascensions={o['ascensions']}")
        check("prestige earned", o["prestige"] >= 99, f"prestige={o['prestige']:.0f}")
        heavenly_bought = env.evaluate(
            "Game.PrestigeUpgrades.filter(u=>u.bought).map(u=>u.name)"
        )
        check("heavenly upgrades bought", len(heavenly_bought) >= 2, f"{heavenly_bought}")
        check(
            "prestige CpS multiplier active",
            o["globalCpsMult"] > mult_before,
            f"mult {mult_before:.2f} -> {o['globalCpsMult']:.2f}",
        )
        check("run cookies wiped after ascend", o["cookies"] < 1e6, f"bank={o['cookies']:.3g}")
        check("totalBaked preserved across ascension", o["totalBaked"] >= 1e18, f"baked={o['totalBaked']:.3g}")

        # --- 5. hard reset -------------------------------------------------------
        o = env.reset_game(seed=999)
        check(
            "hard reset wipes save",
            o["totalBaked"] == 0 and o["prestige"] == 0 and o["cookies"] == 0,
            f"baked={o['totalBaked']} prestige={o['prestige']}",
        )

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
