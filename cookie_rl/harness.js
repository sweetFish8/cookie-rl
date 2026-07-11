// Cookie Clicker RL harness — injected before page load (add_init_script).
// Provides: virtual clock (Date/performance mock), seeded RNG, and window.__cc
// with step/observe/act APIs driven from Python via page.evaluate.
//
// Design notes (verified against main.js v2.058):
// - 1 Game.Logic() call == 1 frame == 1/30 game-second. All economy, buffs and
//   shimmer spawns are frame-based, so looping Logic() is a faithful fast-forward.
// - Sugar lumps / seasons / offline calc use Date.now(), so the virtual clock
//   advances +1000/fps ms per frame, in lockstep. Must be monotonic
//   (Game.lumpT=Math.min(Date.now(),Game.lumpT) punishes time travel).
// - Game.fps must NOT be changed (it is the frames-per-game-second conversion
//   factor used in buff durations, earnings, shimmer lifetimes).
(() => {
	if (window.__cc) return;
	const CFG = window.__CC_CONFIG || {};
	const START = CFG.startTimeMs || Date.UTC(2026, 1, 2, 12, 0, 0); // Feb 2: no season
	const SEED = (CFG.seed === undefined) ? 1 : CFG.seed;

	// ---- virtual clock -------------------------------------------------------
	let offsetMs = 0;
	const RealDate = Date;
	const vnow = () => Math.floor(START + offsetMs);
	class MockDate extends RealDate {
		constructor(...args) {
			if (args.length === 0) super(vnow());
			else super(...args);
		}
		static now() { return vnow(); }
	}
	window.Date = MockDate;
	const perfProto = Object.getPrototypeOf(performance);
	try { perfProto.now = () => offsetMs; } catch (e) { performance.now = () => offsetMs; }

	// ---- seeded RNG (mulberry32) --------------------------------------------
	let rngState = (SEED >>> 0) || 1;
	Math.random = function () {
		rngState |= 0; rngState = (rngState + 0x6D2B79F5) | 0;
		let t = Math.imul(rngState ^ (rngState >>> 15), 1 | rngState);
		t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};

	// ---- language: skip first-launch friction --------------------------------
	try { localStorage.setItem('CookieClickerLang', 'EN'); } catch (e) {}

	const G = () => window.Game;

	// Autoclick and golden-cookie popping are always optimal in Cookie Clicker,
	// so they are ON by default rather than agent-controlled toggles. This removes
	// the start-of-game exploration barrier (0 cookies + 0 CpS => the agent must
	// otherwise discover "toggle autoclick" before any reward exists), which was
	// collapsing the policy to always-noop.
	const state = {
		clicksPerSec: 10,       // big-cookie autoclick rate during step()
		autoPop: true,          // pop golden (non-wrath) shimmers during step()
		popWrath: false,        // also pop wrath cookies
		burstClicksPerFrame: 30, // clicks/frame while a Click-frenzy/Dragonflight buff is up
		initDone: false,
	};

	// true while a click-power buff (Click frenzy ×777, Dragonflight) is active —
	// each ClickCookie() is then worth `multClick`× more, so bursting the big cookie
	// during these ~13s windows is the single biggest active golden-cookie play.
	function clickBuffActive() {
		const buffs = G().buffs;
		for (const k in buffs) if ((buffs[k].multClick || 1) > 1) return true;
		return false;
	}

	function popShimmers() {
		const Game = G();
		// iterate backwards: pop() splices Game.shimmers
		for (let j = Game.shimmers.length - 1; j >= 0; j--) {
			const s = Game.shimmers[j];
			if (s.type === 'golden' && s.wrath && !state.popWrath) continue;
			try { s.pop(); } catch (e) {}
		}
	}

	function frame() {
		offsetMs += 1000 / G().fps;
		G().Logic();
	}

	window.__cc = {
		state,
		now: vnow,

		// one-time setup after Game.ready
		init: () => {
			const Game = G();
			if (state.initDone || !Game || !Game.ready) return false;
			Game.Loop = function () {};        // pending setTimeout becomes a no-op; loop dies
			Game.timedout = false;
			const p = Game.prefs;
			p.particles = 0; p.numbers = 0; p.autosave = 0; p.autoupdate = 0;
			p.wobbly = 0; p.cursors = 0; p.milk = 0; p.fancy = 0; p.animate = 0;
			p.timeout = 0; p.notifs = 1; p.showBackupWarning = 0;
			Game.volume = 0; Game.volumeMusic = 0;
			Game.Notify = function () {};      // UI-only; avoids unbounded DOM growth
			// Logic() reads Game.bounds=Game.l.getBounds() every frame (forced layout);
			// freeze it — only used for pointer/shimmer positioning, not economy.
			const frozen = Game.l.getBounds();
			Game.l.getBounds = () => frozen;
			Game.CloseNotes();
			state.initDone = true;
			return true;
		},

		// advance n frames; honors clicksPerSec/autoPop set via setState
		step: (n) => {
			const Game = G();
			const fps = Game.fps;
			const clickEvery = state.clicksPerSec > 0 ? Math.max(1, Math.round(fps / state.clicksPerSec)) : 0;
			for (let i = 0; i < n; i++) {
				if (state.autoPop && Game.shimmers.length) popShimmers();
				if (state.burstClicksPerFrame > 0 && clickBuffActive()) {
					for (let c = 0; c < state.burstClicksPerFrame; c++) Game.ClickCookie();
				} else if (clickEvery && (Game.T % clickEvery === 0)) {
					Game.ClickCookie();
				}
				frame();
			}
			return window.__cc.observe();
		},

		setState: (s) => { Object.assign(state, s); },

		// full wipe (buildings, upgrades, achievements, prestige, lumps) + reseed RNG.
		// Much cheaper than reloading the page; virtual clock keeps running forward
		// (must stay monotonic for sugar lumps).
		reset: (seed) => {
			const Game = G();
			if (Game.OnAscend) Game.Reincarnate(1);
			Game.HardReset(2);
			if (seed !== undefined) rngState = (seed >>> 0) || 1;
			state.clicksPerSec = 10;
			state.autoPop = true;
			state.popWrath = false;
			state.burstClicksPerFrame = 30;
			Game.Notify = function () {};
			frame(); // settle one frame so CpS/store are recalculated
			return window.__cc.observe();
		},

		// ---- actions ----------------------------------------------------------
		buyBuilding: (i, amount) => {
			const o = G().ObjectsById[i];
			if (!o) return false;
			const before = o.amount;
			o.buy(amount || 1);
			return o.amount > before;
		},

		// k-th upgrade in store (Game.UpgradesInStore is price-sorted); buy(1) bypasses confirm prompts
		buyUpgradeSlot: (k) => {
			const u = G().UpgradesInStore[k];
			if (!u) return false;
			return !!u.buy(1);
		},

		popNow: () => { popShimmers(); },

		clickLump: () => { G().clickLump(); },

		// full ascension: intro anim (skipped) -> earn chips -> greedy cheapest-first
		// heavenly purchases -> reincarnate. Runs synchronously; costs ~6 game-seconds
		// of animation frames, matching a real speedy ascension.
		ascend: () => {
			const Game = G();
			if (Game.OnAscend || Game.AscendTimer > 0) return false;
			Game.Ascend(1);
			Game.AscendTimer = Game.AscendDuration; // skip intro animation
			let guard = 0;
			while (!Game.OnAscend && guard++ < 300) frame(); // Logic drives UpdateAscendIntro
			// greedy cheapest-first over purchasable prestige upgrades
			let bought = true, safety = 0;
			while (bought && safety++ < 200) {
				bought = false;
				Game.BuildAscendTree(); // refresh canBePurchased flags
				const candidates = Game.PrestigeUpgrades
					.filter(u => !u.bought && u.canBePurchased && u.getPrice() <= Game.heavenlyChips)
					.sort((a, b) => a.getPrice() - b.getPrice());
				if (candidates.length) bought = !!candidates[0].buy();
			}
			Game.Reincarnate(1);
			guard = 0;
			while (Game.ReincarnateTimer > 0 && guard++ < 100) frame();
			return true;
		},

		// ---- observation --------------------------------------------------------
		observe: () => {
			const Game = G();
			const buildings = Game.ObjectsById.map(o => {
				let unitCps = 0;
				try {
					unitCps = (o.amount > 0 ? o.storedTotalCps / o.amount : o.cps(o)) * Game.globalCpsMult;
				} catch (e) {}
				return {
					id: o.id, amount: o.amount, price: o.getPrice(),
					locked: o.locked ? 1 : 0, unitCps: unitCps,
				};
			});
			const upgrades = Game.UpgradesInStore.slice(0, 12).map(u => ({
				id: u.id, name: u.name, price: u.getPrice(), pool: u.pool,
			}));
			const buffs = Object.values(Game.buffs || {}).map(b => ({
				name: b.name, time: b.time, maxTime: b.maxTime,
				multCpS: b.multCpS === undefined ? 1 : b.multCpS,
				multClick: b.multClick === undefined ? 1 : b.multClick,
			}));
			let shimGold = 0, shimWrath = 0, shimOther = 0;
			for (const s of Game.shimmers) {
				if (s.type === 'golden') { if (s.wrath) shimWrath++; else shimGold++; }
				else shimOther++;
			}
			const totalBaked = Game.cookiesReset + Game.cookiesEarned;
			return {
				t: Game.T,
				timeMs: vnow(),
				cookies: Game.cookies,
				cookiesPs: Game.cookiesPs,
				cookiesEarned: Game.cookiesEarned,
				cookiesReset: Game.cookiesReset,
				totalBaked: totalBaked,
				mouseCps: Game.computedMouseCps,
				globalCpsMult: Game.globalCpsMult,
				prestige: Game.prestige,
				heavenlyChips: Game.heavenlyChips,
				prestigePotential: Game.HowMuchPrestige(Game.cookiesReset + Game.cookiesEarned),
				elderWrath: Game.elderWrath,
				pledges: Game.pledges,
				lumps: Game.lumps,
				lumpAgeMs: Game.lumpsTotal >= 0 && Game.lumpT ? (vnow() - Game.lumpT) : -1,
				lumpMatureAgeMs: Game.lumpMatureAge,
				lumpRipeAgeMs: Game.lumpRipeAge,
				canLumps: Game.canLumps() ? 1 : 0,
				shimmersGold: shimGold, shimmersWrath: shimWrath, shimmersOther: shimOther,
				ascensions: Game.resets,
				onAscend: Game.OnAscend ? 1 : 0,
				buildings: buildings,
				upgrades: upgrades,
				buffs: buffs,
				clicksPerSec: state.clicksPerSec,
				autoPop: state.autoPop ? 1 : 0,
			};
		},
	};
})();
