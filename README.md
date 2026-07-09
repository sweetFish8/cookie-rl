# cookie-rl

Cookie Clicker を「最高効率」でプレイするAI。**本家のゲームコード(v2.058)をヘッドレスChromiumで実時間の約2,000〜3,000倍に加速**して動かし、その上で MaskablePPO(強化学習)と payback-period 貪欲ベースラインを比較する。

## 仕組み

- `vendor/cookieclicker/` — 本家コードのミラー([ozh/cookieclicker](https://github.com/ozh/cookieclicker))。プロプライエタリなのでコミットしない(各自 `scripts/setup_game.sh` で取得)
- `cookie_rl/harness.js` — ページに注入。仮想時計(`Date.now`/`performance.now` をフレームと同期して進める)、シード付きRNG、`step(n)`(`Game.Logic()` をn回 = n/30ゲーム秒)、購入・転生・観測API
  - `Game.fps` はいじらない(仕様が壊れる)。`Game.Logic()` の直接ループが忠実な加速手段
  - シュガーランプ等の wall-clock 依存系は仮想時計の同期で正しく動く
- `cookie_rl/browser_env.py` — Playwright ラッパ。ローカルHTTPサーバでゲームを配信、外部リクエストは遮断
- `cookie_rl/gym_env.py` — Gymnasium 環境。1ステップ=30ゲーム秒。Discrete(33)+action mask。報酬は `Δlog10(累計焼成)`(= エピソードリターンが最終 log10 焼成数に一致)
- `cookie_rl/agents/greedy.py` — CookieMonster流 payback period 貪欲ベースライン
- `cookie_rl/train.py` — MaskablePPO 学習(カリキュラム: 0.5日→1日→2日)

## セットアップ

```sh
./scripts/setup_game.sh          # ゲーム本体を vendor/ に取得
uv sync
uv run playwright install chromium
```

## 使い方

```sh
uv run python scripts/bench_throughput.py   # 加速性能の実測 (~60-88k frames/s)
uv run python scripts/test_fidelity.py      # env忠実性テスト (11項目)
uv run python -m cookie_rl.train --horizon-days 0.5 --timesteps 1000000 --run-name h05
uv run python eval/compare.py --horizon-days 0.5 --seeds 3 --model checkpoints/h05_final.zip
uv run tensorboard --logdir runs            # ep_rew_mean = log10(累計焼成)
```

## 行動空間

| action | 内容 |
|---|---|
| 0 | noop(時間を進めるだけ) |
| 1–20 | 建物 0–19 を1つ購入 |
| 21–28 | ストア内アップグレード(価格順スロット0–7)購入 |
| 29 | オートクリック切替(10クリック/秒) |
| 30 | ゴールデンクッキー自動ポップ切替 |
| 31 | 転生(天国アップグレードは安い順に自動購入) |
| 32 | シュガーランプ収穫 |

## 実測値 (M-series Mac, 8コア)

- スループット: 62k〜88k Logic frames/s(1ゲーム日 ≈ 30〜40秒)
- 忠実性: CpS会計誤差0.00%、仮想時計ドリフト0ms、GC出現28回/3h(理論12〜30)
- ベースライン(0.5ゲーム日): 貪欲 1e13.4、ランダム 1e5.4
