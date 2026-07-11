# cookie-rl

Cookie Clicker を「最高効率」でプレイするAI。**本家のゲームコード(v2.058)をヘッドレスChromiumで実時間の約2,000〜3,000倍に加速**して動かし、その上で MaskablePPO(強化学習)と payback-period 貪欲ベースラインを比較する。

## 仕組み

- `vendor/cookieclicker/` — 本家コードのミラー([ozh/cookieclicker](https://github.com/ozh/cookieclicker))。プロプライエタリなのでコミットしない(各自 `scripts/setup_game.sh` で取得)
- `cookie_rl/harness.js` — ページに注入。仮想時計(`Date.now`/`performance.now` をフレームと同期して進める)、シード付きRNG、`step(n)`(`Game.Logic()` をn回 = n/30ゲーム秒)、購入・転生・観測API
  - `Game.fps` はいじらない(仕様が壊れる)。`Game.Logic()` の直接ループが忠実な加速手段
  - シュガーランプ等の wall-clock 依存系は仮想時計の同期で正しく動く
- `cookie_rl/browser_env.py` — Playwright ラッパ。ローカルHTTPサーバでゲームを配信、外部リクエストは遮断
- `cookie_rl/gym_env.py` — Gymnasium 環境。1ステップ=30ゲーム秒。Discrete(31)+action mask。報酬は `Δlog10(累計焼成)` + ポテンシャルベースのCpSシェーピング `0.5·(γΦ'−Φ)`, `Φ=log10(1+CpS)`(購入で即加点し信用割当を助ける)
- `cookie_rl/agents/greedy.py` — CookieMonster流 payback period 貪欲ベースライン(+ サニティ用ランダム)
- `cookie_rl/bc_pretrain.py` — 貪欲を模倣する behavior cloning。PPOの初期方策を購入レジームに置く warm-start
- `cookie_rl/train.py` — MaskablePPO 学習。`--load` でBC初期値から微調整(`--ent-coef 0.005` で argmax を鋭く)
- `eval/compare.py` / `eval/eval_checkpoint.py` — held-out シードでの多方策比較・チェックポイント評価

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

# 学習パイプライン: BC warm-start → PPO 微調整(1ゲーム日)
uv run python -m cookie_rl.bc_pretrain --horizon-days 1 --episodes 24 --out checkpoints/bc_h1
uv run python -m cookie_rl.train --horizon-days 1 --timesteps 600000 \
    --load checkpoints/bc_h1.zip --ent-coef 0.005 --lr 1e-4 --run-name h1ft

# 評価(train=seed 0-3, BC=5000+ に対し held-out の 2000番台で比較)
uv run python eval/compare.py --horizon-days 1 --seeds 6 --seed-base 2000 \
    --models "bc=checkpoints/bc_h1.zip,ppo=checkpoints/h1ft_final.zip"
uv run tensorboard --logdir runs
```

## 行動空間

オートクリック(10/秒)、ゴールデンクッキーのポップ、Click Frenzy中のバースト連打(30/秒)は**常時ON**(harnessで自動)。いずれも常に最適な行動であり、トグルにすると「開始時クッキー0・CpS0でまずクリックを押さないと報酬が発生しない」探索障壁が生まれ、方策がnoopに退化するため。

| action | 内容 |
|---|---|
| 0 | noop(時間を進める / 貯金) |
| 1–20 | 建物 0–19 を1つ購入 |
| 21–28 | ストア内アップグレード(価格順スロット0–7)購入 |
| 29 | 転生(天国アップグレードは安い順に自動購入) |
| 30 | シュガーランプ収穫 |

## 実測値 (M-series Mac, 8コア)

- スループット: 62k〜88k Logic frames/s(1ゲーム日 ≈ 30〜40秒)
- 忠実性: CpS会計誤差0.00%、仮想時計ドリフト0ms、GC出現28回/3h(理論12〜30)

### 方策比較(1ゲーム日、log10 累計焼成クッキー、held-out 6シード平均)

| 方策 | 受動プレイ | **能動ゴールデンクッキー活用** |
|---|---|---|
| random | 12.75 ± 0.28 | 12.97 ± 0.11 |
| greedy (payback) | 15.58 ± 0.66 | **17.56 ± 0.94** |
| BC (貪欲模倣) | 14.75 ± 0.31 | 16.37 ± 0.51 |
| PPO (BC+微調整) | ~14(貪欲を超えず) | — |

要点:

- **ホライズン依存**: 0.25ゲーム日(6時間)は短すぎて全方策が誤差内で横並び(~11)。1ゲーム日で初めて payback-period 貪欲がランダムを約1000倍上回り、比較が意味を持つ。
- **能動ゴールデンクッキー活用で天井が約100倍上がる**: Click Frenzy(クリック×777)発動中に大クッキーを毎フレーム(30クリック/秒 = シミュのtick速度、自動)連打するだけで、貪欲は 1e15.6 → **1e17.6**。ランダムがほぼ不変なのは、Click Frenzyの価値がクリック単価(≈CpS)に比例し、CpSを育てないランダムは恩恵ゼロだから。「良い経済を作った上でコンボを叩く」という定石を再現している。
- **貪欲がほぼフロンティア**: behavior cloning は貪欲をニューラルネット方策として概ね再現(能動時 16.4)。PPO 微調整は貪欲を超えられず、むしろ BC をわずかに劣化させる(下記)。この単純化した行動空間では payback-period 貪欲 + 能動コンボがほぼ最適。

## 設計上の要点・ハマりどころ

- **アクションマスクは `locked` ではなく affordability で判定する。** `Object.buy()`(main.js ~8136)は `Game.cookies>=price` しか見ず `locked` を無視。ヘッドレスでは建物 unlock 処理(Draw/ストア更新側)が走らず `locked` は永久に1のまま。マスクに `not locked` を入れると全建物購入がブロックされ、RLは序盤 noop しか選べず**全 noop に退化**する(これが最大のバグだった)。
- **ホライズンが短いと方策の優劣が出ない。** 0.25ゲーム日(6時間)では複利も転生も効かず全方策が誤差内で横並び。1ゲーム日で初めて貪欲がランダムを約1000倍(3桁)上回り、比較が意味を持つ。
- **PPO を scratch から回すと決定論方策が noop⇄購入で振動する。** 序盤の「買う/待つ」確率が50%付近で argmax が反転するため。貪欲を模倣する **BC warm-start** で購入レジームに初期化してから微調整すると安定する。
- 単一シード評価はゴールデンクッキーRNG等で分散が大きい。結論は必ず複数シードで取る。

## 継続課題

- **PPO 微調整が BC 方策を超えられない。** critic ウォームアップ・advantage 正規化オフ・エントロピーゼロ・低学習率まで試したが、貪欲近傍の良い方策を PPO の確率的更新で動かすと平均的に劣化する(この環境では貪欲がほぼ最適で、微調整の伸びしろが小さい)。貪欲超えには複数ゲーム日ホライズンでの転生戦略など、貪欲の近視眼が効かない領域が要る。
- **さらに天井を上げる余地**: Frenzy(CpS×7)×Click Frenzy の同時発動を狙う combo、"Force the Hand of Fate" によるゴールデンクッキー強制発生、Lucky 狙いのバンク温存、グランマポカリプスの判断は未対応。
- 転生(Ascension)を跨ぐ 2〜3 ゲーム日ホライズンでの長期戦略(プレステージ複利)は未評価。

## ライセンスと免責

- 本リポジトリのコード(RL環境・エージェント・学習/評価スクリプト)は MIT ライセンス([LICENSE](LICENSE))。
- **Cookie Clicker 本体は含まれていない。** 本体(`main.js`・画像・音声等)は Orteil / DashNet の著作物(プロプライエタリ)で、`scripts/setup_game.sh` が各自のローカルに第三者ミラーから取得する。`vendor/` は `.gitignore` 済みで、ゲームコードやアセットを再配布しない。
- 本プロジェクトは Orteil / DashNet とは無関係の非公式な研究用ツール。
