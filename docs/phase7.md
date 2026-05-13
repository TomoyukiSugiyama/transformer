# Phase 7-1: コーパス前処理改善 + 作家・戯曲 special token

Phase 6-d までの推論サンプルから観察された 4 つの質的課題:
- 戯曲記号 / 戯曲文体の散文への混入 (「ハムレット」「王妃」が漱石プロンプトに登場)
- 作家ヘッダ (`===== 森鴎外『...』 =====`) を学習・生成してしまう
- 作家ミックス (キャラクタ・文体が複数作家で混在)
- repetition collapse (一部解消、 まだ残る)

これらを解決するため Phase 7-1 では **コーパスを再構築 + special token を追加** したうえで再学習する。

## 1. 現状コーパスの構造分析 (`src/bin/analyze_corpus.rs` 実行結果)

### 1.1 全体規模

| 指標 | 値 |
|------|-----|
| 総 char 数 | 8,274,972 |
| 総 line 数 | 100,588 |
| 平均行長 | 82 char |

### 1.2 作家分布 (作品数 / char 数 / 占有率)

| 作家 | 作品数 | char 数 | 占有率 |
|------|-------:|--------:|-------:|
| **夏目漱石** | 84 | 3,318,785 | **40.11%** |
| **太宰治** | 202 | 2,195,920 | **26.54%** |
| 森鴎外 | 71 | 1,342,399 | 16.22% |
| 宮沢賢治 | 87 | 725,410 | 8.77% |
| 中島敦 | 25 | 367,885 | 4.45% |
| 国木田独歩 | 35 | 312,784 | 3.78% |

⚠️ 漱石 + 太宰だけで **66.7%** を占めるため、 それ以外の作家は **学習サンプル不足** で文体獲得が弱い (Phase 6-d 推論サンプルで「鴎外風」「賢治風」が安定しない一因)。

### 1.3 クレンジング候補

| カテゴリ | 出現 | 占有率 | 対応 |
|---------|------|-------:|------|
| 作家ヘッダ `===== ... =====` | 504 行 / 11,789 char | 0.14% | **special token 化必須** |
| 振り仮名 `《...》` | 0 | 0.000% | (既に除去済み) |
| 注釈 `［＃...］` | 0 | 0.000% | (既に除去済み) |
| 戯曲台詞行 (キャラ名 + 句読点 + 発言) | 2,097 行 / 285,277 char | **3.45%** | 集中 5-7 作品を special token で囲う |
| ト書き行 (行頭 `(` `（`) | 1,085 行 / 84,524 char | 1.02% | 上記と一緒に処理 |

### 1.4 戯曲の集中度

戯曲フォーマットの作品 TOP (台詞行率 50% 超):

| 作家 | 作品 | 戯曲台詞率 | 行数 |
|------|------|-----------:|-----:|
| 森鴎外 | 最終の午後 | **87.7%** | 57 |
| 森鴎外 | 家常茶飯　附・現代思想 | **87.1%** | 557 |
| 太宰治 | **新ハムレット** | **72.1%** | 603 |
| 森鴎外 | 辻馬車 | 71.6% | 74 |
| 森鴎外 | 痴人と死と | 66.7% | 45 |
| 宮沢賢治 | ペンネンノルデはいまはいないよ | 60.9% | 23 |
| 太宰治 | 花吹雪 | 48.2% | 83 |

→ Phase 6-d 推論で観察された **「メロスは...ハムレット。王妃」** はこの 7 作品 (特に新ハムレット) からの汚染。

## 2. P7-1 実装方針

### 2.1 新コーパス構造 (`corpus/aozora_meiji_taisho_v2.txt`)

```
<BOS><AUTHOR=夏目漱石><TITLE>三四郎</TITLE>
本文1行目
本文2行目
...
<EOS>
<BOS><AUTHOR=森鴎外><TITLE>最終の午後</TITLE><DRAMA>
ハムレット　お、何ぞ
王妃　わが子よ……
...
</DRAMA><EOS>
<BOS><AUTHOR=太宰治><TITLE>走れメロス</TITLE>
メロスは激怒した。...
<EOS>
```

特徴:
- **作家ヘッダ → `<AUTHOR=...>` `<TITLE>...</TITLE>` special token に変換** (生成時にこのトークンを抑制すれば「ヘッダ生成」が止まる)
- **戯曲集中作品 (台詞率 50% 超) を `<DRAMA>...</DRAMA>` で囲う** → モデルが「DRAMA 内」と「散文」を内部状態で区別できる
- 旧 `===== ... =====` は完全削除
- 各作品を `<BOS>` で開始、 `<EOS>` で終了 (現状 BOS/EOS 不在)

### 2.2 Tokenizer 拡張

CharBPE 8K に下記 special token を追加 (`extend_coverage` API を使う):

| Token | 用途 | 個数 |
|-------|------|-----:|
| `<BOS>` `<EOS>` | 作品境界 | 2 |
| `<AUTHOR=夏目漱石>` ... `<AUTHOR=国木田独歩>` | 作家識別 | 6 |
| `<TITLE>` `</TITLE>` | タイトル境界 | 2 |
| `<DRAMA>` `</DRAMA>` | 戯曲モード | 2 |
| **合計** | | **12** |

新 vocab: 8000 + 12 = **8,012** (model 形状に影響なし)

`<UNK>` は既存利用。 `<PAD>` も既存。

### 2.3 推論時の制御

- **作家指定生成**: prompt の先頭に `<BOS><AUTHOR=夏目漱石>` を挿入 → 漱石風出力に強制
- **戯曲モードを抑制**: top-k/top-p sampling 時に `<DRAMA>` token を mask (logit = -∞) → 散文プロンプトに戯曲が混入しない
- **ヘッダ生成防止**: `<AUTHOR=...>` `<TITLE>` token を生成時に mask

## 3. Phase 7-a 学習設定

| 設定項目 | Phase 6-d (実測) | **Phase 7-a** | 変更理由 |
|---------|-------------------|---------------|---------|
| コーパス | `aozora_meiji_taisho.txt` (8.27M char) | `aozora_meiji_taisho_v2.txt` (8.28M char、 章番号削除済) | 戯曲・ヘッダ・章番号擾乱の構造的排除 |
| Tokenizer | CharBPE 8K (vocab=8000) | CharBPE 8010 (+10 special token) | 作家・戯曲・タイトル・BOS/EOS の atomic tokenize |
| 形状 | d=512, n=6, max_len=1024 | 同 | (互換) |
| LR scheduler | WSD (warmup=300, stable=2160, decay=540) | 同 | (Phase 6-d で良好) |
| バイナリ | Phase 7-1/7-2 (per-step ~9,200 ms) | **Phase 7-4** (per-step ~5,800 ms 想定) | Phase 7-3 (+1.12x) + Phase 7-4 (+1.43x) で **累積 1.59x speedup** |
| total step | 3000 (best step 2800) | 3000 (要再検討、 速度余裕あれば 3500 へ増量も可) | データ量ほぼ不変 |
| 想定 total time | 7h 41min (実測) | **~5 h** (Phase 7-4 効果) | -35% |

期待効果 (Phase 6-d 実測 BPC=4.22 基準):
- **BPC: 4.22 → 3.9-4.1** (-3 〜 -7%)。 主因は (a) 戯曲台詞 (3.45% of tokens) のノイズ除去で perplexity 改善、 (b) 作家トークンによる「どの作家の語彙か」を学習可能に。
- 推論品質: **戯曲混入解消** (推論時に `<DRAMA>` token を logit mask) + **作家文体の使い分け鮮明化** (prompt 先頭に `<AUTHOR=...>` を挿入で強制)
- **ヘッダ生成消失** (`<AUTHOR=...>` `<TITLE>` token を logit mask)

期待が控えめ (Phase 6-d 計画の -5 〜 -8% より小さく設定) なのは、 Phase 6-d が当初期待 -2 〜 -5% に対し実測 -0.6% に留まった経験を踏まえての保守的見積り。

## 4. 実装手順 (P7-1-B 以降)

| ステップ | 内容 | 状態 |
|---------|------|------|
| **P7-1-B** | `src/bin/clean_aozora_corpus.rs` 作成。 旧コーパスを v2 形式に変換 | ✅ 完了 (504 作品 → 8,285,143 char、 戯曲 8 作品を `<DRAMA>` で囲む) |
| **P7-1-C** | `CharBpeTokenizer::add_special_token` API + `src/bin/extend_tokenizer.rs` で 10 個の special token を追加 | ✅ 完了 (`tokenizers/charbpe_v8010_aozora_meiji_taisho_v2_s500000.bin`) |
| **P7-1-D** | `main.rs` に `aozora_meiji_taisho_charbpe8k_max1024_wsd_v2()` config 追加。 学習起動 | ✅ config 追加済、 ⏳ ユーザー判断待ち (Phase 7-4 バイナリで学習 ~5 h 想定) |

### 4.1 P7-1-B: corpus cleaning (`src/bin/clean_aozora_corpus.rs`)

実行結果:
- 入力: `corpus/aozora_meiji_taisho.txt` (8,274,972 char)
- 出力: `corpus/aozora_meiji_taisho_v2.txt` (8,279,260 char、 +0.05%)
- **作家ヘッダ変換**: 504 行 (`===== 作家『タイトル』 =====` → `<BOS><AUTHOR=作家>`)
- **戯曲扱い (DRAMA で囲む)**: **8 作品** (172,382 char、 2.08%)
  - 鴎外『最終の午後』『家常茶飯　附・現代思想』『辻馬車』『痴人と死と』『夏目漱石論』
  - 太宰治『新ハムレット』『花吹雪』
  - 宮沢賢治『ペンネンノルデはいまはいないよ ...』
- 散文: 496 作品
- **章番号行削除 (Phase 7-1-B 第 2 弾)**: **1,131 行** 削除
  - パターン 1: 行頭全角/半角空白 + 漢数字 1-3 文字 (例: 「　　　　　五」) — 329 行
  - パターン 2: 単独漢数字 1-3 文字 (例: 「一」 「二」 「三」) — 790 行
  - パターン 3: 「第X章/節/部/編/話/卷/巻/回」 形式 — 14 行 (中間に漢数字/全角数字許容)
  - 「（一）」「(1)」 等の本文に出現する括弧付き番号は対象外
  - 検証: パターン 1+2+3 全て **残存 0 行** (rg で confirmed)

### 4.2 P7-1-C: tokenizer 拡張 (`src/bin/extend_tokenizer.rs` + `CharBpeTokenizer::add_special_token`)

`CharBpeTokenizer` に追加した API:
- `pub fn add_special_token(&mut self, token: &str) -> usize`: special token を atomic に登録。
- `pub fn special_tokens(&self) -> &[String]`: 登録 special token の参照。
- `encode_inner` を修正: テキスト中の `<...>` を **登録 special token と完全一致** したら 1 token として
  ID 化、 そうでなければ通常 BPE にフォールバック。
- `encode_long` を修正: 既に先頭が `<BOS>` token のとき auto-prepend を skip (BOS 重複防止)。
- `Checkpointable::to/from_weight_map`: `special_tokens` の永続化 (旧 cache は fallback)。

拡張後トークナイザ:
- `tokenizers/charbpe_v8010_aozora_meiji_taisho_v2_s500000.bin`
- vocab: 8000 → **8010** (+10 special token)
- 動作確認: `<BOS><AUTHOR=国木田独歩><TITLE>あの時分</TITLE>...` → `[2, 8009, 8000, 5330, 6338, 8001, ...]`
  (BOS=2, AUTHOR=8009, TITLE=8000, 「あの時分」 BPE = 5330+6338, /TITLE=8001) と完全に special token として認識される

新規テスト (3 ケース、 全 pass):
- `add_special_token_then_recognize_in_text`
- `special_tokens_survive_checkpoint_roundtrip`
- `unregistered_angle_brackets_are_char_encoded`
- `encode_long_skips_redundant_bos_eos`

## 5. リスクと判断ポイント

- **作家偏り** (漱石 40%, 太宰 27%) は Phase 7-1 で解決しない。 もし作家分散が必要なら Phase 7-b として「作家別サンプリング weighted」 を検討。
- **戯曲行検出ヒューリスティック** は完全ではない。 上位 7 作品を特殊扱いするのが安全。
- v2 コーパスでは作品数 = 504 (変わらず) だが、 BOS/EOS 追加で 504 \* 12 char ≒ 6K char 増。

## 関連ドキュメント

- [Phase 6 (CharBPE 化, 6-a/b/c/d)](phase6.md)
- [Performance (Phase 7 高速化)](performance.md)
- [Roadmap](roadmap.md)
