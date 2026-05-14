# Phase 8: 大規模コーパス + Tokenizer 拡大 + モデル拡大

## 背景

Phase 7-a までで CharBPE 8K + Aozora v2 (8.28M char) で **BPC 4.22 (val)** を達成した。
次の質的向上には **データ量 + パラメタ量** の同時拡大が必要であり、 [Chinchilla scaling law](https://arxiv.org/abs/2203.15556) に従えば
20M params に対し最適データ量は ~400M token (= ~600M char @ 1.5 chars/token)。
現状の 5M token は **80x 不足**しており、 モデル拡大の前にデータ拡大が必須となる。

Phase 8 では以下の 3 軸を順に拡大する。

| 軸 | 現状 (Phase 7-a) | Phase 8 目標 | 比率 |
|---|---:|---:|---:|
| コーパス chars | 8.28 M | **~1 B** | 120x |
| Tokenizer vocab | 8,009 | **32,000** | 4x |
| モデル params | ~20 M | **~50 M** (d=768, L=8) | 2.5x |

Phase 8-1 (コーパス) → 8-2 (tokenizer) → 8-3 (model) の順で着手。

## サブフェーズ

### Phase 8-1: コーパス拡大 (Aozora + Wikipedia 日本語版 混合) — 🟡 [E] 起動待ち

#### [A] Wikipedia 日本語版の取得 ✅ 完了 (2026-05-13)

- **入手元**: HuggingFace `wikimedia/wikipedia` dataset, snapshot `20231101.ja`
  ([URL](https://huggingface.co/datasets/wikimedia/wikipedia/tree/main/20231101.ja))
- **方法**: Pure Rust 実装 (`src/bin/fetch_wikipedia_ja.rs`) で parquet を逐次 DL → `arrow` で `text` 列を抽出
- **使用ファイル**: 15 ファイル中 `train-00000-of-00015.parquet` 〜 `train-00003-of-00015.parquet` (4 ファイル)
- **結果**:
  - DL 合計: 1,490 MB (parquet)
  - 抽出 text: **1,043,726,050 chars** / 370,523 記事
  - 処理時間: 63 秒 (DL 55 秒 + parquet 解凍 8 秒)
  - 出力: `corpus/wikipedia_ja_raw.txt` (2.5 GB)
- **ライセンス**: CC-BY-SA 4.0 (Wikimedia Foundation)

#### [B] Wikipedia コーパスのクレンジング ✅ 完了 (2026-05-13)

- **実装**: `src/bin/clean_wikipedia_corpus.rs`
- **クレンジングルール**:
  1. 末尾の reference 系セクション (`脚注`, `注釈`, `出典`, `関連項目`, `外部リンク`, `参考文献`,
     `関連書籍`, `参照`, `補注`, `脚注・参考文献`, `参考`, `脚注・出典`, `注`, `出典・脚注`)
     以降を切り捨て (自然文 prose に集中させるため)
  2. 連続空行を 1 行に正規化
  3. trailing whitespace を除去
  4. 200 chars 未満の記事を破棄 (stub 排除)
  5. 記事区切りは 1 空行 (`\n\n`)
- **結果**:
  - 入力: 1,043,726,050 chars, 370,523 記事
  - 出力: **974,734,708 chars** (93.4% 保持) / 345,958 記事
  - 短い記事破棄: 24,565 (6.6%)
  - trailing section truncate: 327,621 記事 (88.4%)
  - 処理時間: ~8 秒
  - 出力: `corpus/wikipedia_ja.txt` (2.4 GB)
- **品質統計**:
  - Unique chars (Unicode codepoints): **15,834** (Aozora v2 の 5,220 から 3x 拡大)
  - 改行密度: 2.58% (Aozora v2 1.19% より高い、 記事スタイルの段落多め)
  - 平均記事長: ~2,820 chars
  - 中央値記事長: ~1,303 chars

#### [C] Aozora + Wikipedia 混合コーパス ✅ 完了 (2026-05-14)

- **実装**: `src/bin/mix_corpus.rs` で単純連結 (Aozora を `REPEAT_AOZORA=1` 回、 `\n\n` で繋ぐ)
- **結果**:
  - Aozora: 8,273,062 chars (0.842%)
  - Wikipedia: 974,734,708 chars (99.158%)
  - 合計: **983,007,772 chars** (~983M, 2.4 GB)
  - 処理時間: 4.0 秒
  - 出力: `corpus/aozora_wikipedia_mixed.txt`
- **境界**: 最後の Aozora 作品の `<EOS>` の後に `\n\n` を挟んで Wikipedia の最初の記事 (アンパサンド) が続く。 Aozora の `<BOS><AUTHOR=...><TITLE>...</TITLE>` special token はそのまま保持。
- **学習時の挙動**: chunk_len=1024 のスライディングウィンドウで random sampling → Aozora 由来の窓は ~0.84% 程度 (= 3000 step × batch 16 ≒ 48000 窓 中 ~400 窓)。 Wikipedia 主体だが文学的文体の信号は維持される想定。
- **注**: Aozora の重み付けが不足だった場合、 `REPEAT_AOZORA` を 5-20 に上げて再生成可能。 Phase 8-1 [E] 実行後の生成品質を見て判断する。

#### [D] CharBPE vocab 8K → 32K 再訓練 ✅ 完了 (2026-05-14)

- **実装**: `src/bin/train_tokenizer_phase8.rs`
- **入力 coverage_text**: `corpus/aozora_wikipedia_mixed.txt` 全体 (983,007,772 char、 unique char ~15.8K)
- **入力 merge_text**: stratified sample (Aozora 先頭 500K chars + Wikipedia 先頭 1.5M chars = **2,000,002 chars**)
  - Aozora の `<BOS><AUTHOR=...><TITLE>...</TITLE>` 区切りパターンと Wikipedia の現代日本語 + 英数記号の両方を BPE merge が学習できるよう構成
- **target vocab**: 32,000 (BPE) + 10 special token = **32,010** 想定 → **実 32,009** (`</TITLE>` が BPE merge と衝突して既存 id=18334 を再利用、 重複追加なし)
- **special tokens**: Phase 7-1 と同一 10 個 (`<TITLE>` / `</TITLE>` / `<DRAMA>` / `</DRAMA>` + 6 作家)
- **実測 chars/token** (50K char サンプル):
  - Aozora: **1.795** (Phase 7-a 8K の 1.46 から +23%)
  - Wikipedia: **1.873** (英数 + 漢字混在で BPE が効く)
- **出力**: `tokenizers/charbpe_v32010_aozora_wikipedia_mixed.bin`
  (Phase 8-1 [E] config の `tokenizer_cache_path()` に合わせて元の `charbpe_v32010_aozora_wikipedia.bin` からリネーム)
- **学習時間**:
  - 初回試行 (Phase 7-a 並走): 11 分稼働後に中断 (CPU 競合で per-step が 7,700ms → 10,000ms に劣化)
  - **正式実行 (Phase 7-a 完走後の単独実行)**: 110.8 分 (rayon フル稼働)
- **動作確認**: 混合コーパス先頭の `<BOS><AUTHOR=国木田独歩><TITLE>あの時分</TITLE>` が special token として正しく解釈

#### [E] Phase 8 config 追加 + 学習起動 — 🟡 config 実装済 / 起動待ち

- **config method**: `Config::aozora_wikipedia_mixed_d768_n8_charbpe32k_max1024_wsd()` (`src/main.rs`)
- **run_name**: `phase8a_aozora_wikipedia_mixed_d768_n8_charbpe32k_rms_swiglu_rope_max1024_wsd`
- **パラメタ**:
  - shape: d=768, n_heads=12, n_layers=8, d_ff=3072, max_len=1024 → ~50M params (Phase 5-4c 予約スケール)
  - vocab=32,010 (cfg側、 実 tokenizer 32,009)
  - batch_size=16
  - lr_max=5e-4, lr_min=5e-5, warmup_steps=500 (Phase 7-a の 7e-4 から GPT-2 small 慣例に合わせ控えめに)
  - **WSD**: warmup 500 + stable 8000 + decay 1500 = **end_step 10,000** (decay 比 15%、 Phase 6-d の 18% より緩め)
  - 観測: log 50 step / save+val 500 step (長期 run のため間隔拡大)
- **計算量見積**:
  - per-step: ~17-20 s (Phase 7-a ~7,700 ms から params 2.4x + d² 比例な部分で約 2.2-2.6x)
  - 10,000 steps × ~18 s ≈ **50 時間 ≈ 2 日** (M1 Max + Accelerate)
- **期待**:
  - val_ppl は corpus 領域差 (Wikipedia 主体) で Phase 7-a と直接比較不能、 BPC を主指標に。
  - 期待 BPC: 3.5-3.8 (Aozora v2 7-a 比 -15% 〜 -17%)。
  - Wikipedia の説明文体、 人物名・歴史的事実・カタカナ概念の生成品質が主観的評価ポイント。
- **起動コマンド** (確認後にコメントアウトを外して実行):
  ```bash
  cargo run --release 2>&1 | tee logs/phase8a_aozora_wikipedia_mixed_d768_n8_charbpe32k_rms_swiglu_rope_max1024_wsd.log
  ```
  起動前に `# loaded cached tokenizer from tokenizers/charbpe_v32010_aozora_wikipedia_mixed.bin (vocab=32009)` のログを必ず確認 (cache miss だと 110 分の BPE 再訓練が走る)。

### Phase 8-2 / 8-3 (将来): bf16 mixed precision + GPT-2 Small Compact

詳細は [`docs/roadmap.md`](roadmap.md) を参照。

## 実装メモ

### `src/bin/fetch_wikipedia_ja.rs`

```rust
const DATASET_BASE_URL: &str = "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.ja";
const NUM_FILES: usize = 15;
const TARGET_CHARS: usize = 1_000_000_000;
```

- `reqwest::blocking` (rustls-tls) で parquet を逐次 DL
- `parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder` で text 列を抽出
- 記事間区切り: `\n\n<DOC_SEP>\n\n` (後段クレンジングで除去)
- 累計 char 数が `TARGET_CHARS` を超えたら停止

### `src/bin/clean_wikipedia_corpus.rs`

- `\n\n<DOC_SEP>\n\n` で記事に分割
- 各記事を `clean_article` で処理 (trailing section 切り捨て、 空行正規化)
- 200 chars 未満の記事を破棄
- 出力時は記事間を `\n\n` (1 空行) で区切る

## 依存関係追加 (Cargo.toml)

Phase 8-1 用の heavy deps を追加した:

```toml
parquet = { version = "53", default-features = false, features = ["arrow", "snap", "zstd"] }
arrow-array = "53"
arrow-schema = "53"
reqwest = { version = "0.12", default-features = false, features = ["blocking", "rustls-tls"] }
```

学習側 binary には影響しないが、 cargo build 時の transitive deps は増える。
将来コーパス取得が完了した後、 ワークスペース分離を検討してもよい。

## ストレージ

- `corpus/wikipedia_ja_raw.txt`: 2.5 GB (中間生成物、 削除可)
- `corpus/wikipedia_ja.txt`: 2.4 GB (学習用)
- `corpus/wikipedia_ja_report.txt`: 取得・クレンジングサマリ
- 一時 parquet キャッシュ `corpus/_wiki_tmp/`: 1.5 GB (削除済み)

## 関連ドキュメント

- [Phase 6 (CharBPE 化)](phase6.md)
- [Phase 7 (Aozora v2 + special token)](phase7.md)
- [Roadmap](roadmap.md)
- [SoTA 比較](sota_comparison.md)
