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

### Phase 8-1: コーパス拡大 (Aozora + Wikipedia 日本語版 混合) — 🟡 進行中

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

#### [C] Aozora + Wikipedia 混合コーパス ⏳ 未着手

- 目標: `corpus/aozora_wikipedia_mixed.txt` を生成
- 構成: Aozora v2 (8M char) + Wikipedia (~990M char) を結合 → **~1B char**
- 配合: 約 1:120 (Aozora は希少だが文学的文体を保持するため repeat なしで投入)
- 記事/作品の区切り: `\n\n` (CharBPE が改行を独立 token として保持するので段落構造の信号になる)

#### [D] CharBPE vocab 8K → 32K 再訓練 ⏳ 未着手

- **入力**: `corpus/aozora_wikipedia_mixed.txt` (~1B char)
- **vocab**: 32,000 (Aozora 文学語彙 + Wikipedia 一般語彙 + Aozora special token 10 個)
- **学習サンプル**: 1M chars (Aozora 8M を全域、 Wikipedia から sampling)
- **期待 chars/token**: 1.8-2.0 (vocab 倍増の効果、 現 8K で 1.46 → 32K で +20-30% 圧縮)
- **出力**: `tokenizers/charbpe_v32010_aozora_wikipedia.bin`
- **学習時間予測**: rayon 並列化 + sample 1M chars で 5-10 分

#### [E] Phase 8 config 追加 + 学習起動 ⏳ 未着手

- `Config::aozora_wikipedia_d768_n8_charbpe32k_max1024_wsd()` を `main.rs` に追加
- パラメタ: d=768, n_heads=12, n_layers=8, d_ff=3072, max_len=1024, vocab=32010, batch=16
- 期待 params: ~50M
- end_step: 5000-8000 (Wikipedia 規模で 1-2 epoch)
- 期待 BPC: 3.5-3.8 (Aozora v2 7-a 比 -15% 〜 -17%)
- per-step 予測: 18-25 s (Phase 7-5 binary、 d=768 で AMX 効率向上)
- 完走時間予測: 25-40 h (連続稼働)

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
