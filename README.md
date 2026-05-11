# transformer

Rust で書かれた Transformer (decoder-only) 言語モデルの学習・推論実装。
外部 ML フレームワークに依存せず、 行列演算から自前で実装している学習用プロジェクト。

## ドキュメント

詳細はフェーズ別・トピック別に `docs/` に整理してある:

| ドキュメント | 内容 |
|------------|------|
| [docs/performance.md](docs/performance.md) | 並列化と性能 (rayon / matrixmultiply / Apple Accelerate のベンチマーク、 `Matrix` 設計) |
| [docs/tuning.md](docs/tuning.md) | チューニングのコツ (学習率・ミニバッチ・過学習・推論サンプリング) と CSV ログの読み方 |
| [docs/phase2.md](docs/phase2.md) | Phase 2: Tiny Shakespeare (BPE 4k, d_model=256) ベスト推論サンプル |
| [docs/phase3.md](docs/phase3.md) | Phase 3: nanoGPT 等価設定 (char 65, d_model=384) との直接比較・val_loss で 3.4% 上回る |
| [docs/phase4.md](docs/phase4.md) | Phase 4: 青空文庫 / 漱石 7 作品 (char) への拡張・容量律速の観察 |
| [docs/phase_d.md](docs/phase_d.md) | Phase D: モダンアーキテクチャ導入 (RMSNorm + SwiGLU 完了、 RoPE 学習中で early step -22% 改善) |
| [docs/roadmap.md](docs/roadmap.md) | 今後の改善案 (Phase D 進捗 / KV cache / OpenBLAS 等) |

## 依存

- Rust (edition 2024)
- `rand = "0.10.1"`
- `rayon = "1.10"` — 行列演算・損失計算の並列化
- `matrixmultiply = "0.3"` — `Matrix::matmul` の SIMD 最適化された pure-Rust BLAS（非 macOS 環境のフォールバック）
- macOS: Apple Accelerate Framework — OS 標準なので追加クレート不要、 `#[link(name = "Accelerate", kind = "framework")]` で直接リンク

## 実行

### コーパス取得

学習データは外部から取得する。 用途別にスクリプトを用意してある:

```bash
# Tiny Shakespeare (1.1 MB, 英語) — Phase 2 / 3 用
./scripts/download_tiny_shakespeare.sh
# → corpus/tiny_shakespeare.txt

# 夏目漱石「こころ」 (162k char ≈ 484 KB UTF-8, 日本語) — Phase 4a 用
./scripts/download_aozora_kokoro.sh
# → corpus/aozora_kokoro.txt

# 夏目漱石主要長編 7 作品 (1.21M char ≈ 3.5 MB UTF-8, 日本語) — Phase 4b 用
./scripts/download_aozora_soseki_works.sh
# → corpus/aozora_soseki_works.txt
# 含まれる作品: 吾輩は猫である / 坊っちゃん / 草枕 / 三四郎 / 行人 / こころ / 道草
# (全て新字新仮名のみ。 「それから」「門」 は仮名遣いが異なるため除外)
```

青空文庫系のスクリプトは Shift-JIS zip から UTF-8 に変換し、
ルビ (`《...》`)・ 編集注記 (`［＃...］`)・ 底本情報を除去した本文を出力する
(python3 が必要)。

### 学習

```bash
cargo run --release
```

`src/main.rs` の `main()` で選択した `Config` (例: `Config::aozora_kokoro()`)
の設定で学習が始まり、 `checkpoints/<run_name>/` 配下に checkpoint が保存される。

### CSV ログとして保存

学習ログは CSV 形式 (`step,loss,ema,min,max,lr,ms_per_step,elapsed_s`) で標準出力に流れる。
コンパイラ出力やヘッダーコメントを除外して CSV を取り出すには:

```bash
cargo run --release -q 2>&1 | tee train.log
grep -E '^(step,|[0-9]+,)' train.log > train.csv
```

`-q` で cargo の `Compiling`/`Finished` メッセージを抑制し、 `grep -E` で
ヘッダー (`step,...`) と数字始まりの行のみを抽出する。

ログ形式の詳細は [docs/tuning.md#ログの読み方](docs/tuning.md#ログの読み方) を参照。

### checkpoint から再開・推論

`src/main.rs` の `main()` 内で対応する関数の呼び出しを切り替える:

```rust
fn main() {
    // 学習対象の Config を選ぶ (corpus_path はそれぞれの Config 内で固定)
    let cfg = Config::aozora_kokoro();    // または ::nano_gpt_equivalent() / ::tiny_shakespeare()

    // 新規学習
    training_and_inference(&cfg);

    // checkpoint から再開（同じモデル構造の checkpoint のみ）
    // training_from_checkpoint(&cfg, "checkpoints/<run_name>/latest.bin");

    // checkpoint を読み込んで推論のみ
    // inference_from_checkpoint(&cfg, "checkpoints/<run_name>/inference.bin");
}
```

`training_from_checkpoint` で再開する場合、 checkpoint の `d_model` / `n_heads` / `d_ff` /
`n_layers` / `vocab_size` が `Config` と一致している必要がある。 構造を変えた場合は
`training_and_inference` (fresh start) を使う。

## 設定

`src/main.rs` の `Config::tiny_shakespeare()` で全パラメータを指定する。

| 項目 | 役割 |
|------|------|
| `run_name` | checkpoint 保存先サブディレクトリ名 |
| `corpus_path` | 学習データのパス (各 Config 内で固定) |
| `tokenizer_kind` | `TokenizerKind::Bpe` / `Char` の切替 |
| `d_model` / `n_heads` / `d_ff` / `n_layers` | モデル構造 |
| `max_len` | 最大コンテキスト長（位置エンコーディング上限） |
| `vocab_size` | BPE トークナイザの語彙サイズ (Char では無視) |
| `normalization_kind` | `NormalizationKind::Layer` / `Rms` の切替 (Phase D-1 で追加) |
| `feed_forward_kind` | `FeedForwardKind::Gelu` / `SwiGlu` の切替 (Phase D-3 で追加) |
| `positional_encoding_kind` | `PositionalEncodingKind::Sinusoidal` / `Rope` の切替 (Phase D-2 で追加) |
| `dropout` / `weight_decay` / `beta2` | 正則化・最適化のハイパラ |
| `lr_max` / `lr_min` / `warmup_steps` | 学習率スケジュール（warmup + cosine decay） |
| `end_step` | 総学習ステップ数 |
| `batch_size` | ミニバッチサイズ |
| `save_every` / `log_every` | checkpoint 保存・ログ出力間隔 |
| `val_every` / `val_n_batches` / `val_split_ratio` | validation 計測の頻度・粒度 |

各値の選び方の指針は [docs/tuning.md](docs/tuning.md) を参照。

## ディレクトリ構成

```
src/
├── main.rs                    # エントリ・学習ループ・Config (corpus 別プリセット)
├── language_model.rs          # モデル全体（埋め込み→Transformer→出力）+ generate / val 用 forward_loss
├── transformer.rs             # Transformer (block の積み重ね、 final_norm 含む)
├── transformer_block.rs       # 1 ブロック (MHA + Norm + FFN + Norm + Dropout × 2)
├── multi_head_attention.rs    # マルチヘッドアテンション (RoPE 統合済)
├── feed_forward.rs            # FeedForward trait + FeedForwardKind enum (Gelu / SwiGlu 切替)
├── feed_forward_network.rs    # 位置ごとの FFN (Linear → GELU → Linear)
├── swiglu_feed_forward_network.rs  # SwiGLU FFN (gate/up/down 3 行列、 LLaMA 流) ※Phase D-3 で組込済
├── normalization.rs           # Normalization trait + NormalizationKind enum (Layer / Rms 切替)
├── layer_normalization.rs     # Layer Normalization (γ, β 学習 + 数値安定化)
├── root_mean_square_layer_normalization.rs  # RMSNorm (γ のみ、 mean 計算なし) ※Phase D-1 で組込済
├── positional_encoding.rs     # PositionalEncodingKind enum (Sinusoidal / Rope 切替)
├── rope.rs                    # Rotary Position Embedding (回転で相対位置を内積に保存) ※Phase D-2 で組込済
├── dropout.rs                 # Inverted dropout (training / eval 切替)
├── embedding.rs               # トークン埋め込み
├── sinusoidal_pe.rs           # 正弦波位置エンコーディング
├── output_head.rs             # 語彙への射影
├── adam_w.rs                  # AdamW オプティマイザ (weight decay / β2 設定可)
├── lr_scheduler.rs            # warmup + cosine スケジューラ
├── cross_entropy_loss.rs      # 系列全体のクロスエントロピー損失 (PAD は loss から除外)
├── tokenizer.rs               # Tokenizer trait と TokenizerKind enum (BPE / Char の切替)
├── bpe_tokenizer.rs           # BPE トークナイザ (byte-level + 句読点 split)
├── char_tokenizer.rs          # 文字単位トークナイザ (vocab はコーパス文字種から自動生成)
├── eval.rs                    # 学習中の val_loss 計測 (90/10 split + ランダム窓)
├── checkpoint.rs              # 重み・状態の保存/読込
└── matrix.rs                  # 行優先 flat 表現の `Matrix` と並列化された行列演算

scripts/
├── download_tiny_shakespeare.sh    # Karpathy char-rnn から取得
├── download_aozora_kokoro.sh       # 青空文庫「こころ」 → UTF-8 + ルビ除去 (要 python3)
└── download_aozora_soseki_works.sh # 漱石主要長編 7 作品を連結 (要 python3)

corpus/                             # gitignore (各種スクリプトで再生成可能)
├── tiny_shakespeare.txt            # Phase 2 / 3 用 (英語 1.1 MB, ~330k token)
├── aozora_kokoro.txt               # Phase 4a 用 (日本語 484 KB, ~162k char)
└── aozora_soseki_works.txt         # Phase 4b 用 (日本語 3.5 MB, ~1.21M char)

logs/                              # gitignore (学習ログの保存先)
└── <run_name>.log

docs/                              # 詳細ドキュメント (本 README からリンク)
├── performance.md                 # 並列化と性能ベンチマーク
├── tuning.md                      # チューニングのコツ + ログ仕様
├── phase2.md                      # Tiny Shakespeare 推論サンプル
├── phase3.md                      # nanoGPT との比較
├── phase4.md                      # 日本語コーパスへの拡張
├── phase_d.md                     # モダンアーキテクチャ導入 (D-1 + D-3 完了、 D-2 RoPE 学習中)
├── roadmap.md                     # 今後の改善案
├── learning_rate.png              # tuning.md から参照
└── lr_schedule.png                # tuning.md から参照

checkpoints/<run_name>/
├── step_NNNNNN.bin            # 学習途中の checkpoint
├── latest.bin                 # 直近の checkpoint
├── best.bin                   # val_loss 最良時の checkpoint (Phase 4 以降)
└── inference.bin              # 学習完了後の推論用 checkpoint
```
