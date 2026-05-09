# transformer

Rust で書かれた Transformer (decoder-only) 言語モデルの学習・推論実装。
外部 ML フレームワークに依存せず、行列演算から自前で実装している学習用プロジェクト。

## 依存

- Rust (edition 2024)
- `rand = "0.10.1"`
- `rayon = "1.10"` （行列演算・損失計算の並列化）
- `matrixmultiply = "0.3"` （`Matrix::matmul` の SIMD 最適化された pure-Rust BLAS 実装、 非 macOS 環境のフォールバック）
- macOS: Apple Accelerate Framework（OS 標準なので追加クレート不要、 `#[link(name = "Accelerate", kind = "framework")]` で直接リンク）

## 実行

### 学習

```bash
cargo run --release
```

`Config::tiny_shakespeare()` で定義された設定で学習が始まり、`checkpoints/<run_name>/` 配下に
チェックポイントが保存される。

### CSV ログとして保存

学習ログは CSV 形式 (`step,loss,ema,min,max,lr`) で標準出力に流れる。
コンパイラ出力やヘッダーコメントを除外して CSV を取り出すには:

```bash
cargo run --release -q 2>&1 | tee train.log
grep -E '^(step,|[0-9]+,)' train.log > train.csv
```

`-q` で cargo の `Compiling`/`Finished` メッセージを抑制し、`grep -E` で
ヘッダー (`step,...`) と数字始まりの行のみを抽出する。

## 設定

`src/main.rs` の `Config::tiny_shakespeare()` で全パラメータを指定する。

| 項目 | 役割 |
|------|------|
| `run_name` | チェックポイント保存先サブディレクトリ名 |
| `d_model` / `n_heads` / `d_ff` / `n_layers` | モデル構造 |
| `max_len` | 最大コンテキスト長（位置エンコーディング上限）|
| `vocab_size` | BPE トークナイザの語彙サイズ |
| `lr_max` / `lr_min` / `warmup_steps` | 学習率スケジュール（warmup + cosine decay） |
| `end_step` | 総学習ステップ数 |
| `batch_size` | ミニバッチサイズ |
| `save_every` / `log_every` | チェックポイント保存・ログ出力間隔 |

## ディレクトリ構成

```
src/
├── main.rs                    # エントリ・学習ループ
├── language_model.rs          # モデル全体（埋め込み→Transformer→出力）+ generate
├── transformer.rs             # Transformer (block の積み重ね)
├── transformer_block.rs       # 1 ブロック (MHA + FFN + LayerNorm)
├── multi_head_attention.rs    # マルチヘッドアテンション
├── feed_forward_network.rs    # 位置ごとの FFN
├── layer_normalization.rs
├── embedding.rs               # トークン埋め込み
├── sinusoidal_pe.rs           # 正弦波位置エンコーディング
├── output_head.rs             # 語彙への射影
├── adam_w.rs                  # AdamW オプティマイザ
├── lr_scheduler.rs            # warmup + cosine スケジューラ
├── cross_entropy_loss.rs
├── bpe_tokenizer.rs           # BPE トークナイザ
├── checkpoint.rs              # 重み・状態の保存/読込
└── matrix.rs                  # 行優先 flat 表現の `Matrix` と並列化された行列演算

corpus/
└── train.txt                  # 学習データ（全行を連結し、ランダム窓でサンプリング）

checkpoints/<run_name>/
├── step_NNNNNN.bin            # 学習途中の checkpoint
├── latest.bin                 # 直近の checkpoint
└── inference.bin              # 学習完了後の推論用 checkpoint
```

## チェックポイントから再開・推論

`src/main.rs` の `main()` 内で対応する関数の呼び出しを切り替える:

```rust
fn main() {
    let corpus_text = load_corpus("corpus/train.txt");
    let cfg = Config::tiny_shakespeare();

    // 新規学習
    training_and_inference(&corpus_text, &cfg);

    // チェックポイントから再開（同じモデル構造の checkpoint のみ）
    // training_from_checkpoint(&corpus_text, &cfg, "checkpoints/<run_name>/latest.bin");

    // チェックポイントを読み込んで推論のみ
    // inference_from_checkpoint(&cfg, "checkpoints/<run_name>/inference.bin");
}
```

`training_from_checkpoint` で再開する場合、checkpoint の
`d_model` / `n_heads` / `d_ff` / `n_layers` / `vocab_size` が `Config` と一致している必要がある。
モデル構造を変えた場合は `training_and_inference`（fresh start）を使う。

## 並列化と性能

行列演算は `crate::matrix::Matrix` に集約されている。
内部表現は **行優先の flat `Vec<f32>`**（jagged な `Vec<Vec<f32>>` ではない）。
重い演算は `rayon` で並列化されており、CPU の全コアを活用する。

### `Matrix` の主な API

- 構築: `zeros`, `from_jagged`, `from_flat`
- 演算: `matmul`, `transpose`, `add_in_place`, `add_row_bias_in_place`,
  `sum_rows_into_cols`, `map`, `elementwise_with`, `softmax_rows_in_place`
- MHA 用: `split_columns(n)` / `concat_columns(&[Matrix])`

### 並列化・最適化の対象

| ファイル / 関数 | 最適化手段 | 並列粒度 |
|----------------|-----------|---------|
| `matrix::matmul` (macOS) | **Apple Accelerate `cblas_sgemm` (内部で AMX 活用)** | Accelerate 内部で AMX + マルチコア |
| `matrix::matmul` (その他 OS) | **`matrixmultiply::sgemm` (pure-Rust SIMD)** | クレート内部で SIMD + キャッシュタイル |
| `matrix::matmul_naive` | rayon `par_chunks_mut`、`i-k-j` ループ | 行ごと（数値検証・ベンチ用に保持） |
| `matrix::transpose` | rayon `par_chunks_mut` | 行ごと |
| `matrix::add_in_place` | rayon `par_iter_mut` | 要素ごと |
| `matrix::softmax_rows_in_place` | rayon `par_iter` | 行ごと |
| `output_head::logits_last` | rayon `par_iter` | vocab 次元 |
| `cross_entropy_loss::forward_sequence` | rayon `par_iter` | token ごと |

`matmul` は **OS 別に最適な BLAS バックエンドへ委譲**する:
- macOS: `cfg(target_os = "macos")` で `extern "C" { fn cblas_sgemm(...) }` を直接呼ぶ。
  Accelerate Framework は M1/M2/M3 系で内部的に **AMX co-processor** を使うため、
  大きめの行列で 2〜10× の高速化が得られる (詳細は後述の参考値)。
- その他 OS: `matrixmultiply::sgemm` で pure-Rust の SIMD カーネルにフォールバック。

flat row-major 表現を採用しているため、 どちらのバックエンドにもゼロコピーで
`leading_dim = cols` をそのまま渡せる。

`matmul_blas_matches_naive_for_random_matrices` テストで `(1,1,1)` から `(128, 256, 256)` まで
の各サイズで `matmul_naive` と一致することを保証している (浮動小数誤差 ≤ `1e-3 × k`)。

### Matrix を経由する主な処理

- `multi_head_attention`: Q/K/V/O 射影、scaled-dot-product attention、head 分割・結合
- `feed_forward_network`: forward / backward すべて、bias 加算、GELU、勾配集計
- `output_head`: forward / backward すべて、`logits_last` （単一トークン推論最適化）

### Matrix を経由していない箇所（軽量で並列化非対象）

- `embedding`: token id ベースの lookup
- `layer_normalization`: 行ごとの統計量計算（d_model 方向のみで小規模）
- `sinusoidal_pe`: 加算のみで軽量

### AdamW との橋渡し

`AdamW::step_matrix_flat(&mut self, &str, &mut Matrix, &Matrix)` が `Matrix` を直接受け取る。
内部の `AdamWParam.data` も `Vec<f32>` (flat) なので、 jagged ↔ flat の **flatten 変換コストはゼロ**。

### 効果（M1 Max / 10 コア環境での観察）

`d_model=128, max_len=64, batch_size=16` での 1 step あたりの所要時間がおよそ
**1 桁短く** なった（並列化前は MHA 以外がシリアル実行だったため）。
この高速化を前提に `d_model=256, max_len=128, d_ff=1024` といった大きめの構成も
数時間で 10000 step 学習できる。

`d_model=256, n_heads=8, d_ff=1024, n_layers=4, max_len=128, batch_size=16`
構成での参考値:

| 実装 | 1 step | 累積高速化 (推定含む) | 出典 |
|------|-------|---------------------|------|
| Vec<Vec<f32>> + シリアル (推定) | 〜30 s | 1.0× (基準) | 直接計測なし、 後述の根拠で外挿 |
| `Vec<Vec<f32>>` + rayon (Phase 2) | 約 3.0〜3.6 s | 約 8〜10× | Phase 2 (checkpoint mtime) |
| `Matrix` + rayon naive (BLAS 前) | 約 3.0 s (中央値 3030 ms) | 約 10× | `phase2_d256_ff1024_max128_before_blas.log` |
| `Matrix` + `matrixmultiply` (BLAS) | 約 1.24 s (中央値 1240 ms) | 約 24× | `phase2_d256_ff1024_max128_with_blas.log` |
| `Matrix` + Apple Accelerate (AMX) | （測定予定） | — | macOS 専用 |

シリアル版の推定は「シングルスレッド `Vec<Vec<f32>>` で実効 0.5〜1 GFLOP/s × 全 step ~20 GFLOPs」より外挿。
rayon の単独効果が ~10× と読み取れる。 BLAS で更に 2.4×、 AMX で更に 2× 程度を期待。

## チューニングのコツ

実装と実験を通じて得られた知見のまとめ。

### 学習データの整形
- **行ごと（短文）に切らず、コーパス全体を 1 つの token 列に連結**してランダム窓でサンプリングする方が、対話・改行・句読点などの構造を学習できる。
- 行ごとに `<EOS>` を付けると EOS バイアスが発生し、推論時に 1 トークンで打ち切られやすくなる。連結方式では EOS は不要（`max_new_tokens` で停止）。

### 学習率
- `lr_max ≒ 3e-4`、`lr_min ≒ 1e-5`、`warmup_steps ≒ 200` が小規模 Transformer の典型値。
- `lr_min` を `0` 付近にすると終盤が完全に止まるため、**`1e-5` 程度のフロアを残す**。
- checkpoint から end_step を伸ばして再開すると、scheduler の進捗がリセットされて lr が上振れする（**warm restart 効果**）。停滞局所解からの脱出に使える。

### バッチとミニバッチ勾配
- `batch_size` 個の `forward_backward` で勾配を累積し、まとめて 1 回 `apply_gradients` する。
- AdamW の `step_count` も 1 step に 1 回しか進めない（パラメータごとに進めない）。
- 累積勾配は `set_grad_scale(batch_size)` で平均化扱いにし、`apply_gradients` 後に `zero_grad` でリセットする。

### モデル構造の比率
- `d_ff = 4 × d_model` が標準（GPT-2、nanoGPT 等）。FFN は知識を蓄える主役なので `d_model` を増やすときは `d_ff` も比例して増やす。
- `d_head = d_model / n_heads = 32〜64` が扱いやすい。
- `max_len` を上げると attention は `O(n²)` で重くなる。学習対象（対話 1 ターン: 〜100 token、シーン: 〜500 token）に合わせて選ぶ。

### 推論時の繰り返し対策
- greedy はすぐに同じ語句に落ち込みやすいので、**top-k サンプリング + 適度な temperature**（例: `top_k=5, temperature=1.0`）の方が自然な文章になる。
- `my lord, my lord, ...` のような繰り返しは **repetition penalty** で軽減する。
  `LanguageModel::generate*` は `repetition_penalty` 引数を受け取り、 `1.1〜1.3` 程度が無難。
  HuggingFace と同じ式（正は割り算、負は掛け算）で過去に出現した token を抑制する。

### checkpoint 運用
- `run_name` を分けることで、設定違いの実験を上書きせずに並走できる（`checkpoints/<run_name>/`）。
- モデル構造（`d_model` 等）を変えると checkpoint 互換性が失われるため、構造変更時は fresh start（`training_and_inference`）を使う。

## 今後の改善案

### Linux/Windows 向け OpenBLAS / Intel MKL バックエンド
macOS では Accelerate を直接リンクしているが、 他 OS では `matrixmultiply` フォールバック。
Linux で更なる高速化が必要なら `openblas-src` + 自前 extern、 もしくは `ndarray-linalg`
経由で OpenBLAS / MKL を呼ぶ手がある。 macOS と同じ構造 (`#[cfg(target_os = ...)]` 分岐) で
追加できる。

### モデルサイズ拡大による AMX の本領発揮
現状の `d_model=256` では 1 つの matmul サイズが中規模で AMX の旨味が部分的。
`d_model=512〜768` まで拡大すると matmul の比率も問題サイズも大きくなり、
Accelerate が更に効く。 ただしパラメータ数が `d_model²` で増えるので、 corpus 規模との
バランスに注意 (Tiny Shakespeare では過学習しやすい)。

### 推論時のキャッシュ機構（KV cache）
現在の `generate` は token 1 つ生成するたびに過去のトークンを含む全 context を
attention で再計算している。KV cache（Q/K/V の中間結果を保持）を導入すれば
1 token 生成あたりの計算量が `O(n)` → `O(1)` 近くまで下がる。
`max_len=128` 以上の生成で大きく効く。

### 生成制御の追加
- top-p (nucleus) sampling
- 最小生成トークン数 (`min_new_tokens`)
- bad words / banned ngrams フィルタ

### `layer_normalization`, `embedding` 等の統一
これらは現在 `Vec<Vec<f32>>` をやりとりしているが、内部の API 境界で `Matrix::from_jagged`/
`to_jagged` 変換が走っている。すべてを `Matrix` で統一すれば変換コストが消え、コードも整う。
ただし計算ボトルネックではないので優先度は低い。

## ログの読み方

```
# run_name=phase2_d256_ff1024_max128
# d_model=256, n_heads=8, ...
# lr_max=0.0003, lr_min=0.00001, warmup_steps=200, ...
step,loss,ema,min,max,lr,ms_per_step,elapsed_s
20,6.7432,7.6912,6.7432,8.4294,3.000e-5,425.3,8.5
40,6.1839,6.7892,6.0621,6.7436,6.000e-5,418.7,16.9
...
```

| 列 | 意味 |
|----|------|
| `step` | 現在の学習ステップ |
| `loss` | 当該ステップでのバッチ平均 loss |
| `ema` | 指数移動平均 loss (α=0.05) |
| `min` / `max` | 直近 `log_every` ステップ内の最小/最大 loss |
| `lr` | 当該ステップでの学習率 |
| `ms_per_step` | 直近 `log_every` ステップでの 1 ステップあたり平均所要時間 (ミリ秒) |
| `elapsed_s` | 学習開始からの累積経過時間 (秒) |
