# 並列化と性能

## ベンチマーク (M1 Max / 10 コア)

`d_model=256, n_heads=8, d_ff=1024, n_layers=4, max_len=128, batch_size=16, vocab_size=4000`
構成での 1 step あたり所要時間:

| 実装 | 1 step (steady) | vs 前段 | vs 初期 (推定) | 出典 |
|------|----------------|--------|---------------|------|
| ① Vec<Vec<f32>> + シリアル | ~30 s | — | 1× | 推定 (実効 ~0.7 GFLOP/s × 全 step ~20 GFLOPs) |
| ② Vec<Vec<f32>> + rayon | ~3.0 s | **10×** | 10× | Phase 2 (checkpoint mtime) |
| ③ Matrix + rayon naive | ~3.0 s | 1.0× | 10× | `phase2_d256_ff1024_max128_before_blas.log` |
| ④ Matrix + matrixmultiply | ~1.29 s | **2.3×** | 23× | `phase2_d256_ff1024_max128_with_blas.log` |
| ⑤ Matrix + Apple Accelerate (AMX) | **~0.745 s** | **1.7×** | **約 40×** | `phase2_d256_ff1024_max128_with_accelerate.log` |

10000 step 学習が **約 8 時間 → 約 2 時間** (Accelerate) に短縮。 ① の数値は実測ではなく
シングルスレッド `Vec<Vec<f32>>` matmul の経験則 (実効 0.5〜1 GFLOP/s) からの外挿。

長時間負荷では M1 Max の thermal throttling で初期 660 ms → 定常 745 ms に落ち着く。
短時間ベンチでは更に速い値が出る。

## `Matrix` の役割

行列演算は `crate::matrix::Matrix` に集約されている。
内部表現は **行優先 (row-major) の flat `Vec<f32>`** で、 jagged な `Vec<Vec<f32>>` ではない。

主な API:
- 構築: `zeros`, `from_jagged`, `from_flat`
- 演算: `matmul`, `transpose`, `add_in_place`, `add_row_bias_in_place`,
  `sum_rows_into_cols`, `map`, `elementwise_with`, `softmax_rows_in_place`
- MHA 用: `split_columns(n)` / `concat_columns(&[Matrix])`

flat 表現により、 BLAS への ptr/stride 渡しが **ゼロコピー** (`leading_dim = cols`)。

## `matmul` の OS 別バックエンド

```rust
#[cfg(target_os = "macos")]
unsafe { accelerate::cblas_sgemm(...) }   // AMX 自動活用

#[cfg(not(target_os = "macos"))]
unsafe { matrixmultiply::sgemm(...) }     // pure-Rust SIMD カーネル
```

- **macOS**: Apple Accelerate Framework の `cblas_sgemm` を `extern "C"` + `#[link]` で直接呼び出す。 M1/M2/M3 系では行列サイズに応じて **AMX co-processor** が自動選択される。
- **その他 OS**: `matrixmultiply` クレート (pure-Rust の SIMD/キャッシュタイル実装) にフォールバック。

`matmul_naive` (rayon `i-k-j`) も保持しており、 `matmul_blas_matches_naive_for_random_matrices`
テストで両者の数値一致 (浮動小数誤差 ≤ `1e-3 × k`) を検証している。

## rayon で並列化されている処理

| 処理 | 並列粒度 |
|------|---------|
| `matrix::matmul_naive` | 出力行 (`par_chunks_mut`) |
| `matrix::transpose` | 出力行 |
| `matrix::add_in_place` | 要素 |
| `matrix::softmax_rows_in_place` | 行 |
| `output_head::logits_last` | vocab 次元 |
| `cross_entropy_loss::forward_sequence` | token |

`matmul` 本体は rayon 不要 (BLAS バックエンドが内部でマルチコア活用)。

## Matrix を経由する主要処理

- `multi_head_attention`: Q/K/V/O 射影、 scaled-dot-product attention、 head 分割・結合
- `feed_forward_network`: forward / backward すべて、 bias 加算、 GELU、 勾配集計
- `output_head`: forward / backward すべて、 `logits_last` (単一トークン推論最適化)

## Matrix 非経由 (軽量で並列化対象外)

- `embedding`: token id ベースの lookup
- `layer_normalization`: 行ごとの統計量計算 (d_model 方向のみで小規模)
- `sinusoidal_pe`: 加算のみ

## AdamW との橋渡し

`AdamW::step_matrix_flat(&mut self, &str, &mut Matrix, &Matrix)` が `Matrix` を直接受け取る。
内部の `AdamWParam.data` も `Vec<f32>` (flat) なので、 jagged ↔ flat の **flatten 変換コストはゼロ**。
