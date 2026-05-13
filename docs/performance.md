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

## Phase 7: 大規模高速化 (2026-05 〜)

Phase 6-c で 1 step ~11,000 ms (= 3000 step で約 9.2 h) になり、
Apple GPU/CUDA 移植を避けつつ **CPU だけで 2-3x の改善** を狙うのが Phase 7 の目的。

### 取り組み済み

#### B4: WSD (warmup-stable-decay) スケジューラ

`src/lr_scheduler.rs` に `LrScheduleKind::WarmupStableDecay { stable_steps }` を追加。

- 既存の cosine と挙動互換 (`Config::lr_schedule_kind` のデフォルトは `WarmupCosine`)
- decay 区間は MiniCPM 流の `1 - sqrt(progress)` を採用
- 同じ BPC を 15-30% 早く到達できる場合があり、 さらに total_steps を伸ばしても
  過去 step の lr 軌道が変わらないため checkpoint からの追加学習に向く

#### B1: layer trait の Matrix 化

旧 trait は `forward(&[Vec<f32>]) -> Vec<Vec<f32>>` で受け渡していたため、
内部で Matrix 化していた層 (FFN/MHA/OutputHead) も毎回 `from_jagged` / `to_jagged` で
**メモリコピー + 再アロケーション** が走っていた。 Phase 7 ではすべての主要層に
`forward_matrix(&Matrix) -> Matrix` を追加し、 オーケストレーション層
(TransformerBlock → Transformer → LanguageModel) を Matrix 直叩き経路に切り替えた。

| 対象 | 旧 | 新 (Matrix 直叩き) |
|------|-----|-------------------|
| `LayerNormalization` | 行ループ + jagged | flat row-major + rayon `par_chunks_mut` |
| `RootMeanSquareLayerNormalization` | 同上 | 同上 |
| `FeedForwardNetwork` (GELU) | Matrix 内部 + 境界 jagged | Matrix 直叩き、 境界変換ゼロ |
| `SwiGluFeedForwardNetwork` | 同上 | 同上 |
| `Dropout` | 行 × 列の二重 Vec mask | flat mask (`Vec<f32>`) |
| `Embedding` | jagged 行ベース lookup | flat row-major lookup |
| `MultiHeadAttention` | Matrix 内部 + 境界 jagged | Matrix 直叩き |
| `OutputHead` | Matrix 内部 + 境界 jagged | Matrix 直叩き |
| `SinusoidalPE` | 行ループ | flat 加算 |
| `CrossEntropyLoss::forward_sequence_matrix` | jagged grad 出力 | flat grad 出力 |

旧 API (`forward(&[Vec<f32>])`) は **Phase 7-2 で完全削除** (下記「Phase 7-2: 旧 API クリーンアップ」参照)。

#### B2: QKV projection 融合

旧: `Q = X W_Q`, `K = X W_K`, `V = X W_V` の **3 回の matmul**。
新: `QKV = X W_QKV` の **1 回の matmul** (W_QKV: `(d_model, 3*d_model)`) → `split_columns(3)` で分割。

- BLAS 呼び出しのオーバーヘッドが 3 → 1 に減る
- Apple Accelerate の sgemm は出力行列が大きいほど効率が上がる傾向があるため有利
- チェックポイント形式は **w_q/w_k/w_v に分割保存** したまま (Phase 6-c までの best.bin と完全互換)
- Adam optimizer の moment は `w_qkv` 1 つに統一 (新 run のみ)

### ベンチマーク結果

`#[test] bench_phase7_step_time` (`cargo test --release bench_phase7_step_time -- --nocapture --ignored`) で
Phase 6-c と同じ形状 (d_model=512, n_heads=8, n_layers=6, max_len=1024) を batch_size=2 で 3 step 計測。

| 実装 | per-step (batch=2) | Phase 6-c 換算 (batch=16) | speedup |
|------|--------------------|---------------------------|---------|
| Phase 6-c (実測) | ~1,375 ms (推定) | **~11,000 ms** | 1.0x |
| B1 (Matrix 直叩き) | ~1,150 ms | ~9,200 ms | **~1.20x** |
| B1 + B2 (QKV 融合) | ~1,190 ms | ~9,500 ms | **~1.16x** |

> 注: ベンチ実行中は Phase 6-c も並列で走っており CPU 競合があるため、 上の数値は **保守的な下限**。
> また bench は `vocab_size=64` (Char tokenizer) なので、 Phase 6-c の `vocab=8000` より OutputHead matmul が
> 軽い。 同条件で測れば実際の speedup はもう少し大きい (推定 1.3-1.4x)。

### Phase 7-2: 旧 API クリーンアップ + KV-cache 推論パス Matrix 化 (✅ 完了)

Phase 7 第一弾は新旧 API を二重に持つ「移行期」状態。 各層に `forward(&[Vec<f32>])` (旧) と
`forward_matrix(&Matrix)` (新) が並ぶことで dead-code 警告が **14 個** 常駐し、 新規メソッドの
命名規約も曖昧になっていた。 Phase 7-2 で以下を整理:

#### 1. KV-cache 推論パス (`forward_step`) を Matrix 直叩き化

旧:
```rust
// transformer_block::forward_step (KV cache, 1 token)
let single = vec![x_new.to_vec()];           // jagged 1-row
let norm1 = self.norm1.forward(&single);     // 旧 jagged forward
let attn_out = self.mha.forward_step(&norm1[0], cache);
let attn_dropped = self.drop_attn.forward(&[attn_out]);
// ... ベクトル演算で residual ...
```
新:
```rust
let x_m = Matrix::from_flat(x_new.to_vec(), 1, d);   // 1-row Matrix
let norm1 = self.norm1.forward(&x_m);                 // Matrix forward
let attn_out_vec = self.mha.forward_step(norm1.row(0), cache);
let attn_out_m = Matrix::from_flat(attn_out_vec, 1, d);
let attn_dropped = self.drop_attn.forward(&attn_out_m);
let mut x2 = x_m; x2.add_in_place(&attn_dropped);     // Matrix の add
// ... 以降も Matrix のまま ...
```

per-token 推論で発生していた jagged ↔ Matrix 変換 (3 ペア × n_layers) を排除。

#### 2. 旧 API `forward(&[Vec<f32>])` / `backward(&[Vec<f32>])` を全削除

| 削除対象 | 削除メソッド数 |
|----------|---------------|
| `Normalization` trait + LayerNorm + RMSNorm | 6 |
| `FeedForward` trait + GELU + SwiGLU | 6 |
| `Dropout` | 2 |
| `Embedding` (forward / backward jagged) | 2 |
| `SinusoidalPE` (forward jagged) | 1 |
| `OutputHead` (forward / backward jagged) | 2 |
| `MultiHeadAttention` (forward / backward jagged) | 2 |
| `TransformerBlock` (forward / backward jagged) | 2 |
| `Transformer` (forward / backward jagged) | 2 |
| `LanguageModel::forward_ids` (dead code) | 1 |
| `CrossEntropyLoss::forward_sequence` (jagged 版) | 1 |
| **合計** | **27 メソッド削除 (LOC -300 程度)** |

#### 3. `*_matrix` → bare 名にリネーム

`forward_matrix` → `forward`、 `backward_matrix` → `backward`、
`forward_sequence_matrix` → `forward_sequence`。 移行期サフィックスがなくなり API がシンプルに。

#### 4. テスト書き換え

数値勾配チェック (rms / swiglu) と挙動テスト (dropout) を Matrix API 経由に書き換え。
入力ジャグドは `Matrix::from_jagged(&x)` でラップ、 出力比較は `y.row(i)[j]` で参照。

#### 効果

- **dead-code 警告 14 → 0** (本物の警告が埋もれない)
- **LOC -300〜-500 行** (重複 API 削除)
- **KV-cache 推論側も Matrix 直叩き化** = per-token 1 行ぶんの jagged ↔ Matrix 変換が消える
  (推論速度改善効果は数 % 程度を予想、 実測は Phase 6-d 完走後)
- **API 一本化** = 新メソッド追加時に「`_matrix` 付ける?」の判断不要

### Phase 7-3: 転置行列 materialize の排除 (✅ 完了)

backward 中の `_.transpose().matmul(_)` 17 箇所を BLAS の trans フラグ経由に置換し、
**転置行列のメモリアロケーションとコピー** を完全に排除する。

#### 1. 問題

backward では下記のような形が頻出:

```rust
// W2 の grad: cache_a^T @ dl_dz2
let g_w2 = self.cache_a.transpose().matmul(dl_dz2);
// dL/dx: dl_dz1 @ W1^T
let dl_dx = dl_dz1.matmul(&self.w1.transpose());
```

`cache_a.transpose()` や `w1.transpose()` で **転置済みの行列を新規アロケート** していた。
特に `OutputHead` の `W^T` (vocab=8K, d=512 → 16 MB) や `MHA` の `W_QKV^T` (d×3d → 6 MB)
は重い。

#### 2. 対応

`Matrix` に下記 2 メソッドを追加:

```rust
pub fn matmul_t1(&self, other: &Matrix) -> Matrix;  // self^T @ other
pub fn matmul_t2(&self, other: &Matrix) -> Matrix;  // self @ other^T
```

内部実装は **`cblas_sgemm` の `CblasTrans` フラグ** (もしくは `matrixmultiply` の row/col stride
入れ替え) を使い、 転置メタデータだけで処理する。 メモリ上の転置コピーは一切発生しない。

#### 3. 置換対象 (17 箇所)

| ファイル | 箇所 |
|---------|------|
| `multi_head_attention.rs` | W_O, W_QKV, V, K, S の各 backward + scaled-dot-product-attention の Q@K^T (計 7) |
| `feed_forward_network.rs` | W2, W1 の backward (計 4) |
| `swiglu_feed_forward_network.rs` | W_down, W_gate, W_up の backward (計 6) |
| `output_head.rs` | W (vocab × d) の backward (計 2) |

#### 4. 効果

bench (`bench_phase7_step_time`, batch=2, max_len=1024) 実測:

| 段階 | per-step | speedup vs 7-2 |
|------|---------:|---------------:|
| Phase 7-2 (transpose 残存) | ~1500 ms | 1.00x |
| **Phase 7-3 (transpose 排除)** | **~1352 ms** (3 run avg: 1336/1339/1381) | **1.10-1.12x** |

実訓練 (batch=16) 換算では Phase 7-2 ~9000 ms → Phase 7-3 ~8100 ms (10% 短縮、 6-d の
6.4 h → ~5.7 h を期待)。

副次効果として **per-step メモリアロケが ~50-100 MB 減少** (transpose 用一時バッファ消滅) し、
GC/malloc 圧も軽くなる。

### 今後 (検討中)

- **B3 Flash Attention 風 (CPU online softmax + matmul 融合)**:
  attention scores 行列 (`seq² × n_heads × n_layers` = 192 MB @ Phase 6-c) のメモリ I/O を削減。
  実装難度は中、 期待 1.3-2x (attention 部分のみ)。
- **アロケーション削減 (Phase 7-4 候補)**: `split_columns` / `concat_columns` の `_into` API
  と pre-allocated buffer 化で 1.05-1.15x。 残る big alloc は MHA 内の `concat`、 `dl_dqkv` (6MB)、
  per-head `scores`/`dl_dp`/`dl_ds` (各 4MB × 8 head × 6 layer = 576 MB)。
- **bf16 mixed precision (Apple BNNS)**: 期待 1.8-2.5x、 実装難度大。 数値安定性試験要。
- **KvCache の事前確保**: 推論時のみ。 alloc/realloc を削除。
- **Embedding / SinusoidalPE / Rope の table を Matrix 化**: 残った `Vec<Vec<f32>>` の内部表現。
  Matrix にすれば row lookup が `&[f32]` slice で済む (現状 `Vec<f32>` の clone が必要なケースあり)。

