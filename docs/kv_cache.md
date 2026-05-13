# KV Cache (Key-Value Cache) の実装

> **目的**: 自己回帰生成 (autoregressive decoding) の per-token 計算量を
> `O(n²·d)` → `O(n·d)` に削減する。

![Inference Pipeline with KV Cache](inference-pipeline.png)

上図は `language_model.rs` + `kv_cache.rs` で構成される推論パイプライン全体。 左ペインが checkpoint ロードから生成テキスト出力までのフロー、 右ペインが per-layer の `KvCache` の内部構造を示す。 以下は各要素の詳細解説。

## なぜ KV cache が必要か

通常の forward は **「prompt + 既生成 token 全部」** を入力として、 全位置の attention を
毎 step 計算しなおしている:

```
step 1: forward([t0, t1, t2])          → logits[2] → t3
step 2: forward([t0, t1, t2, t3])      → logits[3] → t4
step 3: forward([t0, t1, t2, t3, t4])  → logits[4] → t5
...
step k: forward([t0..t_{k+2}])         → logits[k+2] → t_{k+3}
```

この方式では `O(n²·d)` で計算が膨らむ。 しかし **過去 token の K, V は変わらない** ため、
キャッシュしておけば各 step で **新規 token 1 つだけ** の K, V を計算するだけで済む。

## アーキテクチャ

### 1. `KvCache` ([`src/kv_cache.rs`](../src/kv_cache.rs))

各 attention 層につき 1 つ持つ:

```rust
pub struct KvCache {
    k: Matrix,        // (cur_len, d_model), RoPE 適用後
    v: Matrix,        // (cur_len, d_model), 未回転
    cur_len: usize,
    capacity: usize,  // = max_len
    d_model: usize,
}
```

**重要**: K は **RoPE 適用後** を保管する。 これにより、 過去 K は append 時の位置で
すでに回転されているので再回転不要。 V は RoPE が掛からないのでそのまま。

### 2. `MultiHeadAttention::forward_step` ([`src/multi_head_attention.rs`](../src/multi_head_attention.rs))

1 token 分の hidden state `x_new` を受け取って:

1. Q, K, V を投影 (各 `(1, d_model)`)
2. RoPE: 新規 Q, K を **位置 `cache.cur_len()`** で head ごとに回転
3. 新 K (回転済), V を `cache.append()` で追記 (`cur_len → cur_len + 1`)
4. Attention: `q_new` (1 row) を cache の全行 K, V に対して計算 (causal mask 不要)
5. 出力 projection

**学習用 cache (`self.cache_*`) は触らない** ので、 学習中の validation でも安全。

### 3. `TransformerBlock::forward_step` ([`src/transformer_block.rs`](../src/transformer_block.rs))

Pre-Norm 構成を 1 token 分実行:

```
norm1 → mha.forward_step(x_new, cache) → drop_attn → residual_add(x_new) → x2
→ norm2 → ffn → drop_ffn → residual_add(x2)
```

`norm` / `ffn` / `dropout` は既存の `forward(&[Vec<f32>])` を 1-row Vec で呼び出して再利用。
これらの内部 cache は上書きされるが、 backward は呼ばれない前提なので無害。

### 4. `Transformer::forward_step` ([`src/transformer.rs`](../src/transformer.rs))

各 layer に対応する `&mut [KvCache]` を渡して順に流す:

```rust
pub fn forward_step(&mut self, x_new: &[f32], caches: &mut [KvCache]) -> Vec<f32>
```

### 5. `LanguageModel::generate_*_with_cache` ([`src/language_model.rs`](../src/language_model.rs))

3 種類実装:
- `generate_top_k_with_cache(prompt, max_new, k, temp, rep_penalty)`
- `generate_top_p_with_cache(prompt, max_new, p, temp, rep_penalty)`
- `generate_greedy_with_cache(prompt, max_new)` (検証用)

実行フロー:
1. `transformer.init_kv_caches(max_len, d_model)` で layer 数分の cache を作成
2. **Prefill**: prompt の最後の token を除いて順に `forward_step_last` で流し、 cache を構築
3. **Generation loop**: 直前 token を `forward_step_last` に渡して logits を取得 → サンプリング → append

## 計算量・速度の見積もり

| 段階 | no-cache (per step at len n) | with-cache (per step at len n) | 速度比 |
|------|-----------------------------|-------------------------------|-------|
| Q/K/V projection | `O(n · d²)` | `O(d²)` | n |
| Attention scores | `O(n² · d)` | `O(n · d)` | n |
| Attention output | `O(n² · d)` | `O(n · d)` | n |
| Output projection | `O(n · d²)` | `O(d²)` | n |
| FFN | `O(n · d · d_ff)` | `O(d · d_ff)` | n |
| **合計 per step** | `O(n · d² + n² · d)` | `O(d² + n · d)` | **≈ n** (大規模 n で) |

## 実測 (Phase 5-4a step_000800.bin / **d=512**, n_layers=6, M1 Max + Accelerate)

8 prompt 平均で `max_new_token` を変えてベンチ (`bench_kv_cache`):

| max_new | no-cache total | with-cache total | speedup | **no-cache ms/tok** | **with-cache ms/tok** |
|---------|---------------|------------------|---------|---------------------|------------------------|
| 100 | 14.0 s | 2.97 s | 4.71x | 17.5 | **3.7** |
| 200 | 42.3 s | 6.47 s | 6.55x | 26.4 | **4.0** |
| 400 | 145.0 s | 13.03 s | **11.13x** | 45.3 | **4.1** |

### 重要な観察

**with-cache の per-token 時間は max_new_token によらずほぼ一定 (~4 ms)**。
これは KV cache が **理論通りに機能している** ことの直接的な証拠:

- **no-cache**: per-token cost が `O(n)` で線形増加 (17.5 → 26.4 → 45.3)
- **with-cache**: per-token cost が constant (3.7 → 4.0 → 4.1) — 微増は attention の `O(n)` 項

その結果、 **絶対 speedup は max_new_token に比例して伸びる**:

| max_new | 実測 speedup | 理論上限 (avg_seq_len) | 実効率 |
|---------|-------------|-----------------------|-------|
| 100 | 4.71x | ~50x | 9% |
| 200 | 6.55x | ~100x | 7% |
| 400 | 11.13x | ~200x | 6% |
| 800 (予測) | ~18x | ~400x | — |
| 1600 (予測) | ~30x | ~800x | — |

理論上限との比は 6-9% と低いが、 **重要なのは線形 → 定数の漸近挙動が達成されていること**。

### per-token ~4 ms を構成する固定コストの内訳 (推定)

with-cache が n に依存しないということは、 4 ms は以下の **n 非依存な部分** で構成されている:

| 順位 | コンポーネント | 推定 ms |
|------|---------------|---------|
| 1 | `Vec<Vec<f32>>` ⇄ `Matrix` 変換のヒープ確保 (norm/ffn を 1-row Vec で呼ぶたび) | 1.5-2.0 |
| 2 | per-layer の matmul (Q/K/V/O + FFN gate/up/down) で `m=1` の BLAS dispatch overhead | 1.0-1.5 |
| 3 | RoPE の per-head 回転 + concat | 0.3 |
| 4 | output_head の logits projection | 0.3 |
| 5 | `KvCache::append` の memcpy + scalar attention loop | 0.3 |

### 適用範囲の判断

| 用途 | 推奨 |
|-----|------|
| 短文生成 (max_new ≤ 100) | 4-5x で十分実用的、 そのまま使う |
| 中長文生成 (max_new = 200-400) | 6-11x、 体感で大幅な改善 |
| 長文生成 (max_new ≥ 800) | 18x+ 期待、 追加最適化なしでも十分 |
| max_len ぎりぎりの生成 | sliding window 未実装なので注意 (early-stop) |

### 残り最適化候補 (KV-cache 第 2 弾、 4.71x → 20-30x の見込み)

> 4.71x は max_new=100 で per-prompt 1.4 秒短縮 = 実用上十分な改善。
> 追加最適化は 「将来推論速度が再びボトルネックになった時」 に着手する。
> ※「Phase 6」 という名称は **モデル訓練フェーズ** と衝突するため (現行 Phase 6 は CharBPE 化)、
>   ここでは番号を外して 「KV-cache 第 2 弾」 と呼ぶことに改めた。

優先度を **再推定後のボトルネック順** に並べ直すと:

1. ★ **Norm / FFN に `forward_one(&[f32]) -> Vec<f32>`**  
   `Vec<Vec<f32>>` ⇄ `Matrix` 変換を完全に回避。 ※ Phase 7-2 (Matrix API 一本化) で
   forward は完全 Matrix 化済みなので、 残るのは単一 token 用の専用パスの追加。
2. **`KvCache` を pre-allocated buffer 化**: append が末尾追記だけに (memcpy 不要)
3. **per-head attention を BLAS 化** (`(1, d_h) × (d_h, n)`)
4. **QKV projection 融合**: `(1, d) × (d, 3d)` の 1 matmul ※ Phase 7-1 で **学習側は実装済**
   (`MultiHeadAttention::forward`)、 推論 `forward_step` も同様に融合済 (Phase 7-2)
5. **prefill を 1 回の forward で**: 短プロンプト (10 token 以下) ではほぼ無視可

これらは [`roadmap.md`](roadmap.md) で「KV-cache 追加最適化」として保留。

## 正当性検証

[`src/language_model.rs` の `kv_cache_tests`](../src/language_model.rs) で 4 つの単体テスト:

1. **`greedy_with_cache_matches_no_cache_rope_rms_swiglu`**: RoPE + RMSNorm + SwiGLU 構成で
   no-cache greedy と with-cache greedy が **同じ生成文字列** を出すことを確認
2. **`greedy_with_cache_matches_no_cache_sinusoidal_layernorm_gelu`**: Sinusoidal + LN + GELU 構成で同様
3. **`greedy_with_cache_matches_no_cache_rope_layernorm_gelu`**: RoPE × LN × GELU の混在組合せ
4. **`forward_step_last_logits_match_forward_ids_last_per_position`**: 各位置で logit ベクトルが
   `1e-3` 以内、 argmax は完全一致することを per-token で確認

`kv_cache::tests` の 4 つを合わせて **計 8 テスト 全 pass**。

## RoPE との相互作用

RoPE は位置 `pos` で K, Q を回転する。 KV cache では:
- 新規 K を append 時に **位置 `cur_len` で回転** してから cache に入れる
- 過去 K はその時点 (`cur_len_old`) で回転済 → 取り出してそのまま attention 計算に使える
- 新規 Q も **位置 `cur_len`** で回転する (一時値、 cache には入れない)

これは「位置回転は累積しない」 という RoPE の仕様 (`R(θ_a)·R(θ_b) ≠ R(θ_{a+b})` を Q, K の
**両方** に適用するから内積が `(m-n)` のみに依存) と整合的。

`Rope::apply_at_position(row: &mut [f32], pos: usize)` を新設して、 単一行を任意位置で
回転できるようにした (既存の `apply_in_place(matrix)` は行 index = 位置を仮定するため使えない)。

## 制約と将来課題

| 項目 | 現状 | 将来 |
|------|------|------|
| `cur_len` が `max_len` に到達 | early-stop | sliding window で末尾 `max_len` 個に切詰め |
| Sinusoidal PE の long-seq 外挿 | `max_len` 固定 | 実用上は RoPE を使うので低優先 |
| FlashAttention | 未対応 | KV cache とは独立に効く最適化 |
| MQA / GQA | 未対応 | KV cache 容量を `1/n_heads` に削減できる、 容量がボトルネックになったら検討 |

## 既存推論パスとの関係

`fn infer()` ([`src/main.rs`](../src/main.rs)) は **デフォルトで cache 版** を使うように切替済:
- `generate_top_k_with_cache(...)` (旧 `generate_top_k`)
- `generate_top_p_with_cache(...)` (旧 `generate_top_p`)

旧 no-cache 版 (`generate_top_k`, `generate_top_p`) は **互換性のため残してある**
(テストで使用、 また将来のベンチマーク比較で有用)。

ベンチマーク用 helper として `bench_kv_cache(model, prompts)` を追加。
学習完了後の checkpoint に対して呼ぶと no-cache と with-cache の per-prompt 速度を出力する。
