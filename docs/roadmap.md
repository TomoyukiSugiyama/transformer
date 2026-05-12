# 今後の改善案 / ロードマップ

> 本実装と SoTA LLM (LLaMA 3 / Claude / Gemini / DeepSeek-V3) の要素別比較は
> [`sota_comparison.md`](sota_comparison.md) を参照。 ここで挙げる future work の位置付けが分かります。

## モダンアーキテクチャの導入 (Phase D) — 完了

[Phase 4b](phase4.md) で **コーパス拡大による val_ppl 改善が頭打ち** ( 18.15 → 18.98 ) になったことから、
モデル容量側を強化する方向で **LLaMA / GPT-NeoX 系で標準化された改良要素**を順次取り込んだ:

| 項目 | 効果 | 影響範囲 | 状態 |
|------|------|---------|------|
| **RMSNorm** | val_ppl -1.1% (18.98 → 18.77) / 過学習開始を 100 step 後ろ倒し / 速度差は出ず | `layer_normalization` を差し替え | ✅ 完了 ([Phase D-1](phase_d.md#phase-d-1-rmsnorm-vs-layernorm)) |
| **SwiGLU FFN** | val_ppl -0.4% (18.77 → 18.70) + ms/step **-35%** + best 到達 200 step 早期化 | `feed_forward_network` を差し替え | ✅ 完了 ([Phase D-3](phase_d.md#phase-d-3-swiglu-ffn)) |
| **RoPE** | floor 不変 (+0.7%) / early step **-10〜22%** の収束加速 / ms/step +4% | `MultiHeadAttention` 内に組込、 `pe` を Option 化 | ✅ 完了 ([Phase D-2](phase_d.md#phase-d-2-rope)) |
| **MQA / GQA** | 推論時 KV cache を 1/n_heads に圧縮 | `multi_head_attention` の K/V 次元 | 見送り (KV cache 未実装のため効果検証困難) |

**累積効果**: LN+GELU baseline → RMS+SwiGLU+RoPE で **val_ppl -1.5% / best 到達時間 -41%**。
詳細結果は [docs/phase_d.md](phase_d.md) を参照。

## 生成品質向上 (Phase 5) — 着手中

[Phase D](phase_d.md) でアーキテクチャ刷新が完了したが、 生成文の日本語が局所的にしか正しくない課題が残った。
スケール側の改善で対応する 4 段階の計画:

| 段階 | 項目 | 実測 / 期待 val_ppl | 状態 |
|------|------|-------------------|------|
| 5-1 | top-p (nucleus) sampling | 18.84 (変わらず、 体感品質は限定的) | ✅ 完了 |
| 5-2 | max_len 256 → 512 | **17.76 (実測, vs 4b -5.0%)** | ✅ 完了 |
| 5-3 | コーパス拡大 1M → 8.3M char (5 作家追加) | **18.71 (実測, 想定外悪化)** — モデル容量律速の反証 | ✅ 完了 |
| 5-4 | モデル拡大 d_model 384 → 512/768 (3 段階) | 13〜17 (5-3 比 -8〜30%) | 🚧 5-4a 着手 |

詳細は [docs/phase5.md](phase5.md) を参照。

## 過学習の更なる抑制
- **attention dropout の追加** ([Phase 3](phase3.md) で導入したのは residual 直前の 2 箇所のみ)。
  nanoGPT は softmax 後にも dropout を入れており、 これが本実装より約 650 step 遅く
  ピークが来る要因と推測される

## Linux / Windows 向け OpenBLAS / Intel MKL バックエンド
macOS の Accelerate と同じ構造 (`#[cfg(target_os = ...)]` 分岐) で `openblas-src` + 自前 extern、
もしくは `ndarray-linalg` 経由で OpenBLAS / MKL を呼べる。

## モデルサイズ拡大による AMX の本領発揮
現状の `d_model=256` では 1 つの matmul サイズが中規模で AMX の旨味が部分的。
`d_model=512〜768` に拡大すると matmul の比率も問題サイズも大きくなり、
Accelerate 単独効果が `1.7×` から **`2〜3×`** に伸びる見込み。

## 推論時のキャッシュ機構 (KV cache) ✅ 完了 (4.71x speedup @ d=512)

各 layer の `K` (RoPE 適用済) と `V` (未回転) を `KvCache` に保持し、
1 token 生成あたりの計算量を `O(n²·d) → O(n·d)` に削減。
詳細は [`docs/kv_cache.md`](kv_cache.md) を参照。

- 実装: `src/kv_cache.rs`, `MultiHeadAttention::forward_step`,
  `TransformerBlock::forward_step`, `Transformer::forward_step`,
  `LanguageModel::generate_{top_k,top_p}_with_cache`
- 学習パスは一切変更せず、 推論専用の並行 API として追加 (回帰リスクなし)
- no-cache vs with-cache の **logit が 1e-3 以内 / argmax 完全一致** を単体テストで保証
- Sinusoidal PE / RoPE 両方に対応
- **実測** (Phase 5-4a step_000800.bin, d=512, n_layers=6, M1 Max + Accelerate, 8 prompt 平均):
  - `max_new=100`: **4.71x** (no-cache 17.5 ms/tok → with-cache 3.7 ms/tok)
  - `max_new=200`: **6.55x** (no-cache 26.4 ms/tok → with-cache 4.0 ms/tok)
  - `max_new=400`: **11.13x** (no-cache 45.3 ms/tok → with-cache 4.1 ms/tok)
- **with-cache の per-token 時間は max_new に対し ほぼ定数 (~4 ms)** = KV cache が理論通り機能 ✅
- 絶対 speedup は max_new に **線形に伸びる** (n が大きいほど効果絶大)。
  4 ms の固定コストは `Vec<Vec<f32>>` 変換 + m=1 BLAS dispatch + RoPE + alloc

### KV cache 追加最適化 (Phase 6 候補、 後回し)

> 現行 4.71x は実用上十分なので Phase 5 の優先タスクからは外す。
> 以下は将来 「推論速度がボトルネックになった時」 に着手する。

実装優先順 (推定効果の合計で 4.71x → 20-30x まで伸ばせる見込み):
- ★ **Norm/FFN に `forward_one(&[f32])`** 追加 (`Vec<Vec<f32>>` 変換を回避、 1.5-2x 追加期待) — 主犯
- **KvCache pre-allocated buffer 化** (append の memcpy 不要に、 1.1-1.2x)
- **per-head attention BLAS 化** (`(1, d_h) × (d_h, n)`、 1.1-1.2x)
- **QKV projection 融合** (`(1, d) × (d, 3d)`、 副次効果)
- **Prefill batch forward** (現状: prompt N-1 token を逐次 forward_step)
- **KV cache truncation (sliding window)** で `max_len` 超過時の継続生成

## 生成制御の追加
- ~~top-p (nucleus) sampling~~ → [Phase 5-1](phase5.md#phase-5-1-top-p-nucleus-sampling) で実装完了
- 最小生成 token 数 (`min_new_tokens`)
- bad words / banned ngrams フィルタ

## `layer_normalization`, `embedding` 等の Matrix 統一
これらは現在 `Vec<Vec<f32>>` をやりとりしており、 内部の API 境界で `Matrix::from_jagged`/
`to_jagged` 変換が走っている。 すべて `Matrix` で統一すれば変換コストが消える。
ただし計算ボトルネックではないので優先度は低い。
