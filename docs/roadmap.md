# 今後の改善案 / ロードマップ

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

| 段階 | 項目 | 期待 val_ppl | 状態 |
|------|------|-------------|------|
| 5-1 | top-p (nucleus) sampling | 18.70 (変わらず、 体感品質改善) | 🚧 着手中 |
| 5-2 | max_len 256 → 512 | 17.5〜18.0 | 未着手 |
| 5-3 | コーパス拡大 1M → 5M+ char | 15.5〜17.0 | 未着手 |
| 5-4 | モデル拡大 d_model 384 → 512/768 | 12〜14 | 未着手 |

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

## 推論時のキャッシュ機構 (KV cache)
現在の `generate` は token 1 つ生成するたびに過去 token を含む全 context を attention で再計算。
KV cache (Q/K/V の中間結果を保持) を導入すれば 1 token 生成あたりの計算量が
`O(n)` → `O(1)` 近くまで下がる。 `max_len=128` 以上の生成で大きく効く。

## 生成制御の追加
- top-p (nucleus) sampling: [Phase 5-1](phase5.md#phase-5-1-top-p-nucleus-sampling) で着手中
- 最小生成 token 数 (`min_new_tokens`)
- bad words / banned ngrams フィルタ

## `layer_normalization`, `embedding` 等の Matrix 統一
これらは現在 `Vec<Vec<f32>>` をやりとりしており、 内部の API 境界で `Matrix::from_jagged`/
`to_jagged` 変換が走っている。 すべて `Matrix` で統一すれば変換コストが消える。
ただし計算ボトルネックではないので優先度は低い。
