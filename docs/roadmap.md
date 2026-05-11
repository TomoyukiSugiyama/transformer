# 今後の改善案 / ロードマップ

## モダンアーキテクチャの導入 (Phase D)

[Phase 4b](phase4.md) で **コーパス拡大による val_ppl 改善が頭打ち** ( 18.15 → 18.98 ) になったことから、
モデル容量側を強化する方向が次の改善余地。 GPT-2 (2019) ではなく **LLaMA / GPT-NeoX 系で
標準化された改良要素**を順次取り込む計画:

| 項目 | 効果 | 影響範囲 | 状態 |
|------|------|---------|------|
| **RMSNorm** | val_ppl -1.1% (18.98 → 18.77) / 過学習開始を 100 step 後ろ倒し / 速度差は出ず | `layer_normalization` を差し替え | ✅ 完了 ([Phase D-1](phase_d.md#phase-d-1-rmsnorm-vs-layernorm)) |
| **SwiGLU FFN** | 同 params で val_ppl 0.5〜2% 改善が期待 | `feed_forward_network` を差し替え | ✅ 実装完了 / 学習評価予定 ([Phase D-3](phase_d.md#phase-d-3-swiglu-ffn)) |
| **RoPE** | 位置情報を相対化、 max_len 拡張時の汎化が向上 | `sinusoidal_pe` を撤去、 MHA 内に組込 | 未着手 |
| **MQA / GQA** | 推論時 KV cache を 1/n_heads に圧縮 | `multi_head_attention` の K/V 次元 | 未着手 |

RMSNorm / SwiGLU は他レイヤーへの影響が小さく独立に検証できる。 RoPE は positional encoding を
撤去するためモデル全体への影響が大きく、 最後に導入予定。 詳細結果は [docs/phase_d.md](phase_d.md) を参照。

## 過学習の更なる抑制
- **attention dropout の追加** ([Phase 3](phase3.md) で導入したのは residual 直前の 2 箇所のみ)。
  nanoGPT は softmax 後にも dropout を入れており、 これが本実装より約 650 step 遅く
  ピークが来る要因と推測される
- コーパス拡大 (Phase 4b: 漱石 7 作品 1.08M char → 鴎外・芥川 等を加えて 5M+ char へ)。
  ただし char-level + d_model=384 の現構成では既に容量律速気味なので、 モデル拡大と組合せが必要

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
- top-p (nucleus) sampling
- 最小生成 token 数 (`min_new_tokens`)
- bad words / banned ngrams フィルタ

## `layer_normalization`, `embedding` 等の Matrix 統一
これらは現在 `Vec<Vec<f32>>` をやりとりしており、 内部の API 境界で `Matrix::from_jagged`/
`to_jagged` 変換が走っている。 すべて `Matrix` で統一すれば変換コストが消える。
ただし計算ボトルネックではないので優先度は低い。
