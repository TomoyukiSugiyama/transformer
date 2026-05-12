# SoTA LLM との比較

> **最終更新: 2026 年 5 月** — フロンティアモデルは半年で大きく変わるため、
> あくまでこの時点のスナップショットとして読んでください。

## 本実装の立ち位置

本実装は **教育目的の Rust スクラッチ実装** で、 SoTA 性能を狙うものではありません。

| 軸 | 本実装 | SoTA (2026 年 5 月時点) |
|----|-------|----------------------|
| パラメータ規模 | 10M-50M | 数十 B〜1 T (LLaMA 3.1 405B, Claude/Gemini クラス) |
| ハードウェア | M1 Max CPU + Apple Accelerate / matrixmultiply | 数千 GPU (H100 / B200 / TPU v5p) |
| フレームワーク | なし (Rust + 自前 matmul + 自前 autograd) | PyTorch / JAX + Megatron / DeepSpeed |
| 学習コーパス | 1〜8 M char (青空文庫) | 数兆〜十数兆 token (CommonCrawl + web + code + filtering) |
| 目的 | Transformer 内部を 1 ファイル単位で追える形で示す | 商用品質の汎用言語モデル |

ただし **アーキテクチャの主要要素 (RMSNorm, SwiGLU, RoPE, Pre-Norm 等) は LLaMA / Gemma 系の
モダン構成と整合的** で、 構造的なギャップは限定的です。 規模・並列化・推論最適化が主な差。

## 比較表

| コンポーネント | このリポジトリの実装 | ソース | SoTA (LLaMA 3 / Claude Opus / Gemini 2.5 / DeepSeek-V3 等) | ギャップ分析 |
|--------------|--------------------|-------|----------------------------------------------------------|------------|
| **Attention 機構** | 標準 MHA (全 head 独立 W_Q/K/V/O) | [`multi_head_attention.rs`](../src/multi_head_attention.rs) | **GQA** (LLaMA 2/3, Gemini), **MQA**, **MLA** (DeepSeek-V3/R1) | GQA/MQA/MLA 未実装。 KV 圧縮なしで推論時 KV メモリが線形に増える |
| **位置エンコーディング** | RoPE (Su et al. 2021) + Sinusoidal 切替可 | [`rope.rs`](../src/rope.rs) | RoPE + **ABF** (LLaMA 3 base=500k) / **YaRN / NTK-aware** | base=10000 固定。 ABF / YaRN による長文脈外挿は未実装 |
| **正規化** | RMSNorm + LayerNorm 切替可、 Pre-Norm 構成 | [`root_mean_square_layer_normalization.rs`](../src/root_mean_square_layer_normalization.rs) | RMSNorm + Pre-Norm (LLaMA 3 / Gemma / Mistral) | ✅ **SoTA と同等** |
| **FFN 活性化** | SwiGLU (gate/up/down 3 行列) + GELU 切替可、 `d_ff_g = (d_ff × 2) / 3` | [`swiglu_feed_forward_network.rs`](../src/swiglu_feed_forward_network.rs) | SwiGLU (LLaMA 1/2/3, Gemma)、 GPT-4 は非公開 | ✅ **SoTA と同等**、 LLaMA 流の `d_ff_g` 縮減比に準拠 |
| **オプティマイザ** | AdamW (β1=0.9, β2=0.99-0.999, ε=1e-8, WD=0.1) | [`adam_w.rs`](../src/adam_w.rs) | AdamW + ZeRO / FSDP の分散 sharding | 分散最適化なし (シングルノード CPU のみ) |
| **Gradient Clipping** | `clip_grad_norm(grads, max_norm=1.0)` 実装済 | [`language_model.rs`](../src/language_model.rs) (`clip_grad_norm`) | norm clip (max_norm=1.0) が業界標準 | ✅ **同等** |
| **LR スケジューラ** | Warmup (線形) + Cosine Decay | [`lr_scheduler.rs`](../src/lr_scheduler.rs) | Cosine / Cosine restarts / **WSD** (Warmup-Stable-Decay, MiniCPM 系) | WSD / Restarts 未実装 (training run が短いので影響は小) |
| **Tokenizer** | BPE (byte-level, 4000 vocab) + Char-level (3720-5220 vocab) 切替 | [`bpe_tokenizer.rs`](../src/bpe_tokenizer.rs) / [`char_tokenizer.rs`](../src/char_tokenizer.rs) | SentencePiece / tiktoken (cl100k / o200k / Gemini), 32k-200k vocab | 語彙規模が小さい。 Byte-fallback / pre-tokenization rule なし |
| **Attention 計算** | 標準 Scaled Dot-Product (O(n²) memory) | [`multi_head_attention.rs`](../src/multi_head_attention.rs) | **FlashAttention 2/3** (IO-aware, O(n) memory, GPU 専用) | FlashAttention 未実装 (CPU では効果も限定的だが、 長 seq でメモリ逼迫) |
| **モデル規模** | 最大 ~50M params (d_model=768, n_layers=8) | [`main.rs`](../src/main.rs) `Config::aozora_meiji_taisho_d768_max512` | LLaMA 3: 8B-405B, Claude / Gemini: 数百 B-1 T | **桁違いのスケール差** (~3 桁) — 学習目的としては適切 |
| **Dropout** | Inverted dropout (train/eval 切替) | [`dropout.rs`](../src/dropout.rs) | LLaMA / Mistral 系では `dropout=0` (データ量で代替) | nanoGPT 準拠で適切 (小規模コーパスでは必要)。 Attention 内 dropout は未実装 |
| **KV Cache** | ✅ **実装済** (RoPE 適用後 K + 未回転 V を per-layer で保持) | [`kv_cache.rs`](../src/kv_cache.rs), [`docs/kv_cache.md`](kv_cache.md) | フロンティアモデル全てで必須 | ✅ **同等**。 per-token 計算量が `O(n²·d) → O(n·d)`。 残: KV truncation (sliding window) は未対応 |
| **MoE (Mixture of Experts)** | 未実装 | — | Mixtral, DeepSeek-V3, Gemini 2.5 (sparse MoE 8 of 64 等) | アーキ的に大きなギャップだが、 50M params 規模では本質的に不要 |
| **チェックポイント** | カスタムバイナリ (`best.bin` / `latest.bin` / `inference.bin`) | [`checkpoint.rs`](../src/checkpoint.rs) | **SafeTensors** (HF 標準), GGUF (llama.cpp 系) | HuggingFace エコシステム非対応。 互換 loader は今後の検討 |
| **並列化** | rayon (CPU 並列) + Apple Accelerate / matrixmultiply | [`docs/performance.md`](performance.md) | FSDP, Tensor Parallel, Pipeline Parallel (GPU 分散) | GPU 非対応、 シングルノード CPU のみ |
| **Autograd** | **手動実装** (全レイヤー backward 手書き) | 各 `*.rs` の `backward` 関数 | PyTorch / JAX の自動微分 | ❤️ **教育的価値が極めて高い**。 SoTA とは目的が異なる |
| **混合精度 / 量子化** | f32 のみ | [`matrix.rs`](../src/matrix.rs) | bf16 / fp8 (学習)、 int8 / int4 / nf4 (推論) | M1 Max の AMX が f32 向けに最適化されており、 mixed precision 効果は GPU ほど劇的でない |
| **サンプリング** | Greedy / **top-k** / **top-p (nucleus)** / repetition_penalty | [`output_head.rs`](../src/output_head.rs) | + **temperature scheduling**, beam search (古典), constrained decoding | ✅ 主要手法は実装済。 logit_bias や grammar-based decoding は未実装 |

## ギャップを埋めるための future work (優先度順)

[`roadmap.md`](roadmap.md) と整合的に並べた優先度:

1. ~~**KV cache**~~ ✅ 実装済 → [`kv_cache.md`](kv_cache.md)
2. **FlashAttention (Rust + Accelerate 流)** — 長 seq 対応のためのメモリ最適化。 IO-aware なら CPU でも効果あり
3. **GQA / MQA** — KV cache と組合せて KV メモリを `1/n_heads` に削減
4. **KV cache truncation (sliding window)** — `max_len` 超過時の継続生成
5. **YaRN / ABF (RoPE 拡張)** — 学習時 `max_len=512` を超える長さへの外挿能力
6. **SafeTensors 互換 (read-only)** — HuggingFace モデルを読み込めるようにし、 検証ベンチマーク (HellaSwag / ARC 等) を実装に流せるようにする
7. **bf16 サポート** — M1 系の bf16 AMX を使った matmul 高速化

**意図的に優先度を下げているもの**:
- **MoE**: 50M 規模では理論的にも実用的にも効果薄い
- **GPU 対応 / 分散最適化**: 学習用 OSS としての価値 (1 マシンで完結する読みやすさ) を毀損する
- **量子化**: 教育的価値が低く、 f32 で完結する方が読者に優しい

## まとめ

| カテゴリ | 評価 |
|---------|------|
| **モダン LLM の中核要素 (Norm / FFN / RoPE / Pre-Norm)** | ✅ SoTA と整合的に実装済 |
| **学習の安定化 (AdamW + Gradient Clip + Cosine LR)** | ✅ 標準的な手法は揃っている |
| **推論最適化 (KV cache / FlashAttention / 量子化)** | 🟡 KV cache 実装済、 FlashAttention・量子化は future work |
| **スケール (params / data / compute)** | ❌ 3 桁の差 — 学習用 OSS としては適切 |
| **エコシステム (SafeTensors / HF)** | ❌ 未対応 — 簡易 loader を今後検討 |
| **教育的価値 (Pure Rust / 1 ファイル単位の追跡性)** | ⭐⭐⭐ **本実装の差別化価値** |

「**Rust だけで** モダン LLM のコア技術を **autograd フレームワークなしで** 動作させる」 ことが
本実装の主目的。 SoTA との性能比較は意味を持たないが、 **アーキテクチャ知識の獲得** という意味で
読者にとっての価値は高い。
