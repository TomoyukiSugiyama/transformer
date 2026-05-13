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
| 5-4a | モデル拡大 d_model 384 → **512** (~20M params) | **18.16 (実測, 5-3 比 -2.9%, BPC 4.18, char tokenizer 系で最高)** | ✅ 完了 |
| 5-4b/c | モデル拡大 d=512+L8 / d=768 | 13〜17 | 📋 判断保留 (Phase 6 完了後に再検討) |

> 注: Phase 6-a (CharBPE 8K) で **BPC 3.76 (Phase 5-4a 比 -10.0%)** に更新され、 全 phase 通算で最高 (full-corpus chars/token=1.65 基準)。 Phase 5-4a は 1 token=1 char のため当時の絶対 val_ppl 比較では最高だった。 Phase 6-d は同 BPC 計算法で **3.74** (Phase 6-a 比 -0.6%、 微改善) に到達したが、 期待した -2 〜 -5% は達成できなかった (詳細は [phase6.md](phase6.md#vs-phase-6-a-charbpe-8k-max_len512-比較))。

詳細は [docs/phase5.md](phase5.md) を参照。

## トークナイザ刷新 (Phase 6) — 6-a / 6-d 完了

Phase 5-4a 完了後、 質的課題 (bigram 切り誤り、 短コンテキスト、 文体一貫性) を **トークナイザ側** で改善するアプローチ。

| 段階 | 項目 | 期待 / 実測 | 状態 |
|------|------|------|------|
| 6-a | Unicode char-level BPE (vocab 8K) | **実測 BPC 3.76 (Phase 5-4a 比 -10.0%, full-corpus chars/token=1.65 基準)、 実効 context ~840 char** | ✅ 完了 |
| 6-b | Unicode char-level BPE (vocab 16K) | 起動時実測 chars/token=1.72 (期待 2.5 の半分以下)、 ROI 不足 → 6-c に振替 | ⛔ 中断 |
| 6-c | vocab=8K のまま max_len 512 → 1024 | step 180 で per-step 11,000 ms 確認、 9.2 h 想定で **Phase 7 高速化に着手** → 6-d に振替 | ⛔ 中断 |
| **6-d** | **6-c と同形状 + WSD scheduler + Phase 7-1/7-2 バイナリ** | **best step 2800、 val_loss 4.274、 val_ppl 71.81、 BPC 4.22 (val 基準)、 total 7h 41min** | ✅ 完了 |

実装: `src/char_bpe_tokenizer.rs` (新規、 byte-level の既存 BPE は英語用に保持)。 詳細・最終結果は [docs/phase6.md](phase6.md#phase-6-a-結果--完了) を参照。

### Phase 6-a 達成サマリ

- ✅ **BPC 4.18 → 3.76 (-10.0%)** で全 phase 中ベスト (full-corpus chars/token=1.65 基準)
- ✅ best 到達 step: 2800 → **2400 (-400 step)** で早期収束
- ✅ 実効コンテキスト 512 char → **840 char (+64%)** で同 max_len/batch のまま
- ✅ 質的にも **「津田 + お延」(『明暗』)、 「高柳君」(『野分』)、 「カムパネルラ + 苹果」(『銀河鉄道』)** などの作品横断キャラクタ関係を正確再現
- ⚠️ step 2400 以降 train-val gap 拡大 (軽度オーバーフィット)
- ⚠️ 戯曲記号・作家ヘッダ・作家ミックスは未解消 (Phase 7-a 候補)

### Phase 6-d 達成サマリ

- ✅ **best val_loss 4.274 → val_ppl 71.81 / BPC 4.22 (val 基準)**、 Phase 6-a 比 val_loss -0.6%、 val_ppl -2.6%
- ✅ best 到達 step: **2800** (Phase 6-a の 2400 から +400 後ろにシフト、 LR が長く高い WSD の特性)
- ✅ max_len=1024 で 推論時の長文一貫性が向上
- ⚠️ 期待した「BPC -2 〜 -5%」 は達成できず (-0.6%)。 同 token/step 設定で max_len 倍増だけでは長距離学習加速に限度あり
- ⚠️ 戯曲記号 (「シロオテ」等) は依然混入 → **Phase 7-a (v2 corpus + special token) で構造的解決**
- ⚠️ Phase 7-3/7-4 が間に合わなかったため per-step ~9,200 ms (Phase 7-1/7-2 性能のまま)。 Phase 7-a は Phase 7-4 バイナリで起動可

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

## 学習時間の大規模高速化 (Phase 7-1〜7-4 高速化フェーズ) — 完了

Phase 6-c で per-step ~11,000 ms (3000 step ≒ 9.2 h) になり、 GPU 移植を避けつつ CPU だけで **2-3x の改善**を狙うフェーズ。 累積で bench batch=2 比較で **1500 ms → 945 ms (1.59x)**、 Phase 6 形状 (batch=16) 換算 で **~9,000 ms → ~5,800 ms** を達成見込み。

| 項目 | 内容 | 状態 |
|------|------|------|
| **B4 WSD scheduler** | `LrScheduleKind::WarmupStableDecay` を追加。 Config で切替可能 | ✅ 完了 (Phase 6-d で利用) |
| **Phase 7-1 Matrix 直叩き化** | 全 9 レイヤに `forward_matrix(&Matrix) -> Matrix` を追加。 旧 `Vec<Vec<f32>>` API 経路の `from_jagged`/`to_jagged` 変換を完全排除 | ✅ 完了 |
| **Phase 7-1 QKV projection 融合** | W_Q/W_K/W_V → W_QKV (`d × 3d`)。 forward 1 matmul + backward 1 matmul。 checkpoint は旧形式互換 | ✅ 完了 |
| **計測 (Phase 7-1 終端)** | `bench_phase7_step_time` で ~1.16-1.20x speedup を確認 (Phase 6-c 並列実行時の保守値、 単独条件では推定 1.3-1.4x) | ✅ 完了 |
| **Phase 7-2 旧 API クリーンアップ** | `forward(&[Vec<f32>])` 系を全削除 (27 メソッド)、 `*_matrix` → bare 名にリネーム、 KV-cache 推論パス (`forward_step`) も Matrix 直叩き化 | ✅ 完了 |
| **Phase 7-3 transpose 排除** | backward の `_.transpose().matmul(_)` 17 箇所を `matmul_t1`/`matmul_t2` (BLAS の trans フラグ) に置換、 1.12x speedup vs 7-2 | ✅ 完了 |
| **Phase 7-4 fused matmul-add + alloc 削減** | `matmul_*_add_into` (BLAS beta=1) で 10 個の grad 累積を 1 sgemm に圧縮 + scores.clone() 削除 + softmax backward in-place 化、 **1.43x speedup vs 7-3** (1352→945 ms)、 per-step alloc ~440MB→~50MB | ✅ 完了 |
| **Phase 7-5 MHA buffer field 化 + Flash Attention** | `buf_qkv` 等 7 つの reusable buffer 追加で per-step alloc ~30 MB 削減 + block-tiled online softmax (BLOCK=256) で `seq²` 行列を materialize しない、 causal upper triangle skip で attention FLOP ~半減、 数値一致テスト 10 件全 pass | ✅ 完了 |

詳細は [`docs/performance.md`](performance.md#phase-7-大規模高速化-2026-05-) を参照。

## コーパス前処理 + 作家・戯曲 special token (Phase 7-a) — 起動準備完了

Phase 6-d で残った質的課題 (戯曲記号混入、 作家ヘッダ生成、 作家ミックス、 章番号擾乱) を **コーパス + tokenizer 側** で構造的に解決。

| 項目 | 内容 | 状態 |
|------|------|------|
| P7-1-A コーパス分析 | `src/bin/analyze_corpus.rs`、 戯曲行 / 作家ヘッダ / 注釈の頻度算出 | ✅ 完了 |
| P7-1-B コーパス変換 | `src/bin/clean_aozora_corpus.rs` で `corpus/aozora_meiji_taisho_v2.txt` 生成 (504 作品、 戯曲 8 作品を `<DRAMA>` で囲む、 章番号 1131 行削除) | ✅ 完了 |
| P7-1-C Tokenizer 拡張 | `CharBpeTokenizer::add_special_token` API 追加 + `tokenizers/charbpe_v8010_aozora_meiji_taisho_v2_s500000.bin` (vocab 8010、 +10 special token) | ✅ 完了 |
| P7-1-D config 追加 | `Config::aozora_meiji_taisho_charbpe8k_max1024_wsd_v2()` を `main.rs` に追加 | ✅ 完了 |
| **Phase 7-a 学習起動** | Phase 7-4 バイナリで run、 per-step ~5,800 ms × 3000 step ≒ **5 h** 想定。 BPC: Phase 6-d 4.22 → **3.9-4.1** (-3〜-7%) 期待 | ⏳ ユーザー判断待ち |

詳細は [`docs/phase7.md`](phase7.md) を参照。

## 学習時間の高速化 (Phase 6 候補) — 旧記述

Phase 5-4a (d=512, n_layers=6, batch=32) の現状: per-step ~9.4s、 実効 150 GFLOPS
(M1 Max AMX 理論ピーク 700 GFLOPS の **21%**)。 改善余地あり。

### Tier 1 (大改善, 5-15x): 大きな設計変更が必要
- **Apple GPU (Metal/MPS) 移植** — Pure Rust の精神を壊す
- **CUDA/ROCm 移植** — 巨大な追加コスト

### Tier 2 (1.5-3x, 推奨): エンジニアリング改善

| 案 | 効果 | 工数 | 説明 |
|----|------|------|------|
| **`Vec<Vec<f32>>` → `Matrix` 全面置換** | 1.3-1.7x | 中 (1 日) | KV cache で見えた alloc コストが学習にも効いている。 norm/ffn/dropout/embedding/loss を `Matrix` に統一 |
| **batch_size 拡大 (32→64)** + LR 比例 | 1.2-1.5x | 小 | AMX タイル利用効率向上 |
| **WSD (Warmup-Stable-Decay) スケジューラ** | 同 val_ppl を 15-30% 早く到達 | 小 | MiniCPM/DeepSeek 慣例 |
| **Flash Attention 風融合 (softmax + matmul)** | 1.3-2x (attn 部分のみ) | 中-大 | CPU でも IO 削減で効く |
| **Grad accumulation (effective batch 256-512)** | 収束 step 削減の可能性 | 小 | LLaMA / GPT-3 慣例 |

### Tier 3 (1.8-2.5x): mixed precision
- **bf16 / f16** (Apple BNNS 経由) — Matrix の dtype 抽象化が必要、 数値安定性試験要

### Tier 4: 「収束 step 数を減らす」 アプローチ
- BPE トークナイザに切替 (vocab 8K-16K で token 数 1/2-1/3) — step 数比例で削減
- z-loss / aux loss、 カリキュラム学習、 SP/μP init

### 着手判断
- 現在の Phase 5-4a 完了 (~3.7h) を **見守る** 方針 (実行中の変更はリスクが高い)
- Phase 5-4b/c で **per-step が 1.5-2 倍** になり、 トータル学習時間が 13-21h になるのが現実化したら、 その時点で Tier 2 (Matrix 統一 + batch 拡大) を検討

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
