# Phase D: モダンアーキテクチャ要素の段階的導入

[Phase 4b](phase4.md) で **コーパス拡大による val_ppl 改善が頭打ち** ( 18.15 → 18.98 ) になり、
モデル容量・アーキテクチャ側の改善余地が次の主戦場と判明した。 GPT-2 (2019) ベースから
**LLaMA / GPT-NeoX 系で標準化された改良要素**を 1 つずつ取り込み、 各単独効果を定量評価する。

## ロードマップ

| 段階 | 項目 | 状態 | best val_ppl への効果 |
|------|------|------|----------------------|
| D-1 | RMSNorm | ✅ 完了 | 18.98 → **18.77** (-1.1%) + ピーク 100 step 後ろ倒し (過学習耐性向上) |
| D-3 | SwiGLU FFN | ✅ 実装完了 / 学習評価予定 | (測定予定: RMSNorm + SwiGLU 統合構成で評価) |
| D-2 | RoPE | 未着手 | (測定予定) |
| D-4 | MQA / GQA | 未着手 | 推論速度のみ、 val_ppl 影響は小 |

順序が D-1 → D-3 → D-2 になっているのは:
- D-1 (RMSNorm) と D-3 (SwiGLU) は他レイヤーへの影響が小さく **独立検証可能**
- D-2 (RoPE) は positional encoding 撤去を伴い、 モデル全体への影響が大きい

---

## Phase D-1: RMSNorm vs LayerNorm

### 設定

[Phase 4b](phase4.md) (LayerNorm 版) と **`normalization_kind` のみ** を `Layer` → `Rms` に変更。
他のハイパラ・コーパス・seed 経路は完全に同一。

| 項目 | 値 |
|------|----|
| run_name (LayerNorm) | `phase4b_aozora_soseki_works_d384_n6_char` |
| run_name (RMSNorm) | `phase4b_aozora_soseki_works_d384_n6_char_rms` |
| corpus | 漱石 7 作品 (1.086M char) |
| 構造 | d_model=384, n_heads=6, n_layers=6, d_ff=1536, max_len=256 |
| 正則化 | dropout=0.2, weight_decay=0.1 |
| 最適化 | AdamW (β2=0.99), lr 1e-3 → 1e-4, warmup=100 |

### val_loss / val_ppl 推移

| step | LayerNorm val_ppl | RMSNorm val_ppl | 差 (相対) |
|------|-------------------|-----------------|----------|
| 100 | 49.12 | **48.51** | -1.24% |
| 200 | 30.58 | **30.27** | -1.01% |
| 300 | 24.37 | **24.07** | -1.21% |
| 400 | 21.30 | **21.22** | -0.42% |
| 500 | 20.17 | **20.11** | -0.30% |
| 600 | 19.24 | **18.93** | **-1.59%** |
| 700 | **18.98** ★LN best | 19.00 | +0.10% |
| **800** | 19.33 (overfit 開始) | **18.77** ★RMS best | **-2.90%** |
| 900 | — | 19.48 (overfit 開始) | — |
| 1000 | — | 19.95 | — |

**観察**:
1. **RMSNorm best (18.77 @ step 800) が LayerNorm best (18.98 @ step 700) を 1.11% 上回る**
2. ピーク到達は **100 step 後ろ倒し** (RMSNorm: 142分 @ step 800、 LayerNorm: 120分 @ step 700)。
   学習効率が上がっただけでなく、 **過学習を 100 step 遅らせる耐性**を示した
3. **ピーク前は RMS が常に 0.3-2.9% 良い** — noise でない系統的傾向
4. step 700 → 800 で val_ppl が一旦上昇 (19.00) してから再低下 (18.77) する非単調挙動が観測された。
   train ema_ppl は連続単調減少 (12.71 → 9.96) しており、 内部表現の再構築過程と推測

### 速度比較 (M1 Max)

| 実装 | ms_per_step (steady) |
|------|---------------------|
| LayerNorm | 10500-11400 ms |
| RMSNorm | 10500-11400 ms |

**±5% noise 内で速度差なし**。 RMSNorm 原論文の主張する 「LayerNorm 比 1.2-1.5× 高速」 は
本実装では再現せず。 理由:
- LayerNorm/RMSNorm はどちらも `Vec<Vec<f32>>` ベースで、 行列演算 (matmul) に対して
  1 step 全体の数 % 以下のコスト
- matmul (Apple Accelerate / AMX) が圧倒的支配項なので、 そこを高速化しないと総時間に効かない
- RMSNorm の高速化メリットが顕在化するのは **GPU で memory bandwidth bound** の場合

### 推論サンプル (RMSNorm best.bin = step 800)

`top_k=5, temperature=1.0, repetition_penalty=1.2, max_new_token=100` で生成:

```
[prompt: 私は]
私は父の後で、それを一度に置いた。私はまだ何とも答えなかった。
「あなたが先生は何もいわずに、お前が今まで通り越したのです」
「何だいどんな事をお聞きませんね。何でもお前が東京へ帰ったらよく解るじゃありま

[prompt: 先生は]
先生は私の方がまた私に対してもらわないでも、そこを動かす間、奥さんが私に話し合せる気色だと思うのは。
「奥さんやお嬢さんに頼んでいらっしゃい」
　先生の言葉は、どこでも私を顧みなければならぬように見えた。

[prompt: ある日]
ある日のように思われた。
　私は父の病気を聞いて、先生の死んだままでもっともに病症がなくなる。しばらくしてもいえなくなって、
その病気を思うかがりにやにや笑い出す事もあった。

[prompt: 東京の]
東京の生徒がわるいと言った。
「これからは、今日はどうだい、あなたはお国です」
　三四郎はまた大きな声で答えたのを聞くや否や、そりゃ、また口へ出して、まずに行った。与次郎もこのあいだについて来た。

[prompt: 吾輩は]
吾輩はこの時にも、吾輩の心得と見えてあらず。今までの彼等を訪問するのが一度に出来ないくらいの猫があった事だ。
それから吾輩は吾々を軽蔑するのを恐れ入れてくるごとく、決しかねて忍び込んで来て、吾人を猫に向って吾

[prompt: それから]
それから、この夏はどうしているか知らんが、まあどうぞとも思わなくっちゃ、今に限るだろう。
その上野にゃあばたを向けるのですよ」
　三四郎は黙りの方へ歩き出した。
```

**定性評価**:
- 漱石作品の登場人物・呼称が **複数作品から正しく**生成される: 先生 / 奥さん / お嬢さん / Ｋ (こころ),
  三四郎 / 与次郎 / 野々宮君 (三四郎), 吾輩 / 吾人 / 吾々 (吾輩は猫である)
- 各 prompt が引き出す作品スタイルが切り替わる: 「私は」/「先生は」 → こころ調、 「東京の」/「それから」
  → 三四郎調、 「吾輩は」 → 「吾輩は猫である」 の文語混じり調と、 **作品判別を内部で学習**できている兆候
- 鉤括弧 「」 の開閉、 段落先頭の全角空白、 句読点バランスが自然
- step 600 ベスト時より文脈の繋がりがやや改善 (e.g. 「父の病気」 → 「先生の死んだ」 への話題遷移)

### 結論

- ✅ **品質**: 1.1% val_ppl 改善 (step 800 で 18.77, baseline 18.98)
- ✅ **学習耐性**: 過学習開始が 100 step 後ろ倒し、 train ema が単調に下がり続けた
- ❌ **速度**: CPU + AMX バウンドの本実装では差なし (GPU 化したら出る種類の高速化)
- 採用判定: **採用**。 SwiGLU との **組合せでベース** として後段の Phase D 全実験に使用

### 実装メモ

- `src/root_mean_square_layer_normalization.rs` に `RootMeanSquareLayerNormalization` 構造体
- `src/normalization.rs` に `Normalization` trait + `NormalizationKind` enum + factory
- `Box<dyn Normalization>` で TransformerBlock / Transformer / final_norm を統一
- `LanguageModel` の `meta.normalization_kind` を checkpoint に保存・復元
- 旧 checkpoint (kind 未保存) は `NormalizationKind::Layer` として fallback

実装上のポイント:
- backward の `dl_dx` 式は LayerNorm と異なり mean grad の項 (`-sum_g`) が出ない
- forward の eps は `1.0 / (ms + eps).sqrt()` (sqrt の内側) が正しく、 小さい入力で顕在化する
- 数値微分テスト (中心差分) が backward 数式の唯一信頼できるセーフティネット

---

## Phase D-3: SwiGLU FFN

論文: Shazeer 2020 ["GLU Variants Improve Transformer"](https://arxiv.org/abs/2002.05202)。
LLaMA / PaLM / Mistral など現代主要モデルは標準採用。

### 実装

- `src/swiglu_feed_forward_network.rs` に `SwiGluFeedForwardNetwork` 構造体
- `src/feed_forward.rs` に `FeedForward` trait + `FeedForwardKind` enum + `load_feed_forward` factory
- `Box<dyn FeedForward>` で TransformerBlock 内の FFN を統一 (Normalization と同じパターン)
- `LanguageModel` の `meta.feed_forward_kind` を checkpoint に保存・復元
- 旧 checkpoint (kind 未保存) は `FeedForwardKind::Gelu` として fallback

### GELU FFN との param-matched 比較

| 項目 | GELU FFN (現状) | SwiGLU |
|------|-----------------|--------|
| 隠れ次元 | `d_ff = 1536` | `d_ff_g = (2/3) × d_ff = 1024` |
| 行列数 | 2 (W1, W2) | 3 (W_gate, W_up, W_down) |
| bias | あり | なし (LLaMA 流) |
| 活性化 | GELU | Swish (= SiLU) ⊙ gating: `Swish(xW_gate) ⊙ (xW_up)` |
| params/block | 2 × 384 × 1536 ≈ 1.18M | 3 × 384 × 1024 ≈ 1.18M (完全一致) |

### テスト戦略

`backward_matches_numerical_gradient` を含む 6 ケース ( pure swish / swish_grad の既知値、
fixed weights forward、 d_ff_g 計算、 中心差分での数値勾配チェック、 checkpoint roundtrip ) を
追加して、 LayerNorm 系で発生したような数式バグを事前に防いだ。

### 学習評価予定

`Config::aozora_soseki_works()` を **RMSNorm + SwiGLU** に切替えた `phase4b_aozora_soseki_works_d384_n6_char_rms_swiglu`
で評価予定。 RMSNorm 単独 (best val_ppl 18.77) からの改善幅を測定する。

期待値:
- 論文 (Shazeer 2020) では perplexity 0.5-2% の改善が報告されている (param-matched 比較)
- RMSNorm との組合せで val_ppl が 18.5 前後まで押し下げられる可能性
- per-step 時間は matmul 1 個増加分 +5〜10% を予想

---

## Phase D-2: RoPE (未着手)

(後日記載)

---

## 統合実験 (RMSNorm + SwiGLU + RoPE)

D-1〜D-3 が個別に動いたら、 全部入りの **モダン構成** で漱石 7 作品を再学習し、
LayerNorm + GELU + sinusoidal PE の baseline (Phase 4b: val_ppl 18.98) からの累積改善幅を測定する予定。
