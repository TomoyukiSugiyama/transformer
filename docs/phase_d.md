# Phase D: モダンアーキテクチャ要素の段階的導入

[Phase 4b](phase4.md) で **コーパス拡大による val_ppl 改善が頭打ち** ( 18.15 → 18.98 ) になり、
モデル容量・アーキテクチャ側の改善余地が次の主戦場と判明した。 GPT-2 (2019) ベースから
**LLaMA / GPT-NeoX 系で標準化された改良要素**を 1 つずつ取り込み、 各単独効果を定量評価する。

## ロードマップ

| 段階 | 項目 | 状態 | best val_ppl への効果 |
|------|------|------|----------------------|
| D-1 | RMSNorm | ✅ 完了 | 18.98 → **18.77** (-1.1%) + ピーク 100 step 後ろ倒し (過学習耐性向上) |
| D-3 | SwiGLU FFN | ✅ 完了 | 18.77 → **18.70** (-0.4%) + ms/step **-35%** + best 到達 **200 step 早期化** |
| D-2 | RoPE | ✅ 完了 | floor は不変 (18.70 → 18.84)。 ただし early step で **-10〜22%** の収束加速 |
| D-4 | MQA / GQA | 実施中 | 推論速度のみで val_ppl 影響は小。 KV cache 未実装のため効果検証が困難 |

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

### 学習評価結果

`Config::aozora_soseki_works()` を **RMSNorm + SwiGLU** に切替えた `phase4b_aozora_soseki_works_d384_n6_char_rms_swiglu`
で評価。 設定は RMSNorm 単独試験と完全同一 ( `feed_forward_kind` のみ `Gelu` → `SwiGlu` に変更 )。

#### val_loss / val_ppl 推移 (3 構成比較)

| step | LayerNorm + GELU | RMSNorm + GELU | **RMSNorm + SwiGLU** | vs LN | vs RMS |
|------|------------------|----------------|----------------------|-------|--------|
| 100  | 49.12 | 48.51 | **48.12** | -2.0% | -0.8% |
| 200  | 30.58 | 30.27 | **29.52** | -3.5% | -2.5% |
| 300  | 24.37 | 24.07 | **23.18** | -4.9% | -3.7% |
| 400  | 21.30 | 21.22 | **20.28** | -4.8% | -4.4% |
| 500  | 20.17 | 20.11 | **19.64** | -2.6% | -2.3% |
| **600**  | 19.24 | 18.93 | **18.70 ★** | **-2.8%** | **-1.2%** |
| 700  | 18.98 ★ | 19.00 | 18.76 | -1.2% | -1.3% |
| 800  | 19.33 | 18.77 ★ | 19.10 (overfit) | — | — |
| 900  | — | 19.48 | 19.94 | — | — |
| 1000 | — | 19.95 | 21.15 | — | — |

#### 速度・収束効率

| 指標 | LayerNorm + GELU | RMSNorm + GELU | RMSNorm + SwiGLU |
|---|---|---|---|
| best val_ppl | 18.98 | 18.77 | **18.70** |
| best step | 700 | 800 | **600** |
| best 到達時間 | 7440s (2h 4m) | 8540s (2h 22m) | **4370s (1h 13m)** ⚡ |
| ms_per_step (steady) | ~10800 ms | ~11000 ms | **~7400 ms** ⚡ |
| 過学習開始 step | 800 | 900 | 700 |

**観察**:
1. **best val_ppl の改善幅は控えめ** (-0.4% 対 RMSNorm)。 論文 (Shazeer 2020) の報告 0.5〜2% と整合
2. **速度改善が予想を覆して大きい**: 当初 「matmul +1 個で +5〜10% 遅くなる」 と予想していたが、 **逆に 35% 高速化**
   - SwiGLU は `d_ff_g = 1024 < d_ff = 1536` で 1 個あたりの matmul が小さい (cache friendly)
   - `bias add` が無く、 `Swish (sigmoid)` が `GELU (erf)` より計算が単純
   - 結果として 3 matmul の合計時間 < 2 matmul + bias + GELU
3. **best 到達時間の総合短縮 -41%** (LN baseline 比 7440s → 4370s)。 SwiGLU 採用の **最大の価値はここ**
4. **過学習が早く来る** (step 700 vs RMSNorm 900) のは表現力アップによる副作用。 dropout / weight_decay の再チューニング余地あり

### 推論サンプル (RMSNorm + SwiGLU best.bin = step 600)

`top_k=5, temperature=1.0, repetition_penalty=1.2, max_new_token=100` で生成:

```
[prompt: 私は]
私はすぐ立っているのだから、それを断わなければならない。そこで私はあまり安心したと
見えていると同じ事に思われていた。
　奥さんは自分の前に坐っていました。そうしてお嬢さんが帰りました。

[prompt: 先生は]
先生は、私がそんなに心配した事をいうと、私の顔を見て、「あれほど」と答えている。
奥さんの方から見るより外に私の顔に関するのだろう。
「奥さまもお気が悪くったね。今度は奥さんに何の意味もあったのよ。私にはあな

[prompt: ある日]
ある日かも知れない。
　三四郎は与次郎にとったが、この時はじめて来ますか」
「そうさな。あんなにしろお会いなさらなけりゃ、まだよろしくないと言ったが、
どうしても借金を借せずに行ったものですね。なぜ」と言って

[prompt: 吾輩は]
吾輩は大に感服のために、主人がそこへ行って見るところを見てもらわからず主人の方を
見せようとしなければならぬ。「いや御令嬢さん」と迷亭先生の顔を見て
「おやちょっと御馳走を願いますからね……」
　迷亭が口の中へ
```

**定性評価**:
- **作品判別が更に明確化**: 「吾輩は」 prompt から **迷亭先生 / 主人** (「吾輩は猫である」 固有) が初めて出現
- 「三四郎」 prompt → 三四郎・与次郎、 「私は」/「先生は」 prompt → 奥さん・お嬢さん で適切な作品スタイル
- 鉤括弧構造、 段落先頭全角空白、 漱石らしい敬語が安定。 Phase D-1 step 800 サンプルと比較しても遜色なし

### 結論

- ✅ **品質**: わずかに改善 (val_ppl -0.4%、 LayerNorm baseline 比 -1.5%)
- ✅ **速度**: **大幅高速化** (per-step -35%、 best 到達時間 -41%)
- ❌ **過学習耐性**: SwiGLU の表現力アップで dropout=0.2 では抑え切れず、 過学習が早期化
- 採用判定: **採用**。 学習効率の改善が圧倒的で、 後段の Phase D 実験を半分の時間で回せる効果が大きい

### 実装メモ

- `src/swiglu_feed_forward_network.rs` に `SwiGluFeedForwardNetwork` 構造体
- `src/feed_forward.rs` に `FeedForward` trait + `FeedForwardKind` enum + factory (Normalization と同パターン)
- `Box<dyn FeedForward>` で TransformerBlock 内の FFN を統一
- `LanguageModel` の `meta.feed_forward_kind` を checkpoint に保存・復元、 旧 ckpt は `Gelu` fallback

実装上のポイント:
- `d_ff_g = (d_ff × 2) / 3` で param-matched にする (3 行列 × 1024 ≈ 2 行列 × 1536)
- bias なし (LLaMA 流)。 これが実は**速度改善の隠れた要因**
- backward の `dl_dgate = (dl_da ⊙ up) ⊙ Swish'(gate)` で `Swish'` の適用順序ミスに注意
- 数値勾配チェック (中心差分) が backward の数式バグを防ぐ唯一の手段

---

## Phase D-2: RoPE

論文: Su et al. 2021 ["RoFormer: Enhanced Transformer with Rotary Position Embedding"](https://arxiv.org/abs/2104.09864)。
LLaMA / PaLM / Mistral / GPT-NeoX が標準採用。 attention の Q, K に位置 m に応じた **回転** を掛けることで、
内積 `<Q'(m), K'(n)>` が **相対位置 `(m-n)` のみに依存**する性質を持たせる。

### 実装

- `src/rope.rs` に `Rope` 構造体 (cos/sin テーブル precompute + `apply_in_place` / `apply_backward_in_place`)
- `src/positional_encoding.rs` に `PositionalEncodingKind` enum (`Sinusoidal` / `Rope`)
- `MultiHeadAttention` に `rope: Option<Rope>` フィールド追加、 forward で Q, K のみ回転 (V には掛けない)、
  backward では cache の未回転 Q, K を再回転して attention backward を通し、 dl_dQ/dl_dK には逆回転を適用
- `LanguageModel` の `pe: Option<SinusoidalPE>` 化、 RoPE 選択時は埋め込み素通し
- `meta.positional_encoding_kind` を checkpoint に保存、 旧 ckpt は `Sinusoidal` fallback
- 2 次元ペアの取り方は **インターリーブ式** ( `(x_0,x_1), (x_2,x_3), ...` ) を採用 (LLaMA 半分割式より直感的)

### 設定

`Config::aozora_soseki_works()` を **RMSNorm + SwiGLU + RoPE** に切替え (`feed_forward_kind`, `normalization_kind` は据置)。
RoPE base = 10000.0 (LLaMA 公式値)。 run_name は `phase4b_aozora_soseki_works_d384_n6_char_rms_swiglu_rope`。

### テスト (`src/rope.rs` 内蔵 6 ケース、 すべて pass)

| テスト | 確認内容 |
|---|---|
| `position_zero_is_identity` | 位置 0 で恒等変換 |
| `rotation_preserves_norm` | 回転は等長変換 ( \|rotate(x)\| == \|x\| ) |
| `forward_then_backward_is_identity` | `R^T · R = I` (逆回転 = 転置) |
| `relative_position_invariance` | **`<rotate(q,m), rotate(k,n)>` が `(m-n)` のみ依存** ★中核性質 |
| `precomputed_tables_match_formula` | cos/sin テーブルが `cos(mθ_i), sin(mθ_i)` と一致 |
| `backward_matches_numerical_gradient` | 中心差分による解析勾配の検証 (1e-3) |

### val_loss / val_ppl 推移 (4 構成比較)

| step | LN + GELU | RMS + GELU | RMS + SwiGLU | **RMS + SwiGLU + RoPE** | vs SwiGLU |
|------|-----------|------------|--------------|-------------------------|-----------|
| 100  | 49.12 | 48.51 | 48.12 | **37.54** | **-22.0%** ⚡ |
| 200  | 30.58 | 30.27 | 29.52 | **23.85** | **-19.2%** ⚡ |
| 300  | 24.37 | 24.07 | 23.18 | **20.67** | **-10.8%** ⚡ |
| 400  | 21.30 | 21.22 | 20.28 | **19.05** | -6.1% |
| 500  | 20.17 | 20.11 | 19.64 | 19.15 | -2.5% |
| **600** | 19.24 | 18.93 | **18.70 ★** | **18.84 ★** | +0.7% |
| 700  | **18.98 ★** | 19.00 | 18.76 | 20.03 (overfit 開始) | — |
| 800  | 19.33 | **18.77 ★** | 19.10 | 21.01 (overfit 進行) | — |

### 速度比較 (M1 Max, steady state)

| 構成 | ms_per_step | vs SwiGLU |
|------|-------------|-----------|
| RMS + GELU | ~11000 ms | +50% |
| RMS + SwiGLU | ~7400 ms | baseline |
| **RMS + SwiGLU + RoPE** | **~7700 ms** | **+4%** |

RoPE の per-step overhead は **+4%** 程度。 各 layer で Q, K への 4 乗算 + 2 加算が `seq × d_head / 2` 回追加されるが、
matmul 主体の計算量に対し十分小さい。 事前予想 `+5〜10%` の下端で済んだ。

### 観察と結論

1. **収束加速が劇的** (early step で -10〜22%): RoPE の **相対位置を直接 attention に注入する** 効果が、
   sinusoidal PE が「埋め込みに加算して各層で間接的に学習する」 方式より効率的。
   論文では「同 floor に早く到達」 と報告されているが、 ここまで顕著な差は予想を超えた

2. **floor は更新されず** (step 600: 18.84 vs SwiGLU best 18.70 で +0.7%):
   論文 (Su et al. 2021) の主張 「**floor は同じだが収束が速い**」 と整合的。 我々の事前予想シナリオ B に該当

3. **過学習が早期に開始** (step 700 で 20.03、 step 800 で 21.01): SwiGLU best (step 600 / 18.70) と比較すると
   **過学習開始 step が SwiGLU より早い**。 RoPE で内部表現が早く豊かになる分、
   小規模コーパス (1.08M char) との容量不一致が顕在化したと推定。 SwiGLU 単独でも step 700 で過学習開始だったので、
   RoPE 追加で大きく前倒しになったわけではない

4. **推論サンプルの定性 (step 600 best.bin)**: 三四郎・与次郎・美禰子 (= 「三四郎」 の人物) に偏った生成傾向。
   作品判別 (こころ / 三四郎 / 吾輩は猫である) は SwiGLU best ほど明確でない。
   floor 差 +0.7% は知覚可能な品質差として現れる

### 採用判定: **採用** (max_len 拡張時の伏線として)

| 評価軸 | 結果 |
|---|---|
| best val_ppl | 微悪化 (+0.7%) |
| 学習速度 | **大幅改善 (early step -22%) ⭐** |
| 過学習耐性 | やや劣化 |
| 推論品質 (定性) | SwiGLU best より控えめ |
| Negative regression | best val_ppl のみ |

短い max_len (256) では floor 改善は出ないが、 **RoPE の本領は max_len 拡張時の汎化** にあるため、 Phase 5-2 (max_len 512 化)
での再評価で本評価を確定する。 また学習効率の高さは Phase 5 以降のサブフェーズで **early stopping 戦略** に活かせる。

---

## Phase D-4: MQA / GQA
GQA は MHA (Multi Head Attention) の自然な拡張で、メモリ削減と品質のトレードオフを n_kv_heads 一つで制御できます。

```
MHA: Q heads=8,K heads=8, V heads=8 -> KV head を全 Q head が独立に持つ
GQA: Q heads=8,K heads=2, V heads=2 -> 4つの Q head が 1 KV head を共有
MQA: Q heads=8,K heads=1, V heads=1 -> 全 Q head が 同一 KV head を共有
```

### 実装
- `src/multi_head_attention.rs` の `MultiHeadAttention` 構造体に `n_kv_heads` `d_kv_model` を追加し、`qkv_*` を `q_*` と `kv_*` に分離

### 設定
`Config::aozora_wikipedia_mixed_d768_n8_charbpe32k_max1024_wsd()` に `n_kv_heads=4` を設定した`Config::aozora_wikipedia_mixed_d768_n8_charbpe32k_max1024_wsd_gqa()` を追加。
run_name は `phase8b_aozora_wikipedia_mixed_d768_n8_charbpe32k_rms_swiglu_rope_max1024_wsd`。

### ベンチマーク結果
`#[test] bench_phase8_step_time` (`cargo test --release bench_phase8_step_time -- --nocapture --ignored`)
n_kv_heads_12, n_kv_heads_4 を batch_size=2 で 3 step 計測。

`n_kv_heads_12` : d_model=768, n_heads=12, n_kv_heads=12, n_layers=8, d_ff=3072, max_len=1024
`n_kv_heads_4` : d_model=768, n_heads=4, n_kv_heads=12, n_layers=8, d_ff=3072, max_len=1024

| パラメータ | per-step (batch=2) |speedup|
|------|--------------------|----------|
| n_kv_heads_12 | ~1875.1 ms | (baseline) |
| n_kv_heads_4 | ~1701.2 ms | **1.09x** |


### val_loss / val_ppl / bpc 推移、平均実行時間比較

`n_kv_heads_12` : h12
`n_kv_heads_4` : h4

| step | h12 val_loss | h4 val_loss | 差       | h12 val_ppl | h4 val_ppl | 差     | h12 bpc | h4 bpc | 差       |
| ---- | ----------- | ----------- | ------- | ---------- | ---------- | ----- | ------ | ------ | ------- |
| 500  | 5.7588      | 5.7764      | +0.0176 | 316.98     | 322.61     | +5.63 | 5.0124 | 5.0277 | +0.0153 |
| 1000 | 5.0815      | 5.0787      | −0.0028 | 161.02     | 160.57     | −0.45 | 4.4229 | 4.4204 | −0.0025 |
| 1500 | 4.6445      | 4.6645      | +0.0200 | 104.01     | 106.11     | +2.10 | 4.0424 | 4.0598 | +0.0174 |
| 2000 | 4.3931      | 4.4073      | +0.0142 | 80.89      | 82.05      | +1.16 | 3.8236 | 3.8360 | +0.0124 |
| 2500 | 4.2623      | 4.2808      | +0.0185 | 70.98      | 72.30      | +1.32 | 3.7098 | 3.7259 | +0.0161 |
| 3000 | 4.1763      | 4.2051      | +0.0287 | 65.13      | 67.03      | +1.90 | 3.6350 | 3.6600 | +0.0250 |

val 品質は一貫して h12 が優位。step 1000 のみ h4 がわずかに上回るが誤差範囲
ただし差は全点で bpc 0.03 以内と実用上は同等

STEP3000 までの平均実行時間:
| 指標                  | n_kv_heads_12      | n_kv_heads_4       | 差分                  |
| ------------------- | ------------------- | ------------------- | ------------------- |
| 平均 ms/step          | 16,796.5 ms         | 15,862.5 ms         | −934.0 ms           |
| 最小 ms/step          | 16,499.8 ms         | 15,702.9 ms         | −796.9 ms           |
| 最大 ms/step          | 17,824.5 ms         | 16,228.2 ms         | −1,596.3 ms         |
| elapsed @ step 3000 | 50,389 s（839.8 min） | 47,587 s（793.1 min） | −2,802 s（−46.7 min） |

n_kv_heads_4 は n_kv_heads_12 より 5.6% 高速

### 結論

- ✅ **品質**: bpc 0.03 以内と実用上は同等
- ✅ **速度**: h4 は速度 5.6% 改善


---

## Phase D 総括 (RMSNorm + SwiGLU + RoPE) — 累積効果

D-1〜D-3 の全実験完了。 累積改善:

| 構成 | best val_ppl | LN baseline 比 | best step | best 到達時間 | LN baseline 比 |
|------|-------------|---------------|----------|-------------|---------------|
| LN + GELU (baseline) | 18.98 | — | 700 | 7440s | — |
| RMS + GELU (D-1) | 18.77 | -1.1% | 800 | 8540s | +14.8% |
| RMS + SwiGLU (D-3) | **18.70 ★** | **-1.5%** | 600 | **4370s** | **-41.3%** |
| RMS + SwiGLU + RoPE (D-2) | 18.84 | -0.7% | 600 | 4350s | -41.5% |

### Phase D で得た知見

1. **「best val_ppl の数値改善」は実は控えめ** (LN→D-2 で 1.5%): max_len=256 / 1.08M char という規模では
   アーキテクチャ改善の効果は飽和に近い
2. **学習効率の改善は圧倒的** (-41% time-to-best): SwiGLU の per-step 高速化 (-35%) と
   RoPE の収束加速 (early step -22%) の合算。 後続実験を 2 倍速で回せる **副次的価値が大きい**
3. **「同じ目標品質に到達する時間」 で評価すると改善幅が見えにくい**: val_ppl 19.0 達成までを比較すると
   LN: step 700 / SwiGLU+RoPE: step **400** で **57% 時間短縮**
4. **大規模化を前提とした基盤整備が完了**: モダン LLaMA スタック (RMSNorm + SwiGLU + RoPE) はそのまま
   d_model 拡大や max_len 拡張に乗せ替えられる

### 残された制約と Phase 5 への接続

- **生成文の日本語が局所的にしか正しくない** (構文・敬語・助詞の細部、 長文構造の追跡)
- 容量律速 (10.7M params)、 データ律速 (1.08M char)、 max_len 律速 (256 = 5-10 文) の **三重制約**
- アーキテクチャ改善だけでは突破不可能と判明 → スケール側の改善が必要

→ **[Phase 5](phase5.md)** で sampling / max_len / モデル / コーパスの 4 軸スケールに進む。
