# Phase 6: トークナイザ刷新 (char → Unicode char-level BPE)

## 背景

[Phase 5-4a](phase5.md) (d=512, char tokenizer, 6 作家 8.3M char) で **val_ppl 18.16, BPC 4.18 (random baseline からの圧縮量 8.17 bits / char, 全 phase 最高)** を達成した。 一方で次の質的課題が残った。

| 課題 | 具体例 (Phase 5-4a step 3000 生成サンプルより) |
|------|--------------------------------------------|
| **bigram 切り誤り** | `ある日` + `本の古今` → `ある日本の古今` (文字の都合で前後 token が癒着) |
| **長文の論理破綻** | 100 token 超で脱線 (1 token = 1 char のため、 512 token で 512 char しか文脈に入らない) |
| **文体一貫性の弱さ** | 段落単位で観点が揺れる |

これらは **コンテキスト長が短い** ことと **頻出 n-gram (「ので」 「ました」 「先生」 等) が複数 token に分割される** ことが主要因と分析される。 モデル拡大 (Phase 5-4b/c) は計算量が線形〜2 次で増えるため、 まず **トークナイザ側で 1 token あたりの情報密度を上げる** アプローチを試す。

## 設計判断

### 既存 BPE (byte-level) を流用しない理由

`src/bpe_tokenizer.rs` の BPE は **byte-level** で実装されている。

- 英文では byte ≈ char で問題ない
- **日本語では 1 char = 3 byte (UTF-8) のため、 merge が UTF-8 境界を跨ぐ可能性があり、 decode 時に文字化けする** (`main.rs` line 175-176 の歴史的コメント参照)

そこで Phase 6 では **既存 BPE を英語用に残し、 char-level BPE を新規実装する**。

### Unicode char-level BPE (`src/char_bpe_tokenizer.rs`)

![Tokenizer Family — 3 implementations + CharBPE (NEW)](tokenizer-family.png)

上図は `Tokenizer` trait を実装する 3 種類のトークナイザの比較。 **CharBpeTokenizer** が Phase 6 で新規追加した実装で、 既存の `CharTokenizer` / `BpeTokenizer` と並列で `TokenizerKind` enum 経由で選択できる。 特殊 token (`<PAD>` / `<UNK>` / `<BOS>` / `<EOS>`) の ID 0-3 は 3 実装で共通。

| 項目 | byte-level BPE (既存) | **char-level BPE (Phase 6)** |
|------|--------------------|----------------------------|
| 初期トークン単位 | byte (`b099` 形式) | **Unicode char そのもの** |
| 日本語安全性 | UTF-8 境界跨ぎでマージされる可能性あり ❌ | char 境界で必ず止まる ✅ |
| 空白の扱い | 廃棄 (decode で再構築) | **独立トークンとして保持** (lossless) |
| 小文字化 | あり (`.to_lowercase()`) | なし (case を保持) |
| 初期 vocab | 特殊 4 + 256 byte × 2 (plain / `</w>`) = **516** | 特殊 4 + ユニーク char (plain のみ) = **~5,226** (日本語 corpus 実測) |
| `</w>` 単語末尾マーカー | あり (英語の word-final/internal の subword 区別に有用) | **不採用** (日本語は空白で語を区切らないため旨味薄、 初期 vocab を半減できる、 SentencePiece と同方針) |
| decode の正確性 | 単語間にスペース挿入の heuristic → 日本語で誤動作 | **完全に lossless** (token をそのまま連結するだけ。 空白・句読点は独立トークンとして保持済) |

主要 API:

```rust
// src/char_bpe_tokenizer.rs
pub struct CharBpeTokenizer { /* ... */ }
impl CharBpeTokenizer {
    pub fn train(text: &str, vocab_size: usize) -> Self;
    pub fn encode_long(&self, text: &str) -> Vec<usize>;   // BOS + ids + EOS
    pub fn encode_prompt(&self, text: &str) -> Vec<usize>; // BOS + ids
    pub fn decode(&self, ids: &[usize]) -> String;         // lossless
    pub fn num_merges(&self) -> usize;
}
impl Tokenizer for CharBpeTokenizer { /* trait 実装 */ }
impl Checkpointable for CharBpeTokenizer { /* save / load */ }
```

`TokenizerKind::CharBpe` を `src/tokenizer.rs` に追加し、 `load_tokenizer` と `train_tokenizer` が分岐する。

### char-level BPE と char tokenizer は同じにならないのか?

vocab_size の選択次第:

| vocab_size | 振る舞い | 効果 |
|-----------|---------|------|
| `= unique_chars` | merge 余地ゼロ | char tokenizer と実質同等 ❌ |
| 8,000 (Phase 6-a) | char ~5,220 + merges ~2,780 | sequence 30-40% 短縮見込み ✅ |
| 16,000 (Phase 6-b) | char ~5,220 + merges ~10,780 | sequence 50-60% 短縮見込み ✅ |

高頻度な 2-gram/3-gram (「ので」 「ました」 「である」 「先生」 「私は」 等) が学習されて 1 token になる。

## 期待効果

### 1 token あたり char 数の増加 → 実質的な context window 拡大

実コーパスでの **ドライラン (500K char サンプルで merge 学習 + 全 8.3M char で coverage)** 結果:

| Tokenizer | 1 token あたり char | max_len=512 で扱える文字数 | 学習時間 (実測 or 推定) |
|-----------|-------------------|--------------------------|---------------------|
| Char (Phase 5-4a) | 1.00 | 512 char | (学習不要) |
| CharBpe vocab=6,000 (bench 実測) | **1.38** | 707 char | **85 s** (実測) |
| **CharBpe vocab=8,000 (Phase 6-a 候補)** | **~1.7-1.8** (推定) | ~870-920 char | ~5 min (推定) |
| CharBpe vocab=12,000 | ~2.0-2.3 (推定) | ~1,020-1,180 char | ~12 min (推定) |
| **CharBpe vocab=16,000 (Phase 6-b 候補)** | **~2.3-2.7** (推定) | ~1,180-1,380 char | ~18 min (推定) |

学習された merges 上位 30 件は **高頻度日本語 n-gram を正確に捕捉**:
- 動詞活用: 「って」 「した」 「いる」 「する」 「ない」
- 助詞・助動詞: 「から」 「ので」 「ました」 「という」
- 指示・接続: 「その」 「これ」 「それ」 「には」
- 名詞: 「自分」 「さん」

→ **同じ計算量 (max_len, batch_size 不変) で 1.7〜2.7x 長い文脈** を 1 step で学習できる。

### bigram 切り誤りの軽減

`ある日` `先生は` `ました` などの高頻出シーケンスが merge されて 1 token になれば、 character 境界での誤接続 (`ある日本` 問題) が構造的に減る。

### 評価軸: BPC で比較

vocab inflation により `val_ppl` の **絶対値** は char と比較不能 (vocab 5,220 → 8,000 でランダムベースライン 12.35 → 12.97 bits)。 公平な比較は **BPC (bits per character)** で行う。

```
BPC = (val_loss [nats] / ln(2)) × (token 数 / char 数)
```

Phase 5-4a の BPC = 4.18 を下回れれば「真の改善」と判定する。

## 実験計画

### Phase 6-a (vocab 8K, d=512)

| Config | aozora_meiji_taisho_charbpe8k_max512 |
|--------|-------------------------------------|
| tokenizer | CharBpe, vocab=8,000 (初期 ~5,226 + merges ~2,774) |
| corpus | aozora_meiji_taisho.txt (8.3M char, Phase 5-3/5-4a と同じ) |
| アーキ | d=512, n_heads=8, n_layers=6, d_ff=2048 (Phase 5-4a と同じ) |
| max_len | 512 |
| 学習設定 | lr_max=7e-4, warmup=300, end_step=3000, batch_size=32, dropout=0.2 |

**期待値**: BPC < 4.18 (Phase 5-4a を下回る)、 val_ppl 20-25 程度 (vocab 増分込み)、 step あたり時間 ほぼ同等 (token 数は短くなるが 1 step 内の batch サイズは不変)。

### Phase 6-b (vocab 16K, d=512)

`aozora_meiji_taisho_charbpe16k_max512`。 vocab を 16,000 に拡張し、 sequence 圧縮を強める。 推論時の埋め込み層が ~4M params 増えるため、 メモリと per-step 時間が +5-10% 程度。

### Tokenizer 学習の現実的考慮

現在の BPE 学習アルゴリズム (`count_pairs` + `merge_vocab` を merge 回数だけ反復) は **O(N_merges × vocab_entries × avg_pair_per_entry)** で大規模 corpus に遅い。
試行錯誤の結果、 次の組み合わせで実用化:

#### 1. サンプル学習 + 全 char カバレッジ (`train_with_coverage`)

`CharBpeTokenizer::train_with_coverage(merge_text, coverage_text, vocab_size)`:
- `merge_text` = サンプル (500K char) → merge 学習を高速化
- `coverage_text` = 全コーパス (8.3M char) → 全 char カバレッジ保証 (UNK 発生なし)

高頻度 n-gram は 500K サンプルで十分に統計収束しているため、 merge 品質への影響は無視できる (実測で確認済)。

#### 2. `count_pairs` / `merge_vocab` の rayon 並列化

`par_iter` で HashMap 集計と置換を並列化。 ただし HashMap allocation/clone が dominant のため
**速度向上は 1% 程度 (期待外れ)**。 将来 incremental update に置き換える価値あり。

#### 3. **トークナイザのディスクキャッシュ**

`build_or_load_tokenizer(cfg, corpus_text)` で:
- `tokenizers/{kind}_v{vocab}_{corpus_stem}_s{sample}.bin` の有無を確認
- ファイルあり → load (~ms)
- ファイル無し → train (5 min) → save

これにより **2 回目以降のフェーズ起動が瞬時** になる。

#### 実測時間 (Phase 6-a: vocab=8000, sample=500K char, full coverage 8.3M char)

| Phase | 動作 | 時間 |
|-------|------|------|
| 1 回目 | merge 学習 + cache 保存 | **318.4 s (5.3 min)** |
| **2 回目以降** | **cache ロード** | **0.00 s (即時)** ✅ |
| (参考) 全コーパスフル学習 | 中断 (>30 min 経過しても未完) | — |

キャッシュ詳細:
- ファイル形式: 既存 `WeightMap` バイナリフォーマットを再利用 (Checkpointable trait)
- サイズ: **~163 KB** (vocab 8000 のとき、 token strings + merges)
- 拡張性: `CharBpeTokenizer::extend_merges` / `extend_coverage` で
  既存トークナイザに merge / char を追加できる (実装済、 4 テスト pass)

### キャッシュの拡張 API (実装済)

| API | 用途 | 計算量 |
|-----|------|--------|
| `extend_merges(text, new_vocab_size)` | 既存トークナイザに BPE 続きから merges を追加 | O(`new_vocab - cur_vocab` × N_words) |
| `extend_coverage(text)` | 新規 char を vocab に追加 (merges 不変) | O(text.chars().count()) |

**使用例**: vocab 8K → 16K の拡大 (Phase 6-a → 6-b)

```rust
// 1. 8K cache をロード
let mut tok = CharBpeTokenizer::empty();
let map = WeightMap::load("tokenizers/charbpe_v8000_aozora_meiji_taisho_s500000.bin")?;
tok.from_weight_map(&map)?;

// 2. 同じサンプルで merges を追加 (約 5-10 min、 ゼロからより半減)
let sample: String = corpus.chars().take(500_000).collect();
tok.extend_merges(&sample, 16_000);

// 3. 新 cache として保存
save_tokenizer_to_file(&tok, "tokenizers/charbpe_v16000_..._s500000.bin")?;
```

**保証される性質**:
- 既存 token ID は不変 (新 token は末尾の新 ID に割り当て)
- 既存 merges の優先順位 (rank) は保持
- encode/decode の lossless 性は維持

**制限**:
- vocab_size 縮小はサポートしない
- 拡張後のトークナイザは **既存モデルでは使えない** (embedding/output head のサイズ不一致)
  → 拡張は **新 phase 用のトークナイザ準備** に使う

### 実行手順 (再現)

```bash
# 1. Phase 6-a 学習 (main.rs の cfg を切り替えて実行)
cargo run --release 2>&1 | tee logs/phase6a_aozora_meiji_taisho_d512_n6_charbpe8k_rms_swiglu_rope_max512.log

# 2. 学習中 / 終了後に推論
cargo run --release   # main.rs 内で best.bin を読んで infer

# 3. BPC 算出
val_loss / ln(2) × (token 数 / char 数) で算出 (token 数はログから, char 数は corpus 文字数)
```

## Phase 6-a 結果 (✅ 完了)

### Validation 推移 (3,000 step, 200 step ごと)

| step | val_loss | val_ppl | **BPC** | best |
|------|---------|---------|---------|------|
| 200 | 5.4970 | 244.20 | 4.80 | ✅ |
| 600 | 4.5769 | 97.21 | 4.00 | ✅ |
| 1000 | 4.4178 | 82.91 | 3.86 | ✅ |
| 1400 | 4.3521 | 77.64 | 3.80 | ✅ |
| 1800 | 4.3183 | 75.06 | 3.77 | ✅ |
| 2000 | 4.3167 | 74.94 | 3.77 | ✅ |
| 2200 | 4.3123 | 74.61 | 3.77 | ✅ |
| **2400** | **4.2999** | **73.70** | **3.76** | ✅ ⭐ **best** |
| 2600 | 4.3140 | 74.74 | 3.77 | — (悪化) |
| 2800 | 4.3121 | 74.59 | 3.77 | — |
| 3000 | 4.3203 | 75.21 | 3.77 | — (終端微悪化) |

best 到達は **step 2400** (中盤)。 以降 train_loss は強く改善 (3.491 → 3.372, -3.4%) が続いたものの val が止まり、 train-val gap = 0.95 まで拡大して **軽度オーバーフィット入り**。

### vs Phase 5-4a (Char tokenizer) 最終比較

| 指標 | Phase 5-4a (Char) | **Phase 6-a (CharBPE)** | 改善 |
|------|------------------|-----------------------|-----|
| best step | 2800 | **2400** | -400 step (より早く到達) |
| best val_ppl | 18.16 | 73.70 | (token 単位が違うため絶対比較不可) |
| **best BPC** | **4.18** | **3.76** | **-10.0%** ⭐ |
| total time | ~7h 50min | **~8h 10min** | +20 min (許容範囲) |
| token/char (実効圧縮率) | 1.00 | **0.61** | -39% |
| 実効コンテキスト (max_len=512) | 512 char | **~840 char** | +64% |

**結論**: BPC で **-10.0% の明確な改善** を達成。 1 step あたりの計算量はほぼ同じまま、 1 step で扱える文字数が **+64%** に増えた。

### 全 Phase 累積 BPC 推移

| Phase | コーパス | 構成 | best BPC |
|-------|--------|------|---------|
| Phase 4b | 漱石 7 作品 (1.21M char) | d=384 char | ~3.71 |
| Phase 5-2 | 同上 | d=384 char + RMS+SwiGLU+RoPE | 3.40 |
| Phase 5-3 | 6 作家 8.3M char | d=384 char | 4.36 |
| Phase 5-4a | 同上 | d=512 char | 4.18 |
| **Phase 6-a** | **同上** | **d=512 CharBPE 8K** | **🏆 3.76** |

Phase 5-3/5-4 はコーパスが約 7 倍に拡大したため難易度が上がり BPC が悪化していたが、 **Phase 6-a でようやく 8.3M char コーパスで Phase 5-2 (1.21M) の BPC 3.40 に近付いた**。

### 質的成果 (生成サンプル)

Phase 5-4a 比で **作家固有の人名・固有名詞・文体** の精度が大きく向上。 step 3000 inference サンプルから:

| プロンプト | 出力ハイライト | 出典 (完全一致) |
|----------|---------------|----------------|
| 「私は」 top-p | 「**津田**さんから」 「**百合子**」 「**新宿の店**」 | **漱石『明暗』** (津田は主人公) |
| 「東京の」 top-p | 「余は鉄壁も眼の前にいる」 「**脳病**」 「**高柳君**」 | **漱石『野分』** (高柳君は主人公) |
| 「それから」 top-k | 「**お延**は苦笑する」 「電話の通う音」 | **漱石『明暗』** (お延は津田の妻) |
| 「ジョバンニ」 top-k | 「**カムパネルラ**のお母さん」 「**苹果**を一つずつ」 | **賢治『銀河鉄道の夜』** (苹果は重要小道具) |
| 「吾輩は」 top-k | 「**主人**」 「**御嬢さん**」 「**猫の癖に猫**」 | **『吾輩は猫である』** |
| 「私は」 top-k | 「**直治**」 「**お母さま**」 「**お熱**」 「**お指図**」 | **太宰『斜陽』** |
| 「メロスは」 top-p (擬古文) | 「神の身の上を思い起こしぬ」 「**夏霧深き紅の木となりぬ**」 「われはその美しき花のごとく」 | **国木田独歩 / 鴎外『舞姫』風** |

特に **「津田 + お延」 (『明暗』 夫妻)**、 **「高柳君」 (『野分』 主人公)**、 **「カムパネルラ + 苹果」 (『銀河鉄道』 名場面)** などの **作品横断の関係性** を正確に再現できた点が、 char tokenizer (Phase 5-4a) からの定性的飛躍。

### 残課題

| 課題 | 観察 | 解決候補 |
|------|------|---------|
| 戯曲記号 (「(伝兵衛)」 等) の混入 | 終始解消せず | コーパス前処理 (戯曲行の除去 / マーキング) |
| 作家ヘッダ (「===== 森鴎外 …」) の出力 | 一部で出現 | 学習時にヘッダ行を BOS/special token 化 |
| 作家ミックス (「ジョバンニ」 → 漱石論文体など) | 高頻度 | `<author=...>` のような **作家トークン** 導入 |
| 長文 (max_new=200+) で repetition collapse | step 1000-1800 では持続、 終盤は緩和 | repetition penalty 強化 / contrastive search |
| 軽度オーバーフィット | step 2400 以降 train-val gap 拡大 | dropout 強化 / 早期停止 |

## Phase 6-b: 起動するも振り替え (vocab=16K)

vocab を 8K → 16K に拡張する案。 起動して tokenizer 訓練 (1M sample, 33.7 min) と corpus encoding まで完了したが、 **圧縮率の実測がきわめて期待外れだったため学習は中断**。

### 実測 (起動時ログより)

| 指標 | 期待 | **実測** | ギャップ |
|------|-----|---------|---------|
| chars/token | 2.3-2.7 | **1.72** (Phase 6-a の 1.65 から +4.2%) | **-30%** |
| 実効コンテキスト (max_len=512) | 1,200-1,400 char | 880 char | -30% |
| corpus_tokens | ~3.5M | **4.32M** | tokens は 14% しか減らない |

### なぜ期待より低かったか

1. **日本語の高頻度 n-gram は ~3,000 で saturate**:
   - 助詞・助動詞・活用語尾などの上位パターンは数百〜数千種類
   - Phase 6-a (vocab=8K, merges 2,774) で既に大半を吸収
   - 追加 8,000 merges は裾尾の固有名詞・低頻度複合語で占有率が低い
2. **Zipf 則の限界収益逓減**:
   - 上位 2K merges で全 token の ~70% カバー
   - 上位 16K merges でも ~92% (16K は +7% しか coverage を伸ばせない)

→ vocab スケーリングは ROI が悪く、 **コンテキスト拡張のほうが筋が良い** と判断。

## Phase 6-c: max_len 512 → 1024 (vocab=8K 据え置き)

Phase 6-a で確立した CharBPE 8K tokenizer (cache 既存) を流用し、 **コンテキスト窓を 2 倍**に拡張する。 Phase 6-b の代替として採用。

| 設定項目 | Phase 6-a | **Phase 6-c** | 変更理由 |
|---------|----------|--------------|---------|
| tokenizer | CharBPE 8K | CharBPE 8K (cache 流用) | 訓練時間 0、 model 互換性維持 |
| max_len | 512 | **1,024** | RoPE 拡張、 attention の context 範囲を倍増 |
| batch_size | 32 | **16** (半減) | per-step 時間とメモリを抑制 |
| 1 step あたり token | 16,384 | 16,384 (32×512 = 16×1024) | データ流量は不変 |
| end_step | 3,000 | 3,000 | Phase 6-a と直接比較するため |
| 実効コンテキスト | **845 char** | **~1,690 char** (+100%) | ⭐ vocab=16K の +4% より遥かに大きい |

### 期待効果

- 実効コンテキスト 845 char → **1,690 char**: 漱石短編 1 段落 (~600 char) の前後関係を完全に保持
- attention の self-similarity による **長文の論理一貫性向上** (Phase 6-a の repetition collapse 緩和)
- BPC 改善見込み: **3.66-3.72** (Phase 6-a 3.76 比 -1〜-3%)
- per-step 時間: ~13 s 想定 (Phase 6-a 9.85s + 30%、 attention 4x + FFN 2x、 batch 半減で打ち消し)
- 総学習時間: ~10-12 h

### 技術的注意

- RoPE は max_len まで sin/cos table を precompute するため、 model 構築時に max_len=1024 を指定するだけで動作 (`src/rope.rs`)
- 既存 tokenizer cache (`tokenizers/charbpe_v8000_aozora_meiji_taisho_s500000.bin`) はそのまま使える
- 学習中の BPC ログ出力は今回追加 (`# val step=N val_loss=... val_ppl=... bpc=...`)

### 起動結果と中断

Phase 6-c は step 180 (elapsed 33 min) で per-step ~11,000 ms を確認。 3000 step 完走まで **約 9.2 h** の見込みだったため、 ユーザー判断で **Phase 7 大規模高速化** に着手し、 完了後 [Phase 6-d](#phase-6-d-wsd--phase-7-高速化-適用) へ振替。

| step | val_ppl | BPC | 備考 |
|------|---------|-----|------|
| 180  | (val 未到達) | — | per-step 10,500-11,500 ms。 中断。 |

## Phase 6-d: WSD + Phase 7 高速化 適用

Phase 6-c と完全に同じ形状を **Phase 7 で高速化したバイナリ + WSD スケジューラ** で再起動する。

| 設定項目 | Phase 6-c | **Phase 6-d** | 狙い |
|---------|-----------|--------------|------|
| 形状 | d=512, n=6, max_len=1024, vocab=8K | 同 | (互換) |
| LR scheduler | warmup-cosine | **warmup-stable-decay** (warmup=300, stable=2160, decay=540) | 同 BPC を 15-30% 早く到達 (MiniCPM/DeepSeek 慣例) |
| バイナリ | 旧 (`Vec<Vec<f32>>` API) | **Phase 7** (Matrix 直叩き + QKV 融合) | per-step 1.2-1.4x |
| per-step 想定 | ~11,000 ms | **~9,000 ms** | ☆ |
| 総時間 想定 | ~9.2 h | **~6-7 h** | -25 〜 -30% |
| Checkpoint 互換 | — | Phase 6-c の best.bin もロード可 (MHA は w_q/w_k/w_v 3 分割を組み立て) | resume 安全 |

### 起動

```bash
cargo run --release 2>&1 | tee logs/phase6d_aozora_meiji_taisho_d512_n6_charbpe8k_rms_swiglu_rope_max1024_wsd.log
```

### 観測ポイント

- ログヘッダに `lr_schedule=WarmupStableDecay { stable_steps: 2160 }` が出ること
- 序盤 step 60-100 で per-step 8,500-9,500 ms 程度に収束 (Phase 6-c より明確に短い)
- step 200 / 400 / 600 の val_ppl と BPC が Phase 6-a (CharBPE 8K, max_len=512) より **同 step で良い** はず (max_len 倍増効果)
- 終盤 step 2,400+ (decay 開始 = step 2,460 以降) で lr が `1 - sqrt(progress)` で急減し、 finetune 効果で BPC がさらに -1 〜 -2% 改善する想定

## 関連ドキュメント

- [Phase 5: 生成品質向上](phase5.md) — 直前のフェーズ。 Phase 5-4a の結果と Phase 6 着手の動機
- [Roadmap](roadmap.md) — 全体俯瞰
- [SoTA 比較](sota_comparison.md) — トークナイザの位置付け (現代 LLM は SentencePiece / BPE が一般的)
