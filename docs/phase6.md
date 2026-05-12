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

## 残課題と次の判断

学習完了後、 次のいずれかに進む:

- **BPC が Phase 5-4a を下回った場合**: Phase 6-b (vocab 16K) でさらに圧縮を試す、 もしくは Phase 5-4b (d=512, L=8) のモデル拡大に戻る
- **下回らなかった場合**: BPE の merge アルゴリズム見直し (頻度のみ → likelihood 最大化、 sentence boundary tokens の追加、 等)、 もしくはモデル拡大に戻る

## 関連ドキュメント

- [Phase 5: 生成品質向上](phase5.md) — 直前のフェーズ。 Phase 5-4a の結果と Phase 6 着手の動機
- [Roadmap](roadmap.md) — 全体俯瞰
- [SoTA 比較](sota_comparison.md) — トークナイザの位置付け (現代 LLM は SentencePiece / BPE が一般的)
