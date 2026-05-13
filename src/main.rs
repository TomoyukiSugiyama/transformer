mod adam_w;
mod bpe_tokenizer;
mod char_bpe_tokenizer;
mod char_tokenizer;
mod cross_entropy_loss;
mod dropout;
mod embedding;
mod eval;
mod feed_forward;
mod feed_forward_network;
mod kv_cache;
mod language_model;
mod layer_normalization;
mod matrix;
mod multi_head_attention;
mod normalization;
mod output_head;
mod positional_encoding;
mod root_mean_square_layer_normalization;
mod rope;
mod sinusoidal_pe;
mod swiglu_feed_forward_network;
mod tokenizer;
mod transformer;
mod transformer_block;

mod checkpoint;
mod lr_scheduler;

use std::fs;
use std::io::Write;
use std::time::Instant;

use rand::{RngExt, SeedableRng, rngs::SmallRng};

use crate::{
    adam_w::AdamW, feed_forward::FeedForwardKind, language_model::LanguageModel,
    lr_scheduler::{LrScheduleKind, LrScheduler},
    multi_head_attention::MultiHeadAttention,
    normalization::NormalizationKind, positional_encoding::PositionalEncodingKind,
    tokenizer::{
        Tokenizer, TokenizerKind, load_tokenizer_from_file, save_tokenizer_to_file,
        train_tokenizer, train_tokenizer_with_coverage,
    },
};

/// コーパスを生のテキストとして読み込む（改行・空行を含む元の構造を保つ）
fn load_corpus(path: &str) -> String {
    fs::read_to_string(path).unwrap_or_else(|e| panic!("corpus file '{}' not found: {}", path, e))
}

/// コーパスを (train, val) に分割する。 char 数で `val_ratio` 比率を末尾に切り出す。
/// nanoGPT (Shakespeare-char) の `prepare.py` と同じ「単純な末尾切り取り」方式で、
/// 学習用テキストには val 部分のテキストが一切含まれないようにする。
fn load_corpus_split(path: &str, val_ratio: f32) -> (String, String) {
    let text = load_corpus(path);
    if val_ratio <= 0.0 {
        return (text, String::new());
    }
    let chars: Vec<char> = text.chars().collect();
    let val_chars = ((chars.len() as f32) * val_ratio).round() as usize;
    let split = chars.len().saturating_sub(val_chars);
    let train: String = chars[..split].iter().collect();
    let val: String = chars[split..].iter().collect();
    (train, val)
}

/// `cfg.tokenizer_cache_path()` にトークナイザが保存済みならロードし、 そうでなければ学習して保存する。
/// `merge_sample_chars` が `Some(n)` のとき、 BPE / CharBpe の merge 学習は先頭 `n` char のサンプルで実施し、
/// 全 char カバレッジは `corpus_text` 全体で保証する (`train_tokenizer_with_coverage`)。
fn build_or_load_tokenizer(cfg: &Config, corpus_text: &str) -> Box<dyn Tokenizer> {
    let cache_path = cfg.tokenizer_cache_path();
    if std::path::Path::new(&cache_path).exists() {
        let t0 = Instant::now();
        match load_tokenizer_from_file(&cache_path) {
            Ok(tok) => {
                println!(
                    "# loaded cached tokenizer from {cache_path} (vocab={}) in {:.2}s",
                    tok.vocab_size(),
                    t0.elapsed().as_secs_f32()
                );
                return tok;
            }
            Err(e) => {
                eprintln!(
                    "# WARN: failed to load tokenizer cache ({cache_path}): {e}. retraining..."
                );
            }
        }
    }

    let t0 = Instant::now();
    let tokenizer = if let Some(sample_chars) = cfg.merge_sample_chars {
        let sample_text: String = corpus_text.chars().take(sample_chars).collect();
        println!(
            "# training tokenizer: kind={:?}, vocab_size={}, sample_chars={} (coverage on full {} chars)",
            cfg.tokenizer_kind,
            cfg.vocab_size,
            sample_text.chars().count(),
            corpus_text.chars().count(),
        );
        train_tokenizer_with_coverage(cfg.tokenizer_kind, &sample_text, corpus_text, cfg.vocab_size)
    } else {
        println!(
            "# training tokenizer: kind={:?}, vocab_size={} (full corpus, {} chars)",
            cfg.tokenizer_kind,
            cfg.vocab_size,
            corpus_text.chars().count(),
        );
        train_tokenizer(cfg.tokenizer_kind, corpus_text, cfg.vocab_size)
    };
    let train_secs = t0.elapsed().as_secs_f32();
    println!(
        "# tokenizer trained in {train_secs:.1}s (vocab={})",
        tokenizer.vocab_size()
    );

    match save_tokenizer_to_file(tokenizer.as_ref(), &cache_path) {
        Ok(()) => println!("# saved tokenizer cache to {cache_path}"),
        Err(e) => eprintln!("# WARN: failed to save tokenizer cache to {cache_path}: {e}"),
    }
    tokenizer
}

struct Config {
    run_name: &'static str,
    /// 学習・推論で読み込む corpus テキストファイルへのパス
    corpus_path: &'static str,
    tokenizer_kind: TokenizerKind,
    normalization_kind: NormalizationKind,
    feed_forward_kind: FeedForwardKind,
    positional_encoding_kind: PositionalEncodingKind,
    d_model: usize,
    n_heads: usize,
    d_ff: usize,
    n_layers: usize,
    max_len: usize,
    /// BPE のときのみ参照される。 char-level では corpus の文字種から自動決定。
    vocab_size: usize,
    lr_max: f32,
    lr_min: f32,
    warmup_steps: usize,
    end_step: usize,
    /// 学習率スケジュールの種類。 デフォルトは `WarmupCosine` (既存挙動互換)。
    /// `WarmupStableDecay { stable_steps }` で WSD (warmup-stable-decay) に切替。
    /// 同じ BPC を 15-30% 早く到達できる場合があり、 さらに total_steps を伸ばしても
    /// 過去 step の lr 軌道が変わらないため checkpoint からの追加学習に向く。
    lr_schedule_kind: LrScheduleKind,
    save_every: usize,
    log_every: usize,
    /// 0 のとき val を計測しない。 それ以外なら毎 `val_every` step で
    /// `val_n_batches` 個のランダム窓に対して val_loss を計測。
    val_every: usize,
    val_n_batches: usize,
    val_split_ratio: f32,
    batch_size: usize,
    /// dropout 率 (0.0 で無効)。 nanoGPT Shakespeare-char は 0.2。
    dropout: f32,
    /// AdamW の weight decay。 nanoGPT は 0.1 を使用。
    weight_decay: f32,
    /// AdamW の beta2。 1 step あたり tokens が少ないコーパスでは 0.99 が推奨。
    beta2: f32,
    prompts: Vec<&'static str>,
    /// `CharBpe` の merge 学習に使う char 数 (先頭からサンプリング)。
    /// `None` のときコーパス全体で学習。 サンプル学習で速度を稼ぐ場合に `Some(500_000)` 等を指定。
    /// `Char` / `Bpe` では無視される。
    merge_sample_chars: Option<usize>,
}

impl Config {
    #[allow(dead_code)]
    fn tiny_shakespeare() -> Self {
        let prompts = vec!["I have seen", "O Romeo", "To be or not to be", "What news"];

        Self {
            run_name: "phase2_d256_ff1024_max128_with_accelerate",
            corpus_path: "corpus/tiny_shakespeare.txt",
            tokenizer_kind: TokenizerKind::Bpe,
            normalization_kind: NormalizationKind::Layer,
            feed_forward_kind: FeedForwardKind::Gelu,
            positional_encoding_kind: PositionalEncodingKind::Sinusoidal,
            d_model: 256,
            n_heads: 8,
            d_ff: 1024,
            n_layers: 4,
            max_len: 128,
            vocab_size: 4000,
            lr_max: 3e-4,
            lr_min: 1e-5,
            warmup_steps: 200,
            end_step: 10000,
            save_every: 500,
            log_every: 20,
            val_every: 0,
            val_n_batches: 16,
            val_split_ratio: 0.0,
            batch_size: 16,
            dropout: 0.0,
            weight_decay: 0.01,
            beta2: 0.999,
            prompts,
            merge_sample_chars: None,
            lr_schedule_kind: LrScheduleKind::WarmupCosine,
        }
    }

    /// nanoGPT (Shakespeare-char) と同等構成 (Phase 3 ターゲット)。
    /// パラメータ ~10.7M、 char tokenizer (vocab はコーパスから自動)。
    #[allow(dead_code)]
    fn nano_gpt_equivalent() -> Self {
        let prompts = vec!["I have seen", "O Romeo", "To be or not to be", "What news"];

        Self {
            run_name: "phase3_nanogpt_equiv_d384_n6_char",
            corpus_path: "corpus/tiny_shakespeare.txt",
            tokenizer_kind: TokenizerKind::Char,
            normalization_kind: NormalizationKind::Layer,
            feed_forward_kind: FeedForwardKind::Gelu,
            positional_encoding_kind: PositionalEncodingKind::Sinusoidal,
            d_model: 384,
            n_heads: 6,
            d_ff: 1536,
            n_layers: 6,
            max_len: 256,
            vocab_size: 0, // unused for Char
            lr_max: 1e-3,
            lr_min: 1e-4,
            warmup_steps: 100,
            end_step: 5000,
            save_every: 500,
            log_every: 20,
            val_every: 100,
            val_n_batches: 16,
            val_split_ratio: 0.1,
            batch_size: 64,
            dropout: 0.2,
            weight_decay: 0.1,
            beta2: 0.99,
            prompts,
            merge_sample_chars: None,
            lr_schedule_kind: LrScheduleKind::WarmupCosine,
        }
    }

    /// Phase 4: 夏目漱石「こころ」 (青空文庫, ~162k char / ~484k UTF-8 bytes) を Char tokenizer で学習。
    /// nanoGPT 相当のアーキ (`nano_gpt_equivalent`) を流用しつつ、 コーパスが Tiny Shakespeare の
    /// 1/7 規模なので過学習が早く来る (前回 BPE 試走で val 最良 step 200, 過学習 step 300+)。
    /// そのため end_step を 1000、 save_every を 100 にして val 最良点を逃さないようにする。
    ///
    /// BPE byte-level だと日本語 (1 char = 3 byte) でマージが UTF-8 境界を跨いで
    /// decode 時に文字化けが出るため、 Char tokenizer (vocab はコーパス文字種から自動) に切替。
    ///
    /// `scripts/download_aozora_kokoro.sh` で corpus/aozora_kokoro.txt を生成しておくこと。
    #[allow(dead_code)]
    fn aozora_kokoro() -> Self {
        let prompts = vec!["私は", "先生は", "ある日", "東京の"];

        Self {
            run_name: "phase4_aozora_kokoro_d384_n6_char",
            corpus_path: "corpus/aozora_kokoro.txt",
            tokenizer_kind: TokenizerKind::Char,
            normalization_kind: NormalizationKind::Layer,
            feed_forward_kind: FeedForwardKind::Gelu,
            positional_encoding_kind: PositionalEncodingKind::Sinusoidal,
            d_model: 384,
            n_heads: 6,
            d_ff: 1536,
            n_layers: 6,
            max_len: 256,
            vocab_size: 0, // unused for Char (コーパスの文字種から自動算出)
            lr_max: 1e-3,
            lr_min: 1e-4,
            warmup_steps: 100,
            end_step: 1000,
            save_every: 100,
            log_every: 20,
            val_every: 100,
            val_n_batches: 16,
            val_split_ratio: 0.1,
            batch_size: 64,
            dropout: 0.2,
            weight_decay: 0.1,
            beta2: 0.99,
            prompts,
            merge_sample_chars: None,
            lr_schedule_kind: LrScheduleKind::WarmupCosine,
        }
    }

    /// Phase 4 拡張: 夏目漱石主要長編 7 作品 (青空文庫, ~1.21M char) を Char tokenizer で学習。
    /// 取得スクリプト: `scripts/download_aozora_soseki_works.sh`
    /// 含まれる作品 (全て新字新仮名): 吾輩は猫である / 坊っちゃん / 草枕 / 三四郎 / 行人 / こころ / 道草
    /// 参考: ユニーク文字数 ~3720 (Phase 4 こころ単独 ~2300 の 1.6 倍, Phase 3 英語 65 の ~57 倍)。
    /// コーパスサイズが Tiny Shakespeare とほぼ同じなので Phase 3 設定をベースに、
    /// vocab 増加分の余裕を見て end_step を 2000 (Phase 3 の 5000 は過剰) に短縮。
    #[allow(dead_code)]
    fn aozora_soseki_works() -> Self {
        let prompts = vec!["私は", "先生は", "ある日", "東京の", "吾輩は", "それから"];

        Self {
            run_name: "phase4b_aozora_soseki_works_d384_n6_char_rms_swiglu_rope",
            corpus_path: "corpus/aozora_soseki_works.txt",
            tokenizer_kind: TokenizerKind::Char,
            normalization_kind: NormalizationKind::Rms,
            feed_forward_kind: FeedForwardKind::SwiGlu,
            positional_encoding_kind: PositionalEncodingKind::Rope,
            d_model: 384,
            n_heads: 6,
            d_ff: 1536,
            n_layers: 6,
            max_len: 256,
            vocab_size: 0, // unused for Char (~3720 自動算出)
            lr_max: 1e-3,
            lr_min: 1e-4,
            warmup_steps: 100,
            end_step: 2000,
            save_every: 200,
            log_every: 20,
            val_every: 100,
            val_n_batches: 16,
            val_split_ratio: 0.1,
            batch_size: 64,
            dropout: 0.2,
            weight_decay: 0.1,
            beta2: 0.99,
            prompts,
            merge_sample_chars: None,
            lr_schedule_kind: LrScheduleKind::WarmupCosine,
        }
    }

    /// Phase 5-2: 漱石 7 作品 + max_len 512 拡張 (RoPE の真価検証用)。
    ///
    /// `aozora_soseki_works` から `max_len: 256 → 512`、 `batch_size: 64 → 32` (メモリ補正)
    /// のみを変更。 1 step あたりの token 数 (max_len × batch_size = 16384) は変えず、
    /// attention の計算量増加分 (seq² で 4x) を batch 半減でほぼ相殺。
    ///
    /// 期待: 段落単位 (10-20 文) の文脈が見えるようになり、 RoPE の相対位置注入が
    /// max_len=256 時より効くため val_ppl が 17.5〜18.0 帯まで下がる可能性。
    /// 同時に、 RoPE vs Sinusoidal の差が顕在化する場面でもある。
    #[allow(dead_code)]
    fn aozora_soseki_works_max512() -> Self {
        let prompts = vec!["私は", "先生は", "ある日", "東京の", "吾輩は", "それから"];

        Self {
            run_name: "phase5b_aozora_soseki_works_d384_n6_char_rms_swiglu_rope_max512",
            corpus_path: "corpus/aozora_soseki_works.txt",
            tokenizer_kind: TokenizerKind::Char,
            normalization_kind: NormalizationKind::Rms,
            feed_forward_kind: FeedForwardKind::SwiGlu,
            positional_encoding_kind: PositionalEncodingKind::Rope,
            d_model: 384,
            n_heads: 6,
            d_ff: 1536,
            n_layers: 6,
            max_len: 512,
            vocab_size: 0,
            lr_max: 1e-3,
            lr_min: 1e-4,
            warmup_steps: 100,
            end_step: 1500,
            save_every: 100,
            log_every: 20,
            val_every: 100,
            val_n_batches: 16,
            val_split_ratio: 0.1,
            batch_size: 32,
            dropout: 0.2,
            weight_decay: 0.1,
            beta2: 0.99,
            prompts,
            merge_sample_chars: None,
            lr_schedule_kind: LrScheduleKind::WarmupCosine,
        }
    }

    /// Phase 5-2 副次実験: max_len=512 + Sinusoidal PE 版 (RoPE 比較用)。
    /// `aozora_soseki_works_max512` から `positional_encoding_kind` のみを `Rope → Sinusoidal` に変更。
    /// これで「max_len 拡張時の RoPE 効果」 を直接測定できる。
    #[allow(dead_code)]
    fn aozora_soseki_works_max512_sinusoidal() -> Self {
        let mut cfg = Self::aozora_soseki_works_max512();
        cfg.run_name = "phase5b_aozora_soseki_works_d384_n6_char_rms_swiglu_sin_max512";
        cfg.positional_encoding_kind = PositionalEncodingKind::Sinusoidal;
        cfg
    }

    /// Phase 5-3: 明治-大正 6 作家 (~3.5-4.5M char) で学習。 モデルは Phase 5-2 と同等。
    /// 取得スクリプト: `scripts/download_aozora_meiji_taisho.sh`
    /// 含まれる作家: 太宰治 / 国木田独歩 / 宮沢賢治 / 中島敦 / 森鴎外 / 夏目漱石 (全て 新字新仮名)
    ///
    /// データが 3-4 倍に増えるので、 過学習開始 step も大幅に後ろ倒し (Phase 5-2 で 600 → 1500+ 予想)。
    /// よって end_step を 3000 に拡張、 save_every は 200 で粒度を緩める。
    /// 文字種は Phase 4b (3720 chars) より増える見込み (4500-5500 chars と推定)。
    #[allow(dead_code)]
    fn aozora_meiji_taisho_max512() -> Self {
        let prompts = vec![
            "私は",
            "先生は",
            "ある日",
            "東京の",
            "吾輩は",
            "それから",
            "メロスは",
            "ジョバンニ",
        ];

        Self {
            run_name: "phase5c_aozora_meiji_taisho_d384_n6_char_rms_swiglu_rope_max512",
            corpus_path: "corpus/aozora_meiji_taisho.txt",
            tokenizer_kind: TokenizerKind::Char,
            normalization_kind: NormalizationKind::Rms,
            feed_forward_kind: FeedForwardKind::SwiGlu,
            positional_encoding_kind: PositionalEncodingKind::Rope,
            d_model: 384,
            n_heads: 6,
            d_ff: 1536,
            n_layers: 6,
            max_len: 512,
            vocab_size: 0,
            lr_max: 1e-3,
            lr_min: 1e-4,
            warmup_steps: 200,
            end_step: 3000,
            save_every: 200,
            log_every: 20,
            val_every: 200,
            val_n_batches: 16,
            val_split_ratio: 0.05,
            batch_size: 32,
            dropout: 0.2,
            weight_decay: 0.1,
            beta2: 0.99,
            prompts,
            merge_sample_chars: None,
            lr_schedule_kind: LrScheduleKind::WarmupCosine,
        }
    }

    /// Phase 5-4a: モデル拡大 (d_model 384 → 512, ~20M params)。 コーパスは Phase 5-3 と同じ。
    /// lr_max を 7e-4 (LLaMA 流の控えめ) に下げ、 warmup を 300 に伸ばす。
    /// 完走時間予測 (M1 Max + Accelerate): ~5 時間 (Phase 5-3 + 1.7-2x compute)。
    #[allow(dead_code)]
    fn aozora_meiji_taisho_d512_max512() -> Self {
        let mut cfg = Self::aozora_meiji_taisho_max512();
        cfg.run_name = "phase5d_aozora_meiji_taisho_d512_n6_char_rms_swiglu_rope_max512";
        cfg.d_model = 512;
        cfg.n_heads = 8;
        cfg.d_ff = 2048;
        cfg.lr_max = 7e-4;
        cfg.lr_min = 7e-5;
        cfg.warmup_steps = 300;
        cfg
    }

    /// Phase 5-4b: モデル拡大 (d_model 512, n_layers 6 → 8, ~26M params)。
    #[allow(dead_code)]
    fn aozora_meiji_taisho_d512_n8_max512() -> Self {
        let mut cfg = Self::aozora_meiji_taisho_d512_max512();
        cfg.run_name = "phase5d_aozora_meiji_taisho_d512_n8_char_rms_swiglu_rope_max512";
        cfg.n_layers = 8;
        cfg
    }

    /// Phase 5-4c: モデル拡大 (d_model 768, n_layers 8, ~50M params)。
    /// 学習時間が 10 時間級になるため batch_size を 16 に下げてメモリ余裕を確保。
    /// lr_max を 5e-4 に下げる (GPT-2 small 124M で使われる慣例)。
    #[allow(dead_code)]
    fn aozora_meiji_taisho_d768_max512() -> Self {
        let mut cfg = Self::aozora_meiji_taisho_d512_n8_max512();
        cfg.run_name = "phase5d_aozora_meiji_taisho_d768_n8_char_rms_swiglu_rope_max512";
        cfg.d_model = 768;
        cfg.n_heads = 12;
        cfg.d_ff = 3072;
        cfg.batch_size = 16;
        cfg.lr_max = 5e-4;
        cfg.lr_min = 5e-5;
        cfg.warmup_steps = 500;
        cfg
    }

    /// Phase 6-a: Phase 5-4a と同一アーキ (d=512, n_layers=6) で **tokenizer を char-level BPE** に切替。
    /// 期待効果: BPE merge により 1 token あたり char 数が増え、 同じ max_len=512 で
    /// **実質的な context window が ~1.8x に拡張**。 さらに頻出 n-gram (「ので」 「ました」 「先生」 等) が
    /// 1 token になることで bigram 切り誤り (例: "ある日本") の減少が期待される。
    ///
    /// vocab_size = 8000 (Phase 5-4a の char vocab 5,220 + merges ~2,780)。
    /// max_len は char 比較を公平にするため 512 のまま (1 token ≈ 1.8 char なので実質 ~922 char)。
    ///
    /// 注意: `val_ppl` の **絶対値** は char tokenizer と比較不能 (vocab inflation で大きくなる)。
    /// 公平比較は **BPC (bits/char)** で行う。
    #[allow(dead_code)]
    fn aozora_meiji_taisho_charbpe8k_max512() -> Self {
        let mut cfg = Self::aozora_meiji_taisho_d512_max512();
        cfg.run_name = "phase6a_aozora_meiji_taisho_d512_n6_charbpe8k_rms_swiglu_rope_max512";
        cfg.tokenizer_kind = TokenizerKind::CharBpe;
        cfg.vocab_size = 8000;
        // BPE 学習は 500K char サンプルで実施 (フル 8.3M だと merge per iter のコストが大、 1+ 時間)。
        // 高頻度 n-gram は 500K サンプルで十分に統計収束する (実測確認済)。
        // 全 char カバレッジは全コーパスで保証 (`train_with_coverage`)。
        cfg.merge_sample_chars = Some(500_000);
        cfg
    }

    /// Phase 6-b: より大きな merge 数で sequence 圧縮を強める (vocab=16K)。
    /// 期待: 1 token ≈ 2.5 char、 実質 context ~1,280 char。
    /// 実測 (起動時): chars/token=1.72 で +4% にとどまる。 BPE は ~3K merges で saturate するため
    /// vocab を倍増しても token 数は -14% しか減らないと判明。 → Phase 6-c に振り替え。
    #[allow(dead_code)]
    fn aozora_meiji_taisho_charbpe16k_max512() -> Self {
        let mut cfg = Self::aozora_meiji_taisho_charbpe8k_max512();
        cfg.run_name = "phase6b_aozora_meiji_taisho_d512_n6_charbpe16k_rms_swiglu_rope_max512";
        cfg.vocab_size = 16000;
        // vocab=16000 だと merge 数 ~10,774。 サンプル拡大して品質を確保。
        cfg.merge_sample_chars = Some(1_000_000);
        cfg
    }

    /// Phase 6-c: vocab=8K (Phase 6-a と同じトークナイザを再利用) で **max_len 512 → 1024** に拡張。
    /// 期待: 実効コンテキスト 845 char → ~1,690 char (+100%, vocab 16K の +4% より遥かに大きい)。
    /// attention は O(n²) で 4x、 FFN/embedding は 2x になるため、 batch_size を 32 → 16 に半減して
    /// per-step 時間を Phase 6-a +30% (~13s) に抑える。 1 step あたりの token 数は不変 (16×1024 = 16,384)。
    #[allow(dead_code)]
    fn aozora_meiji_taisho_charbpe8k_max1024() -> Self {
        let mut cfg = Self::aozora_meiji_taisho_charbpe8k_max512();
        cfg.run_name = "phase6c_aozora_meiji_taisho_d512_n6_charbpe8k_rms_swiglu_rope_max1024";
        cfg.max_len = 1024;
        cfg.batch_size = 16;
        // tokenizer は Phase 6-a の 8K cache を再利用 (merge_sample_chars / vocab_size は変えない)
        cfg
    }

    /// Phase 6-d: Phase 6-c と同じ形状で **WSD スケジューラ + Phase 7 高速化** を適用。
    /// - WSD: warmup=300, stable=2160 (= 80% × (3000-300)), decay=540
    ///   → 大半の step を lr_max で回し、 終盤 18% で 1-sqrt 減衰させる。
    ///   nanoGPT/MiniCPM 流で同じ BPC を 15-30% 早く到達できる場合がある。
    /// - Phase 7 高速化 (Matrix 直叩き + QKV 融合) で per-step ~1.2-1.4x 短縮を見込む。
    /// 期待 total: ~6-7 h (Phase 6-c の ~9 h 比 -25 〜 -30%)。
    #[allow(dead_code)]
    fn aozora_meiji_taisho_charbpe8k_max1024_wsd() -> Self {
        let mut cfg = Self::aozora_meiji_taisho_charbpe8k_max1024();
        cfg.run_name = "phase6d_aozora_meiji_taisho_d512_n6_charbpe8k_rms_swiglu_rope_max1024_wsd";
        cfg.lr_schedule_kind = LrScheduleKind::WarmupStableDecay { stable_steps: 2160 };
        cfg
    }

    fn checkpoint_dir(&self) -> String {
        format!("checkpoints/{}", self.run_name)
    }

    /// トークナイザの永続化パスを返す (再現性のため kind / vocab / sample / corpus 名を埋め込む)。
    /// 同じパラメータの再実行ではこの cache をロードして BPE 学習時間を 5-15 分節約する。
    /// `tokenizer_kind` ごとに用途が違うので Char はキャッシュ不要 (即時)、 BPE/CharBpe のみ恩恵あり。
    fn tokenizer_cache_path(&self) -> String {
        let kind = match self.tokenizer_kind {
            TokenizerKind::Bpe => "bpe",
            TokenizerKind::Char => "char",
            TokenizerKind::CharBpe => "charbpe",
        };
        let corpus_stem = std::path::Path::new(self.corpus_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("corpus");
        let sample = self
            .merge_sample_chars
            .map(|n| format!("_s{n}"))
            .unwrap_or_default();
        format!(
            "tokenizers/{kind}_v{}_{corpus_stem}{sample}.bin",
            self.vocab_size
        )
    }
}
fn main() {
    // Phase 6-d: Phase 6-c と同じ形状 (d=512, n=6, max_len=1024, vocab=8K) に
    //   **WSD scheduler + Phase 7 (Matrix 直叩き + QKV 融合) 高速化** を適用。
    //   期待 total: ~6-7 h (Phase 6-c の ~9 h 比 -25 〜 -30%)、
    //          BPC: Phase 6-c と同等以上 (WSD で同 step 内の収束効率が向上)。
    let cfg = Config::aozora_meiji_taisho_charbpe8k_max1024_wsd();
    // training_and_inference(&cfg);
    inference_from_checkpoint(
        &cfg,
        "checkpoints/phase6d_aozora_meiji_taisho_d512_n6_charbpe8k_rms_swiglu_rope_max1024_wsd/best.bin",
    );

    // Phase 6-c: WarmupCosine 旧バイナリで起動して step 180 まで進めたが、
    //   Phase 7 高速化が完了したので Phase 6-d に振り替え。
    // let cfg = Config::aozora_meiji_taisho_charbpe8k_max1024();
    // training_and_inference(&cfg);

    // Phase 6-b: char-level BPE (vocab=16000) — 起動したが圧縮率が +4% にとどまり、
    //   vocab スケーリングの ROI が低いと判明したため Phase 6-c に振り替え (起動中の run は中断)。
    // let cfg = Config::aozora_meiji_taisho_charbpe16k_max512();
    // training_and_inference(&cfg);

    // Phase 6-a: char-level BPE (vocab=8000) ✅ 完了
    // best val_loss 4.2999 @ step 2400, val_ppl 73.70, BPC 3.76
    // let cfg = Config::aozora_meiji_taisho_charbpe8k_max512();
    // inference_from_checkpoint(
    //     &cfg,
    //     "checkpoints/phase6a_aozora_meiji_taisho_d512_n6_charbpe8k_rms_swiglu_rope_max512/best.bin",
    // );

    // tokenizer の cache 経由動作確認 (training 起動前にトークナイザだけ試したいとき):
    // bench_tokenizer_with_cache(&cfg);

    // let cfg = Config::aozora_meiji_taisho_d512_max512();
    // let mut model = LanguageModel::load_inference_checkpoint(
    //     "checkpoints/phase5d_aozora_meiji_taisho_d512_n6_char_rms_swiglu_rope_max512/best.bin",
    // )
    // .unwrap();
    // bench_kv_cache(&mut model, &cfg.prompts);

    // Phase 5-4a: モデル拡大 d_model 384 → 512 (~20M params), 同コーパス (8.3M char)
    // ✅ 完了: best val_ppl 18.16 @ step 2800, BPC 4.18 (全 phase 最高)
    // let cfg = Config::aozora_meiji_taisho_d512_max512();
    // training_and_inference(&cfg);
    // inference_from_checkpoint(
    //     &cfg,
    //     "checkpoints/phase5d_aozora_meiji_taisho_d512_n6_char_rms_swiglu_rope_max512/best.bin",
    // );

    // Phase 5-3: 明治-大正 6 作家 (8.3M char) + max_len 512 + RMS+SwiGLU+RoPE
    // ✅ 完了: best val_ppl 18.71 @ step 2800 (容量律速で Phase 5-2 17.76 を下回れず)
    // let cfg = Config::aozora_meiji_taisho_max512();
    // inference_from_checkpoint(
    //     &cfg,
    //     "checkpoints/phase5c_aozora_meiji_taisho_d384_n6_char_rms_swiglu_rope_max512/best.bin",
    // );

    // Phase 5-2: 漱石 7 作品 + max_len 512 拡張 (✅ 完了, best val_ppl 17.76 @ step 600)
    // let cfg = Config::aozora_soseki_works_max512();
    // inference_from_checkpoint(
    //     &cfg,
    //     "checkpoints/phase5b_aozora_soseki_works_d384_n6_char_rms_swiglu_rope_max512/best.bin",
    // );

    // Phase 5-2 副次実験: max_len 512 + Sinusoidal PE (RoPE 効果の直接比較)
    // let cfg = Config::aozora_soseki_works_max512_sinusoidal();

    // Phase 5-4b/c: 5-4a 完了後に有効化
    // let cfg = Config::aozora_meiji_taisho_d512_n8_max512();   // 5-4b (~26M params)
    // let cfg = Config::aozora_meiji_taisho_d768_max512();      // 5-4c (~50M params)

    // 過去 phase の checkpoint 推論:
    //
    // Phase 4b: 漱石 7 作品 RMS+SwiGLU+RoPE (max_len=256)
    // let cfg = Config::aozora_soseki_works();
    // inference_from_checkpoint(
    //     &cfg,
    //     "checkpoints/phase4b_aozora_soseki_works_d384_n6_char_rms_swiglu_rope/best.bin",
    // );
    //
    // Phase 4a: 夏目漱石「こころ」 単独
    // let cfg = Config::aozora_kokoro();
    //
    // Phase 3: nanoGPT 相当 char-level Tiny Shakespeare
    // let cfg = Config::nano_gpt_equivalent();
    //
    // Phase 2: BPE Tiny Shakespeare
    // let cfg = Config::tiny_shakespeare();
}

#[allow(dead_code)]
fn inference_from_checkpoint(cfg: &Config, path: &str) {
    let mut model = LanguageModel::load_inference_checkpoint(path).unwrap();
    infer(&mut model, &cfg.prompts);
}

fn run_training_loop(
    model: &mut LanguageModel,
    opt: &mut AdamW,
    rng: &mut SmallRng,
    token_ids: &[usize],
    val_ids: &[usize],
    val_chars: usize,
    cfg: &Config,
    start_step: usize,
) {
    let pad_id = model.pad_id();
    let mut ema_loss: Option<f32> = None;
    let mut window_min = f32::INFINITY;
    let mut window_max = f32::NEG_INFINITY;
    let lr_scheduler = LrScheduler::with_kind(
        cfg.lr_max,
        cfg.lr_min,
        cfg.warmup_steps,
        cfg.end_step,
        cfg.lr_schedule_kind,
    );

    let ckpt_dir = cfg.checkpoint_dir();
    fs::create_dir_all(&ckpt_dir).unwrap();

    let chunk_len = cfg.max_len;
    assert!(
        token_ids.len() > chunk_len,
        "tokenized corpus is shorter than chunk_len; cannot sample windows"
    );
    let max_offset = token_ids.len() - chunk_len;

    let val_enabled = cfg.val_every > 0 && val_ids.len() > chunk_len;
    const VAL_SEED: u64 = 12345;
    // val_loss の最小値を追跡し、 更新時に `best.bin` を保存する。
    // None のとき初回計測 = 自動で best として保存される。
    let mut best_val_loss: Option<f32> = None;

    println!("# run_name={}", cfg.run_name);
    println!(
        "# tokenizer={:?}, d_model={}, n_heads={}, d_ff={}, n_layers={}, max_len={}, vocab_size={}, dropout={}, wd={}, beta2={}",
        model.tokenizer_kind(),
        cfg.d_model,
        cfg.n_heads,
        cfg.d_ff,
        cfg.n_layers,
        cfg.max_len,
        cfg.vocab_size,
        cfg.dropout,
        cfg.weight_decay,
        cfg.beta2,
    );
    println!(
        "# lr_max={}, lr_min={}, warmup_steps={}, end_step={}, lr_schedule={:?}, batch_size={}, log_every={}, save_every={}, start_step={}",
        cfg.lr_max,
        cfg.lr_min,
        cfg.warmup_steps,
        cfg.end_step,
        cfg.lr_schedule_kind,
        cfg.batch_size,
        cfg.log_every,
        cfg.save_every,
        start_step
    );
    println!(
        "# corpus_tokens={}, chunk_len={}, max_offset={}, val_enabled={}, val_tokens={}, val_chars={}, val_chars_per_token={:.4}, val_every={}, val_n_batches={}",
        token_ids.len(),
        chunk_len,
        max_offset,
        val_enabled,
        val_ids.len(),
        val_chars,
        if val_ids.is_empty() {
            0.0
        } else {
            val_chars as f32 / val_ids.len() as f32
        },
        cfg.val_every,
        cfg.val_n_batches,
    );
    println!("step,loss,ema,min,max,ppl,ema_ppl,lr,ms_per_step,elapsed_s");

    let train_start = Instant::now();
    let mut window_start = Instant::now();

    for step in start_step..=cfg.end_step {
        let lr = lr_scheduler.get_lr(step);
        opt.set_lr(lr);
        let mut total_loss = 0.0f32;
        let mut valid_cout = 0;
        for _ in 0..cfg.batch_size {
            let offset = rng.random_range(0..=max_offset);
            let chunk = &token_ids[offset..offset + chunk_len];
            total_loss += model.forward_backward(chunk, pad_id);
            valid_cout += 1;
        }
        let mut avg_loss = 0.0f32;
        if valid_cout > 0 {
            opt.set_grad_scale(valid_cout);
            opt.increment_step();
            model.apply_gradients(opt);
            opt.reset_grad_scale();
            model.zero_grad();
            avg_loss = total_loss / valid_cout as f32;
        }
        let alpha = 0.05;
        ema_loss = Some(match ema_loss {
            Some(e) => e * (1.0 - alpha) + avg_loss * alpha,
            None => avg_loss,
        });
        if valid_cout > 0 {
            window_min = window_min.min(avg_loss);
            window_max = window_max.max(avg_loss);
        }
        if step % cfg.log_every == 0 {
            let window_elapsed = window_start.elapsed();
            let ms_per_step = window_elapsed.as_secs_f64() * 1000.0 / cfg.log_every as f64;
            let elapsed_s = train_start.elapsed().as_secs_f64();
            let ema = ema_loss.unwrap();
            println!(
                "{},{:.6},{:.6},{:.6},{:.6},{:.4},{:.4},{:.3e},{:.1},{:.1}",
                step,
                avg_loss,
                ema,
                window_min,
                window_max,
                eval::perplexity(avg_loss),
                eval::perplexity(ema),
                lr,
                ms_per_step,
                elapsed_s,
            );
            // ファイルにリダイレクト時の block-buffering を回避し、 tail -f で見られるようにする
            let _ = std::io::stdout().flush();
            window_min = f32::INFINITY;
            window_max = f32::NEG_INFINITY;
            window_start = Instant::now();
        }
        if val_enabled && step % cfg.val_every == 0 {
            let val_loss =
                eval::compute_val_loss(model, val_ids, chunk_len, cfg.val_n_batches, VAL_SEED);
            let val_ppl = eval::perplexity(val_loss);
            // BPC = bits per character. tokenizer 横断で公平比較できる正規化指標。
            // val_loss は per-token の平均 nats なので、 tokens/chars 比で per-char に換算。
            let bpc = if val_chars > 0 && !val_ids.is_empty() {
                val_loss / std::f32::consts::LN_2 * (val_ids.len() as f32 / val_chars as f32)
            } else {
                f32::NAN
            };
            println!(
                "# val step={} val_loss={:.6} val_ppl={:.4} bpc={:.4}",
                step, val_loss, val_ppl, bpc
            );
            let _ = std::io::stdout().flush();

            // val_loss が改善 (= 過去最小を更新) したら best.bin を保存
            let is_best = best_val_loss.map_or(true, |prev| val_loss < prev);
            if is_best {
                let prev_repr = best_val_loss
                    .map(|p| format!("{:.6}", p))
                    .unwrap_or_else(|| "(none)".to_string());
                best_val_loss = Some(val_loss);
                let best_path = format!("{ckpt_dir}/best.bin");
                model
                    .save_training_checkpoint(&best_path, opt, step)
                    .unwrap();
                println!(
                    "# best updated: step={} val_loss={:.6} val_ppl={:.4} bpc={:.4} (prev val_loss={}) -> saved {}",
                    step, val_loss, val_ppl, bpc, prev_repr, best_path
                );
                let _ = std::io::stdout().flush();
            }
        }
        if step % cfg.save_every == 0 {
            let path = format!("{ckpt_dir}/step_{step:06}.bin");
            let latest_path = format!("{ckpt_dir}/latest.bin");
            model.save_training_checkpoint(&path, opt, step).unwrap();
            model
                .save_training_checkpoint(&latest_path, opt, step)
                .unwrap();
            println!("saved: {path}");
        }
    }
}

#[allow(dead_code)]
fn training_and_inference(cfg: &Config) {
    let (train_text, val_text) = load_corpus_split(cfg.corpus_path, cfg.val_split_ratio);
    let tokenizer = build_or_load_tokenizer(cfg, &train_text);
    let mut model = LanguageModel::from_tokenizer(
        tokenizer,
        cfg.normalization_kind,
        cfg.feed_forward_kind,
        cfg.positional_encoding_kind,
        cfg.d_model,
        cfg.n_heads,
        cfg.d_ff,
        cfg.n_layers,
        cfg.max_len,
        cfg.dropout,
    );
    let token_ids = model.tokenize_corpus(&train_text);
    let val_ids = if val_text.is_empty() {
        Vec::new()
    } else {
        model.tokenize_corpus(&val_text)
    };
    let val_chars = val_text.chars().count();
    let mut opt = AdamW::new_with_wd(cfg.lr_max, cfg.weight_decay);
    opt.set_beta2(cfg.beta2);
    let mut rng = SmallRng::seed_from_u64(42);
    run_training_loop(
        &mut model,
        &mut opt,
        &mut rng,
        &token_ids,
        &val_ids,
        val_chars,
        cfg,
        1,
    );
    let inference_path = format!("{}/inference.bin", cfg.checkpoint_dir());
    model.save_inference_checkpoint(&inference_path).unwrap();
    infer(&mut model, &cfg.prompts);
}

#[allow(dead_code)]
fn training_from_checkpoint(cfg: &Config, path: &str) {
    let (train_text, val_text) = load_corpus_split(cfg.corpus_path, cfg.val_split_ratio);
    let (mut model, mut opt, checkpoint_step) =
        LanguageModel::load_training_checkpoint(path).unwrap();
    let token_ids = model.tokenize_corpus(&train_text);
    let val_ids = if val_text.is_empty() {
        Vec::new()
    } else {
        model.tokenize_corpus(&val_text)
    };
    let val_chars = val_text.chars().count();
    let mut rng = SmallRng::seed_from_u64(42);
    // RNG を消費して整合させる（任意）: 各 step で batch_size 回 random_range を呼んでいたため
    let max_offset = token_ids.len() - cfg.max_len;
    for _ in 0..(checkpoint_step * cfg.batch_size) {
        let _ = rng.random_range(0..=max_offset);
    }
    run_training_loop(
        &mut model,
        &mut opt,
        &mut rng,
        &token_ids,
        &val_ids,
        val_chars,
        cfg,
        checkpoint_step + 1,
    );
    let inference_path = format!("{}/inference.bin", cfg.checkpoint_dir());
    model.save_inference_checkpoint(&inference_path).unwrap();
    infer(&mut model, &cfg.prompts);
}

fn infer(model: &mut LanguageModel, prompts: &[&str]) {
    let max_new_token = 100;
    let top_k = 5;
    let top_p = 0.9;
    let temperature = 1.0;
    let repetition_penalty = 1.2;
    for prompt in prompts {
        println!("\n=== prompt: {:?} ===", prompt);

        // top-k: KV cache 版を使用 (no-cache 版より高速、 出力サンプリング分布は同じ)
        let t_topk = std::time::Instant::now();
        let topk_text = model.generate_top_k_with_cache(
            prompt,
            max_new_token,
            top_k,
            temperature,
            repetition_penalty,
        );
        let dt_topk = t_topk.elapsed();
        println!(
            "\n--- top-k (k={top_k}, t={temperature}, rep={repetition_penalty}, kv-cache) [{:.2}s] ---",
            dt_topk.as_secs_f32()
        );
        println!("\n{}", topk_text);

        // top-p: KV cache 版
        let t_topp = std::time::Instant::now();
        let topp_text = model.generate_top_p_with_cache(
            prompt,
            max_new_token,
            top_p,
            temperature,
            repetition_penalty,
        );
        let dt_topp = t_topp.elapsed();
        println!(
            "\n--- top-p (p={top_p}, t={temperature}, rep={repetition_penalty}, kv-cache) [{:.2}s] ---",
            dt_topp.as_secs_f32()
        );
        println!("\n{}", topp_text);
    }
}

/// KV cache あり/なしの推論速度を比較するベンチマーク。 学習完了後に呼んで効果を確認するために。
#[allow(dead_code)]
fn bench_kv_cache(model: &mut LanguageModel, prompts: &[&str]) {
    let max_new_token = 100;
    let top_k = 5;
    let temperature = 1.0;
    let repetition_penalty = 1.2;
    let mut total_no_cache = 0.0f32;
    let mut total_with_cache = 0.0f32;
    for prompt in prompts {
        let t1 = std::time::Instant::now();
        let _ = model.generate_top_k(
            prompt,
            max_new_token,
            top_k,
            temperature,
            repetition_penalty,
        );
        let dt_no_cache = t1.elapsed().as_secs_f32();

        let t2 = std::time::Instant::now();
        let _ = model.generate_top_k_with_cache(
            prompt,
            max_new_token,
            top_k,
            temperature,
            repetition_penalty,
        );
        let dt_with_cache = t2.elapsed().as_secs_f32();
        let speedup = dt_no_cache / dt_with_cache.max(1e-6);
        println!(
            "prompt={:?}: no-cache={:.2}s, with-cache={:.2}s, speedup={:.2}x",
            prompt, dt_no_cache, dt_with_cache, speedup
        );
        total_no_cache += dt_no_cache;
        total_with_cache += dt_with_cache;
    }
    println!(
        "\nTOTAL: no-cache={:.2}s, with-cache={:.2}s, speedup={:.2}x",
        total_no_cache,
        total_with_cache,
        total_no_cache / total_with_cache.max(1e-6)
    );
}

/// Phase 6 tokenizer の妥当性チェック。 `cfg.tokenizer_cache_path()` の cache を経由する。
/// 1 回目: 学習 + save。 2 回目: load (ms オーダー)。
/// 学習後に圧縮率と merge 例をレポートする。
#[allow(dead_code)]
fn bench_tokenizer_with_cache(cfg: &Config) {
    let text = fs::read_to_string(cfg.corpus_path).expect("failed to read corpus");
    let char_count = text.chars().count();
    println!(
        "# bench_tokenizer_with_cache: corpus={}, kind={:?}, vocab={}, sample={:?}",
        cfg.corpus_path, cfg.tokenizer_kind, cfg.vocab_size, cfg.merge_sample_chars
    );
    println!("# corpus chars = {char_count} ({} bytes)", text.len());

    let tokenizer = build_or_load_tokenizer(cfg, &text);

    let t1 = Instant::now();
    let ids = tokenizer.encode_long(&text);
    let encode_secs = t1.elapsed().as_secs_f32();
    let token_count = ids.len();
    let chars_per_token = char_count as f32 / token_count.max(1) as f32;
    println!(
        "# corpus encoded in {encode_secs:.1}s: {token_count} tokens ({:.2} chars/token)",
        chars_per_token
    );
    println!(
        "# effective context @ max_len={}: {:.0} chars (vs char tokenizer = {} chars)",
        cfg.max_len,
        chars_per_token * cfg.max_len as f32,
        cfg.max_len
    );

    let decoded = tokenizer.decode(&ids);
    let chars_match = decoded.chars().count();
    if decoded == text {
        println!("# decode roundtrip: OK (lossless, {chars_match} chars)");
    } else {
        let prefix: String = decoded.chars().take(80).collect();
        println!(
            "# decode roundtrip: MISMATCH (orig={char_count} chars, decoded={chars_match} chars). decoded prefix: {prefix:?}"
        );
    }

    // CharBpe の場合のみ merge 例を表示
    if cfg.tokenizer_kind == TokenizerKind::CharBpe {
        // tokenizer は Box<dyn Tokenizer> なので concrete メソッドは呼べない。
        // merge 例はキャッシュファイルに保存された情報からも辿れるので、 別途確認すれば良い。
        println!("\n# (top merges: see {} or rerun bench with concrete CharBpeTokenizer)", cfg.tokenizer_cache_path());
    }
}
