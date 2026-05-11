mod adam_w;
mod bpe_tokenizer;
mod char_tokenizer;
mod cross_entropy_loss;
mod dropout;
mod embedding;
mod eval;
mod feed_forward;
mod feed_forward_network;
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
    lr_scheduler::LrScheduler, multi_head_attention::MultiHeadAttention,
    normalization::NormalizationKind, positional_encoding::PositionalEncodingKind,
    tokenizer::TokenizerKind,
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
        }
    }

    fn checkpoint_dir(&self) -> String {
        format!("checkpoints/{}", self.run_name)
    }
}
fn main() {
    // Phase 4a: 夏目漱石「こころ」 単独 (青空文庫, Char tokenizer, d_model=384, dropout=0.2)
    // let cfg = Config::aozora_kokoro();
    // training_and_inference(&cfg);
    // inference_from_checkpoint(&cfg, "checkpoints/phase4_aozora_kokoro_d384_n6_char/step_000200.bin");

    // 他の Config に切替えるには下記を有効化:
    //
    // Phase 4b: 漱石主要長編 7 作品 (~1.21M char, Char tokenizer)
    let cfg = Config::aozora_soseki_works();
    // training_and_inference(&cfg);
    inference_from_checkpoint(
        &cfg,
        "checkpoints/phase4b_aozora_soseki_works_d384_n6_char_rms_swiglu_rope/best.bin",
    );

    // Phase 4 旧 (BPE) checkpoint で推論:
    // let cfg = Config::aozora_kokoro();   // 一時的に tokenizer_kind を Bpe に変更が必要
    // inference_from_checkpoint(&cfg, "checkpoints/phase4_aozora_kokoro_d384_n6_bpe/step_000250.bin");
    //
    // Phase 3: nanoGPT 相当 char-level Tiny Shakespeare
    // let cfg = Config::nano_gpt_equivalent();
    // inference_from_checkpoint(&cfg, "checkpoints/phase3_nanogpt_equiv_d384_n6_char/step_001000.bin");
    //
    // Phase 2: BPE Tiny Shakespeare
    // let cfg = Config::tiny_shakespeare();
    // inference_from_checkpoint(&cfg, "checkpoints/phase2_d256_ff1024_max128_with_accelerate/step_002500.bin");
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
    cfg: &Config,
    start_step: usize,
) {
    let pad_id = model.pad_id();
    let mut ema_loss: Option<f32> = None;
    let mut window_min = f32::INFINITY;
    let mut window_max = f32::NEG_INFINITY;
    let lr_scheduler = LrScheduler::new(cfg.lr_max, cfg.lr_min, cfg.warmup_steps, cfg.end_step);

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
        "# lr_max={}, lr_min={}, warmup_steps={}, end_step={}, batch_size={}, log_every={}, save_every={}, start_step={}",
        cfg.lr_max,
        cfg.lr_min,
        cfg.warmup_steps,
        cfg.end_step,
        cfg.batch_size,
        cfg.log_every,
        cfg.save_every,
        start_step
    );
    println!(
        "# corpus_tokens={}, chunk_len={}, max_offset={}, val_enabled={}, val_tokens={}, val_every={}, val_n_batches={}",
        token_ids.len(),
        chunk_len,
        max_offset,
        val_enabled,
        val_ids.len(),
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
            println!(
                "# val step={} val_loss={:.6} val_ppl={:.4}",
                step, val_loss, val_ppl
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
                    "# best updated: step={} val_loss={:.6} val_ppl={:.4} (prev val_loss={}) -> saved {}",
                    step, val_loss, val_ppl, prev_repr, best_path
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
    let mut model = LanguageModel::new(
        &train_text,
        cfg.tokenizer_kind,
        cfg.normalization_kind,
        cfg.feed_forward_kind,
        cfg.positional_encoding_kind,
        cfg.vocab_size,
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
    let mut opt = AdamW::new_with_wd(cfg.lr_max, cfg.weight_decay);
    opt.set_beta2(cfg.beta2);
    let mut rng = SmallRng::seed_from_u64(42);
    run_training_loop(&mut model, &mut opt, &mut rng, &token_ids, &val_ids, cfg, 1);
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
    let temperature = 1.0;
    let repetition_penalty = 1.2;
    for prompt in prompts {
        println!("\n--- prompt: {:?} ---", prompt);
        println!(
            "\n{}",
            model.generate_top_k(
                prompt,
                max_new_token,
                top_k,
                temperature,
                repetition_penalty
            )
        );
    }
}
