mod adam_w;
mod bpe_tokenizeer;
mod cross_entropy_loss;
mod embedding;
mod feed_forward_network;
mod language_model;
mod layer_normalization;
mod multi_head_attention;
mod output_head;
mod sinusoidal_pe;
mod transformer;
mod transformer_block;
mod utility;

mod checkpoint;
mod lr_scheduler;

use std::fs;

use rand::{SeedableRng, rngs::SmallRng, seq::IndexedRandom};

use crate::{
    adam_w::AdamW, feed_forward_network::FeedForwardNetwork, language_model::LanguageModel,
    layer_normalization::LayerNormalization, lr_scheduler::LrScheduler,
    multi_head_attention::MultiHeadAttention,
};

fn load_corpus(path: &str) -> Vec<String> {
    fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("corpus file '{}' not found: {}", path, e))
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty()) // 空行スキップ
        .filter(|l| !l.starts_with('#')) // コメント行スキップ
        .map(str::to_string)
        .collect()
}

struct Config {
    run_name: &'static str,
    d_model: usize,
    n_heads: usize,
    d_ff: usize,
    n_layers: usize,
    max_len: usize,
    vocab_size: usize,
    lr_max: f32,
    lr_min: f32,
    warmup_steps: usize,
    end_step: usize,
    save_every: usize,
    log_every: usize,
    batch_size: usize,
}

impl Config {
    fn tiny_shakespeare() -> Self {
        Self {
            run_name: "with_lr_sched",
            d_model: 128,
            n_heads: 4,
            d_ff: 512,
            n_layers: 4,
            max_len: 64,
            vocab_size: 4000,
            lr_max: 3e-4,
            lr_min: 1e-6,
            warmup_steps: 200,
            end_step: 10000,
            save_every: 500,
            log_every: 20,
            batch_size: 64,
        }
    }

    fn checkpoint_dir(&self) -> String {
        format!("checkpoints/{}", self.run_name)
    }
}
fn main() {
    let corpus_strings = load_corpus("corpus/train.txt");
    let corpus: Vec<&str> = corpus_strings.iter().map(String::as_str).collect();
    let cfg = Config::tiny_shakespeare();
    training_and_inference(&corpus, &cfg);
    // training_from_checkpoint(&corpus, &cfg, "checkpoints/step_002500.bin");
    // inference_from_checkpoint("checkpoints/step_009000.bin");
}

#[allow(dead_code)]
fn inference_from_checkpoint(path: &str) {
    let mut model = LanguageModel::load_inference_checkpoint(path).unwrap();
    // let prompt = "To be or not to be";
    let prompt = "I have seen";
    infer(&mut model, &prompt);
}

fn run_training_loop(
    model: &mut LanguageModel,
    opt: &mut AdamW,
    rng: &mut SmallRng,
    corpus: &[&str],
    cfg: &Config,
    start_step: usize,
) {
    let pad_id = model.tokenizer.pad_id();
    let mut ema_loss: Option<f32> = None;
    let mut window_min = f32::INFINITY;
    let mut window_max = f32::NEG_INFINITY;
    let lr_scheduler = LrScheduler::new(cfg.lr_max, cfg.lr_min, cfg.warmup_steps, cfg.end_step);

    let ckpt_dir = cfg.checkpoint_dir();
    fs::create_dir_all(&ckpt_dir).unwrap();

    println!("# run_name={}", cfg.run_name);
    println!("# d_model={}, n_heads={}, d_ff={}, n_layers={}, max_len={}, vocab_size={}", cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.n_layers, cfg.max_len, cfg.vocab_size);
    println!("# lr_max={}, lr_min={}, warmup_steps={}, end_step={}, batch_size={}, log_every={}, save_every={}, start_step={}", cfg.lr_max, cfg.lr_min, cfg.warmup_steps, cfg.end_step, cfg.batch_size, cfg.log_every, cfg.save_every, start_step);
    println!("step,loss,ema,min,max,lr");

    for step in start_step..=cfg.end_step {
        let lr = lr_scheduler.get_lr(step);
        opt.set_lr(lr);
        let batch: Vec<&&str> = corpus.sample(rng, cfg.batch_size).collect();
        let mut total_loss = 0.0f32;
        let mut valid_cout = 0;
        for &&text in &batch {
            let ids = model.tokenizer.encode_simple(text);
            if ids.len() < 2 {
                continue;
            }
            total_loss += model.forward_backward(&ids, pad_id);
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
            println!(
                "{},{:.6},{:.6},{:.6},{:.6},{:.3e}",
                step,
                avg_loss,
                ema_loss.unwrap(),
                window_min,
                window_max,
                lr,
            );
            window_min = f32::INFINITY;
            window_max = f32::NEG_INFINITY;
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
fn training_and_inference(corpus: &[&str], cfg: &Config) {
    let mut model = LanguageModel::new(
        corpus,
        cfg.vocab_size,
        cfg.d_model,
        cfg.n_heads,
        cfg.d_ff,
        cfg.n_layers,
        cfg.max_len,
    );
    let mut opt = AdamW::new(cfg.lr_max);
    let mut rng = SmallRng::seed_from_u64(42);
    run_training_loop(&mut model, &mut opt, &mut rng, corpus, cfg, 1);
    let inference_path = format!("{}/inference.bin", cfg.checkpoint_dir());
    model.save_inference_checkpoint(&inference_path).unwrap();
    let prompt = "To be or not to be";
    infer(&mut model, &prompt);
}

#[allow(dead_code)]
fn training_from_checkpoint(corpus: &[&str], cfg: &Config, path: &str) {
    let (mut model, mut opt, checkpoint_step) =
        LanguageModel::load_training_checkpoint(path).unwrap();
    let mut rng = SmallRng::seed_from_u64(42);
    // RNG を消費して整合させる（任意）
    for _ in 0..checkpoint_step {
        let _: Vec<&&str> = corpus.sample(&mut rng, cfg.batch_size).collect();
    }
    run_training_loop(
        &mut model,
        &mut opt,
        &mut rng,
        corpus,
        cfg,
        checkpoint_step + 1,
    );
    let inference_path = format!("{}/inference.bin", cfg.checkpoint_dir());
    model.save_inference_checkpoint(&inference_path).unwrap();
    let prompt = "To be or not to be";
    infer(&mut model, &prompt);
}

fn infer(model: &mut LanguageModel, prompt: &str) {
    println!("\n--- prompt: {:?} ---", prompt);
    println!("greedy : {}", model.generate(prompt, 100));
    println!("top-k  : {}", model.generate_top_k(prompt, 100, 5, 0.8));
}
