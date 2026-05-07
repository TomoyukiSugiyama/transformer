mod bpe_tokenizeer;
mod embedding;
mod sinusoidal_pe;

mod multi_head_attention;
mod utility;

mod feed_forward_network;
mod layer_normalization;

mod transformer;
mod transformer_block;

mod output_head;

mod adam_w;
mod cross_entropy_loss;

mod language_model;

mod checkpoint;

use std::fs;

use crate::{
    adam_w::AdamW, feed_forward_network::FeedForwardNetwork, language_model::LanguageModel,
    layer_normalization::LayerNormalization, multi_head_attention::MultiHeadAttention,
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

fn main() {
    let corpus_strings = load_corpus("corpus/train.txt");
    let corpus: Vec<&str> = corpus_strings.iter().map(String::as_str).collect();
    training_inference(&corpus);
}

fn training_inference(corpus: &[&str]) {
    let d_model = 64;
    let n_heads = 2;
    let d_ff = 128;
    let n_layers = 2;
    let max_len = 32;
    let vocab_size = 500;

    let mut model = LanguageModel::new(
        corpus, vocab_size, d_model, n_heads, d_ff, n_layers, max_len,
    );
    let mut opt = AdamW::new(1e-4);

    println!("vocab_size: {}", model.tokenizer.vocab_size());
    for text in corpus {
        let ids = model.tokenizer.encode_simple(text);
        println!("train text: {:?} ids: {:?}", text, ids)
    }
    println!("=== 学習 ===");
    let save_every = 50;
    let end_step = 100;
    for step in 1..=end_step {
        let mut total_loss = 0.0f32;
        for text in corpus {
            let ids = model.tokenizer.encode_simple(text);
            total_loss += model.train_step(&ids, &mut opt, 0usize);
        }

        if step % save_every == 0 {
            let path = format!("checkpoints/step_{step:06}.bin");
            model
                .save_training_checkpoint(&path, &opt, end_step)
                .unwrap();
            model
                .save_training_checkpoint("checkpoints/latest.bin", &opt, end_step)
                .unwrap();
            println!("saved: {path}");
            println!(
                "step {:3}  loss: {:.6}",
                step,
                total_loss / corpus.len() as f32
            );
        }
    }

    model
        .save_inference_checkpoint("checkpoints/inference.bin")
        .unwrap();

    println!("\n=== 推論 (greedy) ===");
    let prompt = "the cat";
    println!("prompt: \"{}\"", prompt);
    println!("generated: \"{}\"", model.generate(prompt, 10));

    println!("\n=== 推論 (top-k) ===");
    println!(
        "generated: \"{}\"",
        model.generate_top_k(prompt, 10, 3, 0.8)
    );

    println!("\n=== 推論 (loaded) ===");
    let mut loaded = LanguageModel::load_inference_checkpoint("checkpoints/inference.bin").unwrap();
    println!("generated: \"{}\"", loaded.generate(prompt, 10));

    println!("=== チェックポイントから再学習 ===");
    let (mut l_model, mut l_opt, l_end_step) =
        LanguageModel::load_training_checkpoint("checkpoints/latest.bin").unwrap();
    assert!(end_step == l_end_step);
    let start_step = l_end_step + 1;

    for step in start_step..=end_step + 100 {
        let mut total_loss = 0.0f32;
        for text in corpus {
            let ids = l_model.tokenizer.encode_simple(text);
            total_loss += l_model.train_step(&ids, &mut l_opt, 0usize);
        }

        if step % save_every == 0 {
            let path = format!("checkpoints/step_{step:06}.bin");
            model
                .save_training_checkpoint(&path, &opt, end_step)
                .unwrap();
            model
                .save_training_checkpoint("checkpoints/latest.bin", &opt, end_step)
                .unwrap();
            println!("saved: {path}");
            println!(
                "step {:3}  loss: {:.6}",
                step,
                total_loss / corpus.len() as f32
            );
        }
    }

    println!("\n=== 推論 (greedy) ===");
    let prompt = "the cat";
    println!("prompt: \"{}\"", prompt);
    println!("generated: \"{}\"", l_model.generate(prompt, 10));

    println!("\n=== 推論 (top-k) ===");
    println!(
        "generated: \"{}\"",
        l_model.generate_top_k(prompt, 10, 3, 0.8)
    );
}
