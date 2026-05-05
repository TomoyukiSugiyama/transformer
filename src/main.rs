mod embedding;
mod sinusoidal_pe;
mod tokenizer;

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

use std::vec;

use crate::{
    adam_w::AdamW, feed_forward_network::FeedForwardNetwork, language_model::LanguageModel,
    layer_normalization::LayerNormalization, multi_head_attention::MultiHeadAttention,
};

fn main() {
    let vocab_size = 14; // 予約4 + 通常10
    let d_model = 32;
    let n_heads = 2;
    let d_ff = 64;
    let n_layers = 2;
    let pad_id = 0usize;

    let mut model = LanguageModel::new(vocab_size, d_model, n_heads, d_ff, n_layers, Some(pad_id));
    let mut opt = AdamW::new(1e-4);

    let token_ids: Vec<usize> = vec![4, 5, 6, 7, 4, 5, 6, 7, 4, 5];

    println!("=== LanguageModel 学習 ===");
    for step in 1..=200 {
        let loss = model.train_step(&token_ids, &mut opt, pad_id);
        if step == 1 {
            println!("step   1  loss: {:.6}", loss); // ← 追加
        }
        if step % 20 == 0 {
            println!("step {:3} loss {:.6}", step, loss);
        }
    }

    println!("=== 推論 (greedy) ===");
    let prompt = vec![4usize, 5, 6];
    let generated = model.generate(&prompt, 7);
    println!("prompt:    {:?}", prompt);
    println!("generated: {:?}", generated);
    println!("expected:  [7, 4, 5, 6, 7, 4, 5]");
    println!("=== 推論 (top-k, k=3, temp=0.8) ===");
    let generated = model.generate_top_k(&prompt, 7, 3, 0.8);
    println!("generated: {:?}", generated);

    println!("=== 推論 (別 prompt) ===");
    let prompt2 = vec![7usize];
    let generated2 = model.generate(&prompt2, 5);
    println!("prompt:    {:?}", prompt2);
    println!("generated: {:?}", generated2);
    println!("expected:  [4, 5, 6, 7, 4]");
}
