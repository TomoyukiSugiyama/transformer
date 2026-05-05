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

use std::vec;

use crate::{
    adam_w::AdamW,
    cross_entropy_loss::CrossEntropyLoss,
    embedding::Embedding,
    feed_forward_network::FeedForwardNetwork,
    layer_normalization::LayerNormalization,
    multi_head_attention::{MultiHeadAttention, causal_mask},
    output_head::OutputHead,
    sinusoidal_pe::SinusoidalPE,
    tokenizer::*,
    transformer::Transformer,
};

use rand::RngExt;

fn main() {
    let vocab_size = 16;
    let d_model = 8;
    let seq_len = 7;
    let pad_id = 0usize;

    let mut emb = Embedding::new(vocab_size, d_model, Some(pad_id));
    let mut opt = AdamW::new(1e-2);

    // ダミー
    let mut rng = rand::rng();

    let token_ids: Vec<usize> = vec![0, 3, 1, 5, 0, 2, 4];
    let target = vec![vec![0.0f32; d_model]; seq_len];
    let n = (d_model * seq_len) as f32;

    println!("=== Embedding backward ===");
    for step in 1..=20 {
        let x = emb.forward(&token_ids);

        // MSE loss の勾配
        let loss: f32 = x
            .iter()
            .zip(target.iter())
            .flat_map(|(yr, tr)| yr.iter().zip(tr.iter()).map(|(&yi, &ti)| (yi - ti).powi(2)))
            .sum::<f32>()
            / n;
        let dl_dx: Vec<Vec<f32>> = x
            .iter()
            .zip(target.iter())
            .map(|(yr, tr)| {
                yr.iter()
                    .zip(tr.iter())
                    .map(|(&yi, &ti)| 2.0 * (yi - ti) / n)
                    .collect()
            })
            .collect();

        emb.backward(&dl_dx);
        emb.apply_gradients(1e-3);

        if step % 5 == 0 {
            println!(
                "step {:2}  loss: {:.6}  pad_weight_norm: {:.6}",
                step,
                loss,
                emb.weight_norm(pad_id)
            );
        }
    }
}

fn _make_lm_pair(enc: &Encoding) -> (Vec<usize>, Vec<usize>, Vec<u8>) {
    let ids = &enc.input_ids;
    let mask = &enc.attention_mask;
    let len = ids.len();

    let inputs = ids[..len - 1].to_vec();
    let targets = ids[1..].to_vec();

    let target_mask = mask[1..].to_vec();

    (inputs, targets, target_mask)
}
