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
    let d_model = 8;
    let n_heads = 2;
    let d_ff = 32;
    let n_layers = 2;
    let seq_len = 7;

    let mut transformer = Transformer::new(n_layers, d_model, n_heads, d_ff);
    let mut opt = AdamW::new(1e-3);

    // ダミー
    let mut rng = rand::rng();
    let x: Vec<Vec<f32>> = (0..seq_len)
        .map(|_| {
            (0..d_model)
                .map(|_| rng.random_range(-1.0..1.0f32))
                .collect()
        })
        .collect();
    let target: Vec<Vec<f32>> = vec![vec![0.0f32; d_model]; seq_len];
    let n = (d_model * seq_len) as f32;

    println!("=== Transcormer backward ===");
    for step in 1..=50 {
        let out = transformer.forward(&x, None);

        // MSE loss の勾配
        let loss: f32 = out
            .iter()
            .zip(target.iter())
            .flat_map(|(yr, tr)| yr.iter().zip(tr.iter()).map(|(&yi, &ti)| (yi - ti).powi(2)))
            .sum::<f32>()
            / n;
        let dl_dout: Vec<Vec<f32>> = out
            .iter()
            .zip(target.iter())
            .map(|(yr, tr)| {
                yr.iter()
                    .zip(tr.iter())
                    .map(|(&yi, &ti)| 2.0 * (yi - ti) / n)
                    .collect()
            })
            .collect();

        transformer.backward(&dl_dout);
        transformer.apply_gradients(&mut opt, "transformer");

        if step % 5 == 0 {
            println!("step {:2}  loss: {:.6}", step, loss);
        }
    }
}

