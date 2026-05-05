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
    let d_ff = 32;
    let seq_len = 7;

    let mut ffn = FeedForwardNetwork::new(d_model, d_ff);
    let mut opt = AdamW::new(1e-2);

    // ダミーの hidden: (seq_len, d_model)
    let mut rng = rand::rng();
    let x: Vec<Vec<f32>> = (0..seq_len)
        .map(|_| {
            (0..d_model)
                .map(|_| rng.random_range(-1.0..1.0f32))
                .collect()
        })
        .collect();

    let dl_dz2: Vec<Vec<f32>> = (0..seq_len)
        .map(|_| {
            (0..d_model)
                .map(|_| rng.random_range(-1.0..1.0f32))
                .collect()
        })
        .collect();

    println!("=== FFN backward ===");
    for step in 1..=20 {
        let z2 = ffn.forward(&x);

        // MSE loss の勾配
        let n = (seq_len * d_model) as f32;
        let dl_dz2: Vec<Vec<f32>> = z2.iter()
            .map(|row| row.iter().map(|&zi| 2.0 * zi / n).collect())
            .collect();
    
        let loss: f32 = z2.iter()
            .flat_map(|r| r.iter())
            .map(|v| v.powi(2))
            .sum::<f32>() / n;
    
        let dl_dx = ffn.backward(&dl_dz2);
        ffn.apply_gradients(&mut opt);
    
        if step % 5 == 0 {
            println!("step {:2}  loss: {:.6}  grad_w1_norm: {:.4}",
                step, loss, ffn.grad_w1_norm());
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
