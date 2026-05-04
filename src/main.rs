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

use crate::{
    embedding::Embedding,
    feed_forward_network::FeedForwardNetwork,
    layer_normalization::LayerNormalization,
    multi_head_attention::{MultiHeadAttention, causal_mask},
    output_head::OutputHead,
    sinusoidal_pe::SinusoidalPE,
    tokenizer::*,
    transformer::Transformer,
};

fn main() {
    let d_model = 8;
    let d_ff = d_model * 4;
    let n_heads = 2;
    let n_layers = 2;
    let seq_len = 9;

    let corpus = vec!["Hello, world!", "Attention Is All You Need"];

    // Torkenize
    let tokenizer = Tokenizer::build(&corpus);
    let vocab_size = tokenizer.vocab_size();
    println!("Corpus: {:?}", corpus);
    println!("Vocab Size: {}", vocab_size);

    let mut token_ids = tokenizer.encode("unknown word");
    println!("Token Ids (Unknown Word): {:?}", token_ids);

    token_ids = tokenizer.encode("Hello, world!Attention Is All You Need");
    println!(
        "Token Ids (Hello, world!Attention Is All You Need): {:?}",
        token_ids
    );

    // Token Embedding
    let embedding = Embedding::new(vocab_size, d_model);
    let token_enb = embedding.forward(&token_ids);
    println!("Token Embedding:");
    println!(
        "Output Shape: [{}, {}]",
        token_enb.len(),
        token_enb[0].len()
    );
    for (i, vec) in token_enb.iter().enumerate() {
        println!("[{i}] {:?}", vec);
    }

    // Positional Encording
    let sin_pe = SinusoidalPE::new(512, d_model);
    let x = sin_pe.forward(&token_enb);
    println!("Sinusoidal Positional Encording:");
    println!("Output Shape: [{}, {}]", x.len(), x[0].len());
    for (i, o) in x.iter().enumerate() {
        println!("[{i}] {:?}", o);
    }

    // Transformer (N Layer)
    let transformer = Transformer::new(n_layers, d_model, n_heads, d_ff);
    let mask = causal_mask(seq_len);

    println!("Transformer:");
    let hidden = transformer.forward(&x, Some(&mask));
    println!("Output Shape: [{}, {}]", hidden.len(), hidden[0].len());
    for (i, row) in hidden.iter().enumerate() {
        let formatted: Vec<String> = row.iter().map(|v| format!("{:6.3}", v)).collect();
        println!("  pos[{i}] [{}]", formatted.join(", "))
    }

    // Output Heads
    let heads = OutputHead::new(d_model, vocab_size);
    let last_real_pos = token_ids.len() - 1;
    let logits = heads.logit_last(&hidden[last_real_pos]);
    let probs = OutputHead::softmax(&logits);
    println!("{:?}", probs);

    let next_id = OutputHead::greedy(&probs);
    let next_token = tokenizer.id_to_token_str(next_id).unwrap_or("<UNK>");
    println!("Next Token Greedy (id={next_id}): {next_token}")
}
