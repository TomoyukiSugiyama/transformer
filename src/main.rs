mod embedding;
mod sinusoidal_pe;
mod tokenizer;

mod multi_head_attention;
mod utility;

mod layer_normalization;
mod feed_forward_network;

use crate::{
    embedding::Embedding, feed_forward_network::FFN, layer_normalization::{LayerNormalization, add_and_norm}, multi_head_attention::{MultiHeadAttention, causal_mask}, sinusoidal_pe::SinusoidalPE, tokenizer::*
};

fn main() {
    let d_model = 8;
    let d_ff = d_model * 4;
    let n_heads = 2;
    let seq_len = 9;

    let corpus = vec!["Hello, world!", "Attention Is All You Need"];

    let tokenizer = Tokenizer::build(&corpus);
    println!("Corpus: {:?}", corpus);
    println!("Vocab Size: {}", tokenizer.vocab_size());

    let mut token_ids = tokenizer.encode("unknown word");
    println!("Token Ids (Unknown Word): {:?}", token_ids);

    token_ids = tokenizer.encode("Hello, world!Attention Is All You Need");
    println!(
        "Token Ids (Hello, world!Attention Is All You Need): {:?}",
        token_ids
    );

    let embedding = Embedding::new(tokenizer.vocab_size(), d_model);

    let token_enb = embedding.forward(&token_ids);
    println!("Embedding matrix (seq_len={seq_len}, d_model={d_model}):");
    for (i, vec) in token_enb.iter().enumerate() {
        println!("[{i}] {:?}", vec);
    }

    let sin_pe = SinusoidalPE::new(512, d_model);
    let x = sin_pe.forward(&token_enb);
    println!("Sinusoidal Position Encording:");
    for (i, o) in x.iter().enumerate() {
        println!("[{i}] {:?}", o);
    }

    let mha = MultiHeadAttention::new(d_model, n_heads);
    let mask = causal_mask(seq_len);

    let (attn_output, attention_waight) = mha.forward(&x, Some(&mask));

    println!("Attention Output Shape: [{}, {}]", attn_output.len(), attn_output[0].len());

    println!("Attention Output:");
    for (i, o) in attn_output.iter().enumerate() {
        println!("[{i}] {:?}", o);
    }

    println!("Attention weights [head=0]:");
    for (i, row) in attention_waight[0].iter().enumerate() {
        let formatted: Vec<String> = row.iter().map(|w| format!("{:.2}", w)).collect();
        println!("pos[{i}] {}", formatted.join(", "));
    }

    let norm1 = LayerNormalization::new(d_model);
    let x = add_and_norm(&x, &attn_output, &norm1);
    println!("Add and Normalization:");
    for (i, o) in x.iter().enumerate() {
        let n = o.len() as f32;
        let mean = o.iter().sum::<f32>() / n;
        let var = o.iter().map(|v| (v - mean).powi(2)).sum::<f32>();
        println!("[{i},m≒{:.2},v≒{:.2}]{:?}", mean,var,o);
    }

    let ffn = FFN::new(d_model, d_ff);
    let ffn_out = ffn.forward(&x);
    println!("Feed Forward Network Output:");
    for (i, o) in ffn_out.iter().enumerate() {
        println!("[{i}] {:?}", o);
    }

    let norm2 = LayerNormalization::new(d_model);
    let x = add_and_norm(&x, &ffn_out, &norm2);
    println!("Add and Normalization:");
    for (i, o) in x.iter().enumerate() {
        let n = o.len() as f32;
        let mean = o.iter().sum::<f32>() / n;
        let var = o.iter().map(|v| (v - mean).powi(2)).sum::<f32>();
        println!("[{i},m≒{:.2},v≒{:.2}]{:?}", mean,var,o);
    }    

}
