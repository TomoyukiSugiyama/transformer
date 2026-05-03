mod embedding;
mod sinusoidal_pe;
mod tokenizer;
use crate::{embedding::Embedding, sinusoidal_pe::SinusoidalPE, tokenizer::*};

fn main() {
    let corpus = vec!["Hello, world!", "Attention Is All You Need"];

    let tokenizer = Tokenizer::build(&corpus);
    println!("Corpus: {:?}", corpus);
    println!("Vocab Size: {}", tokenizer.vocab_size());

    let mut token_ids = tokenizer.encode("unknown word");
    println!("Token Ids (Unknown Word): {:?}", token_ids);

    token_ids = tokenizer.encode("Hello, world!Attention Is All You Need");
    println!(
        "Token Ids(Hello, world!Attention Is All You Need): {:?}",
        token_ids
    );

    let d_model = 4;
    let embedding = Embedding::new(tokenizer.vocab_size(), d_model);

    let token_enb = embedding.forward(&token_ids);
    println!(
        "Embedding matrix (seq_len={}, d_model={}):",
        token_ids.len(),
        d_model
    );
    for (i, vec) in token_enb.iter().enumerate() {
        println!("[{i}] {:?}", vec);
    }

    let sin_pe = SinusoidalPE::new(512, d_model);
    let output = sin_pe.forward(&token_enb);
    println!("Sinusoidal Position Encording:");
    for (i, o) in output.iter().enumerate() {
        println!("[{i}] {:?}", o);
    }
}
