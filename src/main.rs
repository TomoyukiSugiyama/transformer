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

mod cross_entropy_loss;

use crate::{
    cross_entropy_loss::CrossEntropyLoss, embedding::Embedding, feed_forward_network::FeedForwardNetwork, layer_normalization::LayerNormalization, multi_head_attention::{MultiHeadAttention, causal_mask}, output_head::OutputHead, sinusoidal_pe::SinusoidalPE, tokenizer::*, transformer::Transformer
};

fn main() {
    let corpus = vec!["hello world rust transformer"];
    let d_model = 8;
    let d_ff = d_model * 4;
    let n_heads = 2;
    let n_layers = 2;
    let seq_len = 8;
    
    let tokenizer = Tokenizer::build(&corpus);
    let vocab_size = tokenizer.vocab_size();
    let embedding = Embedding::new(vocab_size, d_model);
    let pe = SinusoidalPE::new(512, d_model);
    let transformer = Transformer::new(n_layers, d_model, n_heads, d_ff);
    let head = OutputHead::new(d_model, vocab_size);

    let text = "hello world rust";
    let enc =tokenizer.encode_with_padding(text, seq_len);

    let (input_ids,targets,target_mask) = make_lm_pair(&enc);

    let x = pe.forward(&embedding.forward(&input_ids));
    let mask = causal_mask(seq_len);
    let hidden = transformer.forward(&x, Some(&mask));
    let logits = head.forward(&hidden);

    let (loss,grad) = CrossEntropyLoss::forward_sequence(&logits, &targets, &target_mask);

    println!("Loss {:.4}",loss);
    let theorical = (tokenizer.vocab_size() as f32).ln();
    println!("理論初期Loss: {:.4}",theorical);

    println!("grad shape: ({}, {})",grad.len(),grad[0].len());
    println!("grad[0] norm: {:.6}",grad[0].iter().map(|v| v.powi(2)).sum::<f32>().sqrt());

}

fn make_lm_pair(enc: &Encoding) -> (Vec<usize>, Vec<usize>, Vec<u8>) {
    let ids = &enc.input_ids;
    let mask = &enc.attention_mask;
    let len = ids.len();

    let inputs = ids[..len - 1].to_vec();
    let targets = ids[1..].to_vec();

    let target_mask = mask[1..].to_vec();

    (inputs, targets, target_mask)
}

fn _predict(){
    let d_model = 8;
    let d_ff = d_model * 4;
    let n_heads = 2;
    let n_layers = 2;
    let seq_len = 8;

    let corpus = vec!["Hello, world!", "Attention Is All You Need"];

    // Torkenize
    let tokenizer = Tokenizer::build(&corpus);
    let vocab_size = tokenizer.vocab_size();
    println!("Corpus: {:?}", corpus);
    println!("Vocab Size: {}", vocab_size);

    let mut enc = tokenizer.encode_with_padding("unknown word", seq_len);
    println!("Token Ids (Unknown Word): {:?}", enc.input_ids);

    enc = tokenizer.encode_with_padding("Hello, world!Attention Is All You Need", seq_len);
    println!(
        "Token Ids (Hello, world!Attention Is All You Need): {:?}",
        enc.input_ids
    );

    // Token Embedding
    let embedding = Embedding::new(vocab_size, d_model);
    let token_enb = embedding.forward(&enc.input_ids);
    println!("Token Embedding:");
    println!(
        "Output Shape: [{}, {}]",
        token_enb.len(),
        token_enb[0].len()
    );
    for (i, vec) in token_enb.iter().enumerate() {
        println!("[{i}] {:?}", vec);
    }

    // Positional Encoding
    let sin_pe = SinusoidalPE::new(512, d_model);
    let x = sin_pe.forward(&token_enb);
    println!("Sinusoidal Positional Encoding:");
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
    let last_real_pos = enc
        .attention_mask
        .iter()
        .rposition(|&m| m == 1)
        .unwrap_or(0);
    let logits = heads.logit_last(&hidden[last_real_pos]);
    let probs = OutputHead::softmax(&logits);

    let next_id = OutputHead::greedy(&probs);
    let next_token = tokenizer.id_to_token_str(next_id).unwrap_or("<UNK>");
    println!("Next Token Greedy (id={next_id}): {next_token}");

    // Top-3 predictions
    let mut sorted: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (id, prob) in sorted.iter().take(3) {
        let token = tokenizer.id_to_token_str(*id).unwrap_or("<UNK>");
        println!("[{:10}] id={:3}) {:.4}", token, id, prob);
    }

    // Auto regressive
    println!("Auto regressive:");
    let mut generated_ids = tokenizer.encode_simple("hello");
    let max_gen = 5;
    println!("{:?}", generated_ids);
    for step in 0..max_gen {
        let enc = tokenizer.encode_with_padding_from_ids(&generated_ids, seq_len);
        let x = sin_pe.forward(&embedding.forward(&enc.input_ids));
        let hidden = transformer.forward(&x, Some(&causal_mask(seq_len)));

        let last_pos = generated_ids.len().min(seq_len) - 1;
        let logits = heads.logit_last(&hidden[last_pos]);
        let next_id = OutputHead::top_k_sample(&logits, 5, 0.8);
        let next_token = tokenizer.id_to_token_str(next_id).unwrap_or("<UNK>");
        println!(" step[{step}]: next={next_token} (id={next_id})");

        if next_token == Tokenizer::EOS {
            break;
        }
        generated_ids.push(next_id);
    }
    let generated_text = tokenizer.decord(&generated_ids);
    println!("Generated: {generated_text}")
}
