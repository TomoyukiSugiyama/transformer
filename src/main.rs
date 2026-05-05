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
    let vocab_size = 8;
    let d_model    = 8;
    let seq_len    = 7;  // make_lm_pair後の長さ

    let mut head = OutputHead::new(d_model, vocab_size);
    let mut opt  = AdamW::new(1e-2);

    // ダミーの hidden: (seq_len, d_model)
    let mut rng = rand::rng();
    let hidden: Vec<Vec<f32>> = (0..seq_len)
        .map(|_| (0..d_model)
            .map(|_| rng.random_range(-1.0..1.0f32))
            .collect())
        .collect();
    let targets    = vec![4usize; seq_len];
    let target_mask = vec![1u8; seq_len];

    println!("=== OutputHead backward（シーケンス対応版）===");
    for step in 1..=20 {
        // 1. Forward: (seq_len, d_model) → (seq_len, vocab_size)
        let logits: Vec<Vec<f32>> = head.forward(&hidden);

        // 2. Loss
        let (loss, dl_dlogits) = CrossEntropyLoss::forward_sequence(
            &logits, &targets, &target_mask
        );

        // 3. Backward: dL/dW を内部に保存
        let _dl_dhidden = head.backward(&dl_dlogits);

        // 4. パラメータ更新
        head.apply_gradients(&mut opt);

        if step % 5 == 0 {
            let avg_prob: f32 = logits.iter()
                .map(|l| OutputHead::softmax(l)[4])
                .sum::<f32>() / seq_len as f32;
            println!("step {:2}  loss: {:.4}  avg_prob[target]: {:.4}",
                step, loss, avg_prob);
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

fn _predict() {
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
    let logits = heads.logits_last(&hidden[last_real_pos]);
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
        let logits = heads.logits_last(&hidden[last_pos]);
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
