use std::io::Result;

use crate::{
    adam_w::AdamW,
    checkpoint::{Checkpointable, WeightMap},
    cross_entropy_loss::CrossEntropyLoss,
    embedding::Embedding,
    multi_head_attention::causal_mask,
    output_head::OutputHead,
    sinusoidal_pe::SinusoidalPE,
    tokenizer::Tokenizer,
    transformer::Transformer,
};

pub struct LanguageModel {
    pub tokenizer: Tokenizer,
    embedding: Embedding,
    pe: SinusoidalPE,
    transformer: Transformer,
    output_head: OutputHead,
    // ハイパーパラメータ
    d_model: usize,
    n_heads: usize,
    d_ff: usize,
    n_layers: usize,
    max_len: usize,
}

impl LanguageModel {
    pub fn new(
        corpus: &[&str],
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        n_layers: usize,
        max_len: usize,
    ) -> Self {
        let tokenizer = Tokenizer::build(corpus);
        let vocab_size = tokenizer.vocab_size();
        let pad_id = 0usize;
        Self {
            tokenizer,
            embedding: Embedding::new(vocab_size, d_model, Some(pad_id)),
            pe: SinusoidalPE::new(max_len, d_model),
            transformer: Transformer::new(n_layers, d_model, n_heads, d_ff),
            output_head: OutputHead::new(d_model, vocab_size),
            d_model,
            n_heads,
            d_ff,
            n_layers,
            max_len,
        }
    }

    fn forward_ids(&mut self, token_ids: &[usize]) -> Vec<Vec<f32>> {
        let seq = token_ids.len();
        let mask = causal_mask(seq);
        let emb = self.embedding.forward(token_ids);
        let x = self.pe.forward(&emb);
        let h = self.transformer.forward(&x, Some(&mask));
        self.output_head.forward(&h)
    }

    pub fn train_step(&mut self, token_ids: &[usize], opt: &mut AdamW, pad_id: usize) -> f32 {
        let seq = token_ids.len();
        assert!(seq >= 2, "seq_len must be >= 2");

        let mask = causal_mask(seq);

        let emb = self.embedding.forward(token_ids);
        let x = self.pe.forward(&emb);
        let h = self.transformer.forward(&x, Some(&mask));

        let h_shifted = &h[..seq - 1];
        let logits = self.output_head.forward(&h_shifted);

        let targets = &token_ids[1..];

        let mask_ce: Vec<u8> = targets
            .iter()
            .map(|&t| if t == pad_id { 0 } else { 1 })
            .collect();

        let (loss, dl_dlogits) = CrossEntropyLoss::forward_sequence(&logits, targets, &mask_ce);

        let dl_dh_shifted = self.output_head.backward(&dl_dlogits);
        let dl_dh_full = pad_grad(dl_dh_shifted, seq);
        let dl_dh_full = clip_grad_norm(dl_dh_full, 1.0);
        let dl_dx = self.transformer.backward(&dl_dh_full);
        self.embedding.backward(&dl_dx);

        self.output_head.apply_gradients(opt, "head");
        self.transformer.apply_gradients(opt, "transformer");
        self.embedding.apply_gradients(opt, "embedding");

        loss
    }

    pub fn generate(&mut self, prompt_text: &str, max_new_token: usize) -> String {
        let mut ids = self.tokenizer.encode_prompt(prompt_text);
        let eos_id = self.tokenizer.eos_id();
        for _ in 0..max_new_token {
            let logits = self.forward_ids(&ids);
            let last_logits = &logits[ids.len() - 1];
            let next_id = OutputHead::greedy(last_logits);
            if next_id == eos_id {
                break;
            }
            ids.push(next_id);
        }
        let bos_id = self.tokenizer.bos_id();
        let start = ids
            .iter()
            .position(|&id| id == bos_id)
            .map(|p| p + 1)
            .unwrap_or(0);
        self.tokenizer.decord(&ids[start..])
    }

    pub fn generate_top_k(
        &mut self,
        prompt_text: &str,
        max_new_token: usize,
        k: usize,
        temprature: f32,
    ) -> String {
        let mut ids = self.tokenizer.encode_prompt(prompt_text);
        let eos_id = self.tokenizer.eos_id();
        for _ in 0..max_new_token {
            let logits = self.forward_ids(&ids);
            let last_logits = &logits[ids.len() - 1];
            let next_id = OutputHead::top_k_sample(last_logits, k, temprature);
            if next_id == eos_id {
                break;
            }
            ids.push(next_id);
        }
        let bos_id = self.tokenizer.bos_id();
        let start = ids
            .iter()
            .position(|&id| id == bos_id)
            .map(|p| p + 1)
            .unwrap_or(0);
        self.tokenizer.decord(&ids[start..])
    }

    pub fn save_inference_checkpoint(&self, path: &str) -> Result<()> {
        let mut map = WeightMap::new();
        map.insert_scalar("meta.d_model", self.d_model as u64);
        map.insert_scalar("meta.n_heads", self.n_heads as u64);
        map.insert_scalar("meta.d_ff", self.d_ff as u64);
        map.insert_scalar("meta.n_layers", self.n_layers as u64);
        map.insert_scalar("meta.max_len", self.max_len as u64);
        map.merge("tokenizer", self.tokenizer.to_weight_map());
        map.merge("embedding", self.embedding.to_weight_map());
        map.merge("transformer", self.transformer.to_weight_map());
        map.merge("output_head", self.output_head.to_weight_map());
        map.save(path)
    }

    pub fn load_inference_checkpoint(path: &str) -> Result<Self> {
        let map = WeightMap::load(path)?;
        let d_model = map.get_scalar("meta.d_model")? as usize;
        let n_heads = map.get_scalar("meta.n_heads")? as usize;
        let d_ff = map.get_scalar("meta.d_ff")? as usize;
        let n_layers = map.get_scalar("meta.n_layers")? as usize;
        let max_len = map.get_scalar("meta.max_len")? as usize;
        let mut tokenizer = Tokenizer::build(&[""]);
        tokenizer.from_weight_map(&map.scoped("tokenizer"))?;
        let vocab_size = tokenizer.vocab_size();

        let mut model = Self {
            tokenizer,
            embedding: Embedding::new(vocab_size, d_model, Some(0)),
            pe: SinusoidalPE::new(max_len, d_model),
            transformer: Transformer::new(n_layers, d_model, n_heads, d_ff),
            output_head: OutputHead::new(d_model, vocab_size),
            d_model,
            n_heads,
            d_ff,
            n_layers,
            max_len,
        };

        model.embedding.from_weight_map(&map.scoped("embedding"))?;
        model
            .transformer
            .from_weight_map(&map.scoped("transformer"))?;
        model
            .output_head
            .from_weight_map(&map.scoped("output_head"))?;

        Ok(model)
    }
}

fn pad_grad(mut dl: Vec<Vec<f32>>, seq: usize) -> Vec<Vec<f32>> {
    let d_model = dl[0].len();
    while dl.len() < seq {
        dl.push(vec![0.0; d_model]);
    }
    dl
}

fn clip_grad_norm(mut grads: Vec<Vec<f32>>, max_norm: f32) -> Vec<Vec<f32>> {
    let norm: f32 = grads
        .iter()
        .flatten()
        .map(|v| v.powi(2))
        .sum::<f32>()
        .sqrt();
    if norm > max_norm {
        let scale = max_norm / norm;
        grads.iter_mut().flatten().for_each(|v| *v *= scale);
    }
    grads
}
