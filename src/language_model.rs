use crate::{
    adam_w::AdamW, cross_entropy_loss::CrossEntropyLoss, embedding::Embedding,
    multi_head_attention::causal_mask, output_head::OutputHead, transformer::Transformer,
};

pub struct LanguageModel {
    embedding: Embedding,
    transformer: Transformer,
    output_head: OutputHead,
}

impl LanguageModel {
    pub fn new(
        vocab_size: usize,
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        n_layers: usize,
        pad_id: Option<usize>,
    ) -> Self {
        Self {
            embedding: Embedding::new(vocab_size, d_model, pad_id),
            transformer: Transformer::new(n_layers, d_model, n_heads, d_ff),
            output_head: OutputHead::new(d_model, vocab_size),
        }
    }

    fn forward(&mut self, token_ids: &[usize]) -> Vec<Vec<f32>> {
        let seq = token_ids.len();
        let mask = causal_mask(seq);
        let x = self.embedding.forward(token_ids);
        let h = self.transformer.forward(&x, Some(&mask));
        self.output_head.forward(&h)
    }

    pub fn train_step(&mut self, token_ids: &[usize], opt: &mut AdamW, pad_id: usize) -> f32 {
        let seq = token_ids.len();
        assert!(seq >= 2, "seq_len must be >= 2");

        let mask = causal_mask(seq);

        let x = self.embedding.forward(token_ids);
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
        let dl_dx = self.transformer.backward(&dl_dh_full);
        self.embedding.backward(&dl_dx);

        self.output_head.apply_gradients(opt, "head");
        self.transformer.apply_gradients(opt, "transformer");
        self.embedding.apply_gradients(opt, "embedding");

        loss
    }

    pub fn generate(&mut self, prompt: &[usize], max_new_token: usize) -> Vec<usize> {
        let mut ids = prompt.to_vec();
        for _ in 0..max_new_token {
            let logits = self.forward(&ids);
            let last_logits = &logits[ids.len() - 1];
            let next = OutputHead::greedy(last_logits);
            ids.push(next);
        }

        ids[prompt.len()..].to_vec()
    }

    pub fn generate_top_k(
        &mut self,
        prompt: &[usize],
        max_new_token: usize,
        k: usize,
        temprature: f32,
    ) -> Vec<usize> {
        let mut ids = prompt.to_vec();
        for _ in 0..max_new_token {
            let logits = self.forward(&ids);
            let last_logits = &logits[ids.len() - 1];
            let next = OutputHead::top_k_sample(last_logits, k, temprature);
            ids.push(next);
        }

        ids[prompt.len()..].to_vec()
    }
}

fn pad_grad(mut dl: Vec<Vec<f32>>, seq: usize) -> Vec<Vec<f32>> {
    let d_model = dl[0].len();
    while dl.len() < seq {
        dl.push(vec![0.0; d_model]);
    }
    dl
}
