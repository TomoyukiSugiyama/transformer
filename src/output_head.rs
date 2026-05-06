use rand::RngExt;
use rand::rng;

use crate::adam_w::AdamW;
use crate::checkpoint::Checkpointable;
use crate::checkpoint::WeightMap;

pub struct OutputHead {
    w: Vec<Vec<f32>>,
    grad_w: Vec<Vec<f32>>,
    cache_hidden: Vec<Vec<f32>>,
    vocab_size: usize,
    d_model: usize,
}

impl OutputHead {
    pub fn new(d_model: usize, vocab_size: usize) -> Self {
        let mut rng = rng();
        let scale = (1.0 / d_model as f32).sqrt();
        let w = (0..d_model)
            .map(|_| {
                (0..vocab_size)
                    .map(|_| rng.random_range(-scale..scale))
                    .collect()
            })
            .collect();
        Self {
            w,
            grad_w: vec![vec![0.0; vocab_size]; d_model],
            cache_hidden: vec![],
            vocab_size,
            d_model,
        }
    }

    pub fn logits_last(&self, last_hidden: &[f32]) -> Vec<f32> {
        (0..self.vocab_size)
            .map(|j| {
                last_hidden
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| x * self.w[i][j])
                    .sum::<f32>()
            })
            .collect()
    }

    pub fn softmax(logits: &[f32]) -> Vec<f32> {
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        exps.iter().map(|e| e / sum).collect()
    }

    pub fn greedy(probs: &[f32]) -> usize {
        probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    }

    pub fn top_k_sample(logits: &[f32], k: usize, temperature: f32) -> usize {
        let mut indexed: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed.truncate(k);

        let top_logits: Vec<f32> = indexed.iter().map(|(_, l)| l / temperature).collect();
        let probes = Self::softmax(&top_logits);
        let mut rng = rng();
        let r: f32 = rng.random_range(0.0..1.0);
        let mut custom = 0.0;

        for (idx, &p) in probes.iter().enumerate() {
            custom += p;
            if r < custom {
                return indexed[idx].0;
            }
        }
        indexed[0].0
    }

    /// (seq_len, d_model) → (seq_len, vocab_size)
    pub fn forward(&mut self, hidden: &[Vec<f32>]) -> Vec<Vec<f32>> {
        self.cache_hidden = hidden.to_vec().clone();
        hidden
            .iter()
            .map(|row: &Vec<f32>| self.logits_last(row))
            .collect()
    }

    /// dL/d_logits (seq_len, vocab_size) → dL/d_hidden (seq_len, d_model)
    /// grad_w を内部に保存する（apply_gradients で使用）
    pub fn backward(&mut self, dl_dlogits: &[Vec<f32>]) -> Vec<Vec<f32>> {
        self.grad_w = vec![vec![0.0; self.vocab_size]; self.d_model];

        let seq_len = dl_dlogits.len();
        let mut dl_dhidden = vec![vec![0.0f32; self.d_model]; seq_len];

        for t in 0..seq_len {
            // dL/dW += cache_hidden[t]^T ⊗ dl_dlogits[t]
            for i in 0..self.d_model {
                for j in 0..self.vocab_size {
                    self.grad_w[i][j] += self.cache_hidden[t][i] * dl_dlogits[t][j];
                }
            }
            // dL/d_hidden[t] = W × dl_dlogits[t]
            for i in 0..self.d_model {
                dl_dhidden[t][i] = (0..self.vocab_size)
                    .map(|j| self.w[i][j] * dl_dlogits[t][j])
                    .sum();
            }
        }

        dl_dhidden
    }

    pub fn apply_gradients(&mut self, opt: &mut AdamW, prefix: &str) {
        opt.step_matrix(&format!("{prefix}.w"), &mut self.w, &self.grad_w);
    }
}

impl Checkpointable for OutputHead {
    fn to_weight_map(&self) -> WeightMap {
        let mut map = WeightMap::new();
        map.insert_scalar("vocab_size", self.vocab_size as u64);
        map.insert_scalar("d_model", self.d_model as u64);
        map.insert_matrix("w", self.w.clone());
        map
    }
}
