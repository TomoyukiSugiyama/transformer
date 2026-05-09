use std::io::Error;
use std::io::ErrorKind;
use std::io::Result;

use rand::RngExt;
use rand::rng;

use crate::adam_w::AdamW;
use crate::checkpoint::Checkpointable;
use crate::checkpoint::WeightMap;
use crate::matrix::Matrix;

pub struct OutputHead {
    w: Matrix,            // (d_model, vocab_size)
    grad_w: Matrix,       // (d_model, vocab_size)
    cache_hidden: Matrix, // (seq_len, d_model)
    vocab_size: usize,
    d_model: usize,
}

impl OutputHead {
    pub fn new(d_model: usize, vocab_size: usize) -> Self {
        let mut rng = rng();
        let scale = (1.0 / d_model as f32).sqrt();
        let mut w = Matrix::zeros(d_model, vocab_size);
        for v in w.data_mut() {
            *v = rng.random_range(-scale..scale);
        }
        Self {
            w,
            grad_w: Matrix::zeros(d_model, vocab_size),
            cache_hidden: Matrix::zeros(0, 0),
            vocab_size,
            d_model,
        }
    }

    #[allow(dead_code)]
    pub fn logits_last(&self, last_hidden: &[f32]) -> Vec<f32> {
        use rayon::prelude::*;
        let cols = self.w.cols();
        (0..cols)
            .into_par_iter()
            .map(|j| {
                last_hidden
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| x * self.w.get(i, j))
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

    #[allow(dead_code)]
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
        self.cache_hidden = Matrix::from_jagged(hidden);
        // logits = hidden @ W   shape: (seq_len, vocab_size)
        self.cache_hidden.matmul(&self.w).to_jagged()
    }

    /// dL/d_logits (seq_len, vocab_size) → dL/d_hidden (seq_len, d_model)
    /// grad_w を内部に累積する（apply_gradients で使用、バッチ末に zero_grad で初期化）
    pub fn backward(&mut self, dl_dlogits: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let dl_dlogits_m = Matrix::from_jagged(dl_dlogits);

        // grad_w += cache_hidden^T @ dl_dlogits   shape: (d_model, vocab_size)
        let g_w = self.cache_hidden.transpose().matmul(&dl_dlogits_m);
        self.grad_w.add_in_place(&g_w);

        // dl_dhidden = dl_dlogits @ W^T   shape: (seq_len, d_model)
        dl_dlogits_m.matmul(&self.w.transpose()).to_jagged()
    }

    pub fn zero_grad(&mut self) {
        self.grad_w.data_mut().fill(0.0);
    }

    pub fn apply_gradients(&mut self, opt: &mut AdamW, prefix: &str) {
        opt.step_matrix_flat(&format!("{prefix}.w"), &mut self.w, &self.grad_w);
    }
}

impl Checkpointable for OutputHead {
    fn to_weight_map(&self) -> WeightMap {
        let mut map = WeightMap::new();
        map.insert_scalar("vocab_size", self.vocab_size as u64);
        map.insert_scalar("d_model", self.d_model as u64);
        map.insert_matrix("w", self.w.to_jagged());
        map
    }

    fn from_weight_map(&mut self, map: &WeightMap) -> Result<()> {
        let vocab_size = map.get_scalar("vocab_size")? as usize;
        let d_model = map.get_scalar("d_model")? as usize;
        if vocab_size != self.vocab_size || d_model != self.d_model {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "Output head config mismatch",
            ));
        }
        self.w = Matrix::from_jagged(map.get_matrix("w")?);
        Ok(())
    }
}
