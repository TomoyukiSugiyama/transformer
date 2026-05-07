use std::io::Error;
use std::io::ErrorKind;
use std::io::Result;

use crate::{
    adam_w::AdamW,
    checkpoint::{Checkpointable, WeightMap},
};

pub struct LayerNormalization {
    gamma: Vec<f32>,
    beta: Vec<f32>,
    eps: f32,

    grad_gamma: Vec<f32>, // [d_model]
    grad_beta: Vec<f32>,  // [d_model]

    cache_x_hat: Vec<Vec<f32>>, // [seq_len, d_model]
    cache_inv_std: Vec<f32>,    // [seq_len] — 1/σ
}

impl LayerNormalization {
    pub fn new(d_model: usize) -> Self {
        Self {
            gamma: vec![1.0; d_model],
            beta: vec![0.0; d_model],
            eps: 1e-6,
            grad_gamma: vec![0.0; d_model],
            grad_beta: vec![0.0; d_model],
            cache_x_hat: Vec::new(),
            cache_inv_std: Vec::new(),
        }
    }

    fn normalization(&mut self, x: &[f32]) -> Vec<f32> {
        let n = x.len() as f32;
        let mean = x.iter().sum::<f32>() / n;
        let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
        let inv_std = 1.0 / (var + self.eps).sqrt();

        let x_hat: Vec<f32> = x.iter().map(|v| (v - mean) * inv_std).collect();

        let y: Vec<f32> = x_hat
            .iter()
            .enumerate()
            .map(|(i, xh)| self.gamma[i] * xh + self.beta[i])
            .collect();

        self.cache_x_hat.push(x_hat);
        self.cache_inv_std.push(inv_std);

        y
    }

    pub fn forward(&mut self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        self.cache_x_hat.clear();
        self.cache_inv_std.clear();

        x.iter().map(|row| self.normalization(row)).collect()
    }

    /// dl_dy: [seq_len, d_model]
    pub fn backward(&mut self, dl_dy: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let d = self.gamma.len() as f32;

        for (dy_row, xh_row) in dl_dy.iter().zip(self.cache_x_hat.iter()) {
            for j in 0..self.gamma.len() {
                self.grad_gamma[j] += dy_row[j] * xh_row[j];
                self.grad_beta[j] += dy_row[j];
            }
        }

        let mut dl_dx = Vec::with_capacity(dl_dy.len());
        for (i, dy_row) in dl_dy.iter().enumerate() {
            let inv_std = self.cache_inv_std[i];
            let xh_row = &self.cache_x_hat[i];

            // g_j = γ_j * dy_j
            let g: Vec<f32> = dy_row
                .iter()
                .zip(self.gamma.iter())
                .map(|(dy, gam)| dy * gam)
                .collect();

            let sum_g: f32 = g.iter().sum();
            let sum_g_xh: f32 = g.iter().zip(xh_row).map(|(gj, xhj)| gj * xhj).sum();

            let dx_row: Vec<f32> = g
                .iter()
                .zip(xh_row.iter())
                .map(|(gi, xhi)| inv_std / d * (d * gi - sum_g - xhi * sum_g_xh))
                .collect();
            dl_dx.push(dx_row);
        }

        dl_dx
    }

    pub fn zero_grad(&mut self) {
        self.grad_gamma.fill(0.0);
        self.grad_beta.fill(0.0);
    }

    pub fn apply_gradients(&mut self, opt: &mut AdamW, prefix: &str) {
        opt.step_vector(
            &format!("{prefix}.gamma"),
            &mut self.gamma,
            &self.grad_gamma,
        );
        opt.step_vector(&format!("{prefix}.beta"), &mut self.beta, &self.grad_beta);
    }
}

impl Checkpointable for LayerNormalization {
    fn to_weight_map(&self) -> WeightMap {
        let mut map = WeightMap::new();
        map.insert_vector("gamma", self.gamma.clone());
        map.insert_vector("beta", self.beta.clone());
        map
    }

    fn from_weight_map(&mut self, map: &WeightMap) -> Result<()> {
        let gamma = map.get_vector("gamma")?.clone();
        let beta = map.get_vector("beta")?.clone();
        if gamma.len() != self.gamma.len() || beta.len() != self.beta.len() {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "Layer normalization shape mismatch",
            ));
        }
        self.gamma = gamma;
        self.beta = beta;
        Ok(())
    }
}
