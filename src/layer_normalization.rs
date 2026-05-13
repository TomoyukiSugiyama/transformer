use std::io::Error;
use std::io::ErrorKind;
use std::io::Result;

use rayon::prelude::*;

use crate::matrix::Matrix;
use crate::normalization::Normalization;
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

    cache_x_hat: Matrix,     // (seq_len, d_model)
    cache_inv_std: Vec<f32>, // (seq_len,) — 1/σ
}

impl LayerNormalization {
    pub fn new(d_model: usize) -> Self {
        Self {
            gamma: vec![1.0; d_model],
            beta: vec![0.0; d_model],
            eps: 1e-6,
            grad_gamma: vec![0.0; d_model],
            grad_beta: vec![0.0; d_model],
            cache_x_hat: Matrix::zeros(0, 0),
            cache_inv_std: Vec::new(),
        }
    }

    /// 行ごとに rayon で並列化。
    /// cache_x_hat / cache_inv_std を更新し backward から再利用する。
    pub fn forward(&mut self, x: &Matrix) -> Matrix {
        let (seq_len, d_model) = x.shape();
        let n = d_model as f32;
        let eps = self.eps;
        let gamma = &self.gamma;
        let beta = &self.beta;

        // 出力: y, x_hat, inv_std を一括に確保
        let mut y_data = vec![0.0f32; seq_len * d_model];
        let mut xh_data = vec![0.0f32; seq_len * d_model];
        let mut inv_std = vec![0.0f32; seq_len];

        y_data
            .par_chunks_mut(d_model)
            .zip(xh_data.par_chunks_mut(d_model))
            .zip(inv_std.par_iter_mut())
            .enumerate()
            .for_each(|(i, ((y_row, xh_row), inv_s_slot))| {
                let xi = x.row(i);
                let mean = xi.iter().sum::<f32>() / n;
                let var = xi.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
                let inv_s = 1.0 / (var + eps).sqrt();
                *inv_s_slot = inv_s;
                for j in 0..d_model {
                    let xh = (xi[j] - mean) * inv_s;
                    xh_row[j] = xh;
                    y_row[j] = gamma[j] * xh + beta[j];
                }
            });

        self.cache_x_hat = Matrix::from_flat(xh_data, seq_len, d_model);
        self.cache_inv_std = inv_std;
        Matrix::from_flat(y_data, seq_len, d_model)
    }

    /// grad_gamma / grad_beta は加算 (zero_grad 後に呼ぶ前提)。
    pub fn backward(&mut self, dl_dy: &Matrix) -> Matrix {
        let (seq_len, d_model) = dl_dy.shape();
        let d = d_model as f32;
        let gamma = &self.gamma;

        // --- grad_gamma / grad_beta の集計 ---
        // grad_gamma[j] += Σ_i dy[i,j] * x_hat[i,j]
        // grad_beta[j]  += Σ_i dy[i,j]
        // rayon で並列に部分和を取って最後に集計する。
        let xh = &self.cache_x_hat;
        let (sum_gamma, sum_beta) = (0..seq_len)
            .into_par_iter()
            .fold(
                || (vec![0.0f32; d_model], vec![0.0f32; d_model]),
                |(mut g, mut b), i| {
                    let dy_row = dl_dy.row(i);
                    let xh_row = xh.row(i);
                    for j in 0..d_model {
                        g[j] += dy_row[j] * xh_row[j];
                        b[j] += dy_row[j];
                    }
                    (g, b)
                },
            )
            .reduce(
                || (vec![0.0f32; d_model], vec![0.0f32; d_model]),
                |(mut ga, mut ba), (gb, bb)| {
                    for j in 0..d_model {
                        ga[j] += gb[j];
                        ba[j] += bb[j];
                    }
                    (ga, ba)
                },
            );
        for j in 0..d_model {
            self.grad_gamma[j] += sum_gamma[j];
            self.grad_beta[j] += sum_beta[j];
        }

        // --- dl/dx を行ごとに計算 (各行独立) ---
        let inv_std = &self.cache_inv_std;
        let mut dx_data = vec![0.0f32; seq_len * d_model];
        dx_data
            .par_chunks_mut(d_model)
            .enumerate()
            .for_each(|(i, dx_row)| {
                let dy_row = dl_dy.row(i);
                let xh_row = xh.row(i);
                let is = inv_std[i];

                // g_j = γ_j * dy_j
                let mut sum_g = 0.0f32;
                let mut sum_g_xh = 0.0f32;
                let mut g = vec![0.0f32; d_model];
                for j in 0..d_model {
                    let gv = gamma[j] * dy_row[j];
                    g[j] = gv;
                    sum_g += gv;
                    sum_g_xh += gv * xh_row[j];
                }
                let coef = is / d;
                for j in 0..d_model {
                    dx_row[j] = coef * (d * g[j] - sum_g - xh_row[j] * sum_g_xh);
                }
            });
        Matrix::from_flat(dx_data, seq_len, d_model)
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

impl Normalization for LayerNormalization {
    fn forward(&mut self, x: &Matrix) -> Matrix {
        self.forward(x)
    }

    fn backward(&mut self, dl_dy: &Matrix) -> Matrix {
        self.backward(dl_dy)
    }

    fn zero_grad(&mut self) {
        self.zero_grad();
    }

    fn apply_gradients(&mut self, opt: &mut AdamW, prefix: &str) {
        self.apply_gradients(opt, prefix);
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
