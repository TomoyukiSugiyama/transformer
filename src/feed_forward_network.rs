use std::io::Error;
use std::io::ErrorKind;
use std::io::Result;

use rand::RngExt;
use rand::rng;

use crate::adam_w::AdamW;
use crate::checkpoint::Checkpointable;
use crate::checkpoint::WeightMap;

pub struct FeedForwardNetwork {
    w1: Vec<Vec<f32>>, // (d_model, d_ff)
    b1: Vec<f32>,      // (d_ff,)
    w2: Vec<Vec<f32>>, // (d_ff, d_model)
    b2: Vec<f32>,      // (d_model,)

    grad_w1: Vec<Vec<f32>>,
    grad_b1: Vec<f32>,
    grad_w2: Vec<Vec<f32>>,
    grad_b2: Vec<f32>,

    cache_x: Vec<Vec<f32>>,  // 入力 x        (seq_len, d_model)
    cache_z1: Vec<Vec<f32>>, // GELU前の値    (seq_len, d_ff)
    cache_a: Vec<Vec<f32>>,  // GELU後の値    (seq_len, d_ff)

    d_model: usize,
    d_ff: usize,
}

impl FeedForwardNetwork {
    pub fn new(d_model: usize, d_ff: usize) -> Self {
        let mut rng = rng();
        let scale_model = (2.0 / d_model as f32).sqrt();
        let scale_ff = (2.0 / d_ff as f32).sqrt();
        let mut rand_matrix = |raws: usize, cols: usize, scale: f32| -> Vec<Vec<f32>> {
            (0..raws)
                .map(|_| (0..cols).map(|_| rng.random_range(-scale..scale)).collect())
                .collect()
        };
        Self {
            w1: rand_matrix(d_model, d_ff, scale_model),
            b1: vec![0.0; d_ff],
            w2: rand_matrix(d_ff, d_model, scale_ff),
            b2: vec![0.0; d_model],

            grad_w1: vec![vec![0.0; d_ff]; d_model],
            grad_b1: vec![0.0; d_ff],
            grad_w2: vec![vec![0.0; d_model]; d_ff],
            grad_b2: vec![0.0; d_model],

            cache_x: vec![],
            cache_z1: vec![],
            cache_a: vec![],

            d_model,
            d_ff,
        }
    }

    fn gelu(x: f32) -> f32 {
        let c = (2.0_f32 / std::f32::consts::PI).sqrt();
        0.5 * x * (1.0 + (c * (x + 0.044715 * x.powi(3))).tanh())
    }

    fn gelu_grad(x: f32) -> f32 {
        let c = (2.0_f32 / std::f32::consts::PI).sqrt();
        let tanh_val = (c * (x + 0.044715 * x.powi(3))).tanh();
        let sech2 = 1.0 - tanh_val.powi(2);
        0.5 * (1.0 + tanh_val) + 0.5 * x * sech2 * c * (1.0 + 3.0 * 0.044715 * x.powi(2))
    }

    pub fn forward(&mut self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        use crate::utility::matmul;

        self.cache_x = x.to_vec();

        // Layer 1: z1 = x @ W1 + b1   shape: (seq_len, d_ff)
        let mut z1 = matmul(x, &self.w1);
        for row in z1.iter_mut() {
            for (j, b) in self.b1.iter().enumerate() {
                row[j] += *b;
            }
        }

        // a = GELU(z1)
        let a: Vec<Vec<f32>> = z1
            .iter()
            .map(|row| row.iter().map(|&v| Self::gelu(v)).collect())
            .collect();

        // Layer 2: z2 = a @ W2 + b2   shape: (seq_len, d_model)
        let mut z2 = matmul(&a, &self.w2);
        for row in z2.iter_mut() {
            for (j, b) in self.b2.iter().enumerate() {
                row[j] += *b;
            }
        }

        self.cache_z1 = z1;
        self.cache_a = a;

        z2
    }

    pub fn backward(&mut self, dl_dz2: &[Vec<f32>]) -> Vec<Vec<f32>> {
        use crate::utility::{add_matrix_in_place, matmul, transpose};

        // --- W2 の勾配 ---
        // grad_w2 += cache_a^T @ dl_dz2   shape: (d_ff, d_model)
        let g_w2 = matmul(&transpose(&self.cache_a), dl_dz2);
        add_matrix_in_place(&mut self.grad_w2, &g_w2);

        // --- b2 の勾配 ---
        // grad_b2 += Σ_t dl_dz2[t]
        for row in dl_dz2.iter() {
            for (j, v) in row.iter().enumerate() {
                self.grad_b2[j] += *v;
            }
        }

        // --- GELU 手前まで逆伝播 ---
        // dL/da = dl_dz2 @ W2^T   shape: (seq_len, d_ff)
        let dl_da = matmul(dl_dz2, &transpose(&self.w2));

        // dL/dz1 = dL/da ⊙ GELU'(z1)   shape: (seq_len, d_ff)
        let dl_dz1: Vec<Vec<f32>> = dl_da
            .iter()
            .zip(self.cache_z1.iter())
            .map(|(da_row, z1_row)| {
                da_row
                    .iter()
                    .zip(z1_row.iter())
                    .map(|(&da, &z)| da * Self::gelu_grad(z))
                    .collect()
            })
            .collect();

        // --- W1 の勾配 ---
        // grad_w1 += cache_x^T @ dl_dz1   shape: (d_model, d_ff)
        let g_w1 = matmul(&transpose(&self.cache_x), &dl_dz1);
        add_matrix_in_place(&mut self.grad_w1, &g_w1);

        // --- b1 の勾配 ---
        // grad_b1 += Σ_t dl_dz1[t]
        for row in dl_dz1.iter() {
            for (j, v) in row.iter().enumerate() {
                self.grad_b1[j] += *v;
            }
        }

        // --- 上流への勾配 ---
        // dl_dx = dl_dz1 @ W1^T   shape: (seq_len, d_model)
        matmul(&dl_dz1, &transpose(&self.w1))
    }

    pub fn zero_grad(&mut self) {
        for row in &mut self.grad_w1 {
            row.fill(0.0);
        }
        self.grad_b1.fill(0.0);
        for row in &mut self.grad_w2 {
            row.fill(0.0);
        }
        self.grad_b2.fill(0.0);
    }

    pub fn apply_gradients(&mut self, opt: &mut AdamW, prefix: &str) {
        opt.step_matrix(&format!("{prefix}.w1"), &mut self.w1, &self.grad_w1);
        opt.step_matrix(&format!("{prefix}.w2"), &mut self.w2, &self.grad_w2);

        let mut b1_mat = vec![self.b1.clone()];
        opt.step_matrix(
            &format!("{prefix}.b1"),
            &mut b1_mat,
            &[self.grad_b1.clone()],
        );
        self.b1 = b1_mat.remove(0);

        let mut b2_mat = vec![self.b2.clone()];
        opt.step_matrix(
            &format!("{prefix}.b2"),
            &mut b2_mat,
            &[self.grad_b2.clone()],
        );
        self.b2 = b2_mat.remove(0);
    }
}

impl Checkpointable for FeedForwardNetwork {
    fn to_weight_map(&self) -> WeightMap {
        let mut map = WeightMap::new();
        map.insert_scalar("d_model", self.d_model as u64);
        map.insert_scalar("d_ff", self.d_ff as u64);
        map.insert_matrix("w1", self.w1.clone());
        map.insert_vector("b1", self.b1.clone());
        map.insert_matrix("w2", self.w2.clone());
        map.insert_vector("b2", self.b2.clone());
        map
    }

    fn from_weight_map(&mut self, map: &WeightMap) -> Result<()> {
        let d_model = map.get_scalar("d_model")? as usize;
        let d_ff = map.get_scalar("d_ff")? as usize;
        if d_model != self.d_model || d_ff != self.d_ff {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "Feed forward network config mismatch",
            ));
        }
        self.w1 = map.get_matrix("w1")?.clone();
        self.b1 = map.get_vector("b1")?.clone();
        self.w2 = map.get_matrix("w2")?.clone();
        self.b2 = map.get_vector("b2")?.clone();
        Ok(())
    }
}
