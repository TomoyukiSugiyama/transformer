use rand::RngExt;
use rand::rng;

use crate::adam_w::AdamW;

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
        0.5 * x * (1.0 + (c * (x + 0.44715 * x.powi(3))).tanh())
    }

    fn gelu_grad(x: f32) -> f32 {
        let c = (2.0_f32 / std::f32::consts::PI).sqrt();
        let tanh_val = (c * (x + 0.44715 * x.powi(3))).tanh();
        let sech2 = 1.0 - tanh_val.powi(2);
        0.5 + (1.0 + tanh_val) + 0.5 * x * sech2 * c * (1.0 + 3.0 * 0.44715 * x.powi(2))
    }

    fn forward_one(&mut self, x: &[f32]) -> Vec<f32> {
        // Layer 1: (d_model,) × W1(d_model, d_ff) + b1 → (d_ff,)
        // z1 = x @ W1 + b1  (GELU前の値をキャッシュ)
        let z1: Vec<f32> = (0..self.d_ff)
            .map(|j| {
                self.b1[j]
                    + x.iter()
                        .enumerate()
                        .map(|(i, &xi)| xi * self.w1[i][j])
                        .sum::<f32>()
            })
            .collect();

        // a = GELU(z1)
        let a: Vec<f32> = z1.iter().map(|&v| Self::gelu(v)).collect();

        // Layer 2: (d_ff,) × W2(d_ff, d_model) + b2 → (d_model,)
        // z2 = a @ W2 + b2
        let z2: Vec<f32> = (0..self.d_model)
            .map(|j| {
                self.b2[j]
                    + a.iter()
                        .enumerate()
                        .map(|(i, &ai)| ai * self.w2[i][j])
                        .sum::<f32>()
            })
            .collect();

        self.cache_z1.push(z1);
        self.cache_a.push(a);

        z2
    }

    pub fn forward(&mut self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        self.cache_x = x.to_vec();
        self.cache_z1 = vec![];
        self.cache_a = vec![];

        x.iter().map(|row| self.forward_one(row)).collect()
    }

    pub fn backward(&mut self, dl_dz2: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let seq_len = dl_dz2.len();

        self.grad_w1 = vec![vec![0.0; self.d_ff]; self.d_model];
        self.grad_b1 = vec![0.0; self.d_ff];
        self.grad_w2 = vec![vec![0.0; self.d_model]; self.d_ff];
        self.grad_b2 = vec![0.0; self.d_model];

        let mut dl_dx = vec![vec![0.0; self.d_model]; seq_len];

        for t in 0..seq_len {
            // --- W2, b2 の勾配 ---
            // dL/dW2 += a[t]^T ⊗ dl_dz2[t]
            for i in 0..self.d_ff {
                for j in 0..self.d_model {
                    self.grad_w2[i][j] += self.cache_a[t][i] * dl_dz2[t][j];
                }
            }

            // dL/db2 += dl_dz2[t]
            for j in 0..self.d_model {
                self.grad_b2[j] += dl_dz2[t][j];
            }

            // --- GELU 手前まで逆伝播 ---
            // dL/da = dl_dz2[t] @ W2^T   shape: (d_ff,)
            let dl_da: Vec<f32> = (0..self.d_ff)
                .map(|i| {
                    (0..self.d_model)
                        .map(|j| dl_dz2[t][j] * self.w2[i][j])
                        .sum()
                })
                .collect();

            // dL/dz1 = dL/da ⊙ GELU'(z1)   shape: (d_ff,)
            let dl_dz1: Vec<f32> = (0..self.d_ff)
                .map(|i| dl_da[i] * Self::gelu_grad(self.cache_z1[t][i]))
                .collect();

            // --- W1, b1 の勾配 ---
            // dL/dW1 += x[t]^T ⊗ dl_dz1
            for i in 0..self.d_model {
                for j in 0..self.d_ff {
                    self.grad_w1[i][j] += self.cache_x[t][i] * dl_dz1[j];
                }
            }

            // dL/db1 += dl_dz1
            for j in 0..self.d_ff {
                self.grad_b1[j] += dl_dz1[j];
            }

            // --- 上流への勾配 ---
            // dL/dx = dl_dz1 @ W1^T   shape: (d_model,)
            for i in 0..self.d_model {
                dl_dx[t][i] = (0..self.d_ff).map(|j| dl_dz1[j] * self.w1[i][j]).sum();
            }
        }

        dl_dx
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
