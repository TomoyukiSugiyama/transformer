use std::io::Error;
use std::io::ErrorKind;
use std::io::Result;

use rand::RngExt;
use rand::rng;

use crate::adam_w::AdamW;
use crate::checkpoint::Checkpointable;
use crate::checkpoint::WeightMap;
use crate::feed_forward::FeedForward;
use crate::matrix::Matrix;

pub struct FeedForwardNetwork {
    w1: Matrix,   // (d_model, d_ff)
    b1: Vec<f32>, // (d_ff,)
    w2: Matrix,   // (d_ff, d_model)
    b2: Vec<f32>, // (d_model,)

    grad_w1: Matrix,
    grad_b1: Vec<f32>,
    grad_w2: Matrix,
    grad_b2: Vec<f32>,

    cache_x: Matrix,  // 入力 x        (seq_len, d_model)
    cache_z1: Matrix, // GELU前の値    (seq_len, d_ff)
    cache_a: Matrix,  // GELU後の値    (seq_len, d_ff)

    d_model: usize,
    d_ff: usize,
}

impl FeedForwardNetwork {
    pub fn new(d_model: usize, d_ff: usize) -> Self {
        let mut rng = rng();
        let scale_model = (2.0 / d_model as f32).sqrt();
        let scale_ff = (2.0 / d_ff as f32).sqrt();
        let mut rand_matrix = |rows: usize, cols: usize, scale: f32| -> Matrix {
            let mut m = Matrix::zeros(rows, cols);
            for v in m.data_mut() {
                *v = rng.random_range(-scale..scale);
            }
            m
        };
        Self {
            w1: rand_matrix(d_model, d_ff, scale_model),
            b1: vec![0.0; d_ff],
            w2: rand_matrix(d_ff, d_model, scale_ff),
            b2: vec![0.0; d_model],

            grad_w1: Matrix::zeros(d_model, d_ff),
            grad_b1: vec![0.0; d_ff],
            grad_w2: Matrix::zeros(d_ff, d_model),
            grad_b2: vec![0.0; d_model],

            cache_x: Matrix::zeros(0, 0),
            cache_z1: Matrix::zeros(0, 0),
            cache_a: Matrix::zeros(0, 0),

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

    pub fn forward(&mut self, x: &Matrix) -> Matrix {
        self.cache_x = x.clone();

        // Layer 1: z1 = x @ W1 + b1   shape: (seq_len, d_ff)
        let mut z1 = self.cache_x.matmul(&self.w1);
        z1.add_row_bias_in_place(&self.b1);

        // a = GELU(z1)
        let a = z1.map(Self::gelu);

        // Layer 2: z2 = a @ W2 + b2   shape: (seq_len, d_model)
        let mut z2 = a.matmul(&self.w2);
        z2.add_row_bias_in_place(&self.b2);

        self.cache_z1 = z1;
        self.cache_a = a;

        z2
    }

    pub fn backward(&mut self, dl_dz2: &Matrix) -> Matrix {
        // --- W2 の勾配 ---
        // grad_w2 += cache_a^T @ dl_dz2   shape: (d_ff, d_model)
        // Phase 7-4: BLAS の beta=1 で fused matmul-add 化 (旧来は temp alloc + add_in_place)。
        self.cache_a.matmul_t1_add_into(dl_dz2, &mut self.grad_w2);

        // --- b2 の勾配 ---
        // grad_b2 += Σ_t dl_dz2[t]
        let g_b2 = dl_dz2.sum_rows_into_cols();
        for (b, g) in self.grad_b2.iter_mut().zip(g_b2.iter()) {
            *b += *g;
        }

        // --- GELU 手前まで逆伝播 ---
        // dL/da = dl_dz2 @ W2^T   shape: (seq_len, d_ff)
        // Phase 7-3: matmul_t2 で W2 の転置 materialize を回避
        let dl_da = dl_dz2.matmul_t2(&self.w2);

        // dL/dz1 = dL/da ⊙ GELU'(z1)   shape: (seq_len, d_ff)
        let dl_dz1 = dl_da.elementwise_with(&self.cache_z1, |da, z| da * Self::gelu_grad(z));

        // --- W1 の勾配 ---
        // grad_w1 += cache_x^T @ dl_dz1   shape: (d_model, d_ff)
        // Phase 7-4: BLAS の beta=1 で fused matmul-add 化。
        self.cache_x
            .matmul_t1_add_into(&dl_dz1, &mut self.grad_w1);

        // --- b1 の勾配 ---
        // grad_b1 += Σ_t dl_dz1[t]
        let g_b1 = dl_dz1.sum_rows_into_cols();
        for (b, g) in self.grad_b1.iter_mut().zip(g_b1.iter()) {
            *b += *g;
        }

        // --- 上流への勾配 ---
        // dl_dx = dl_dz1 @ W1^T   shape: (seq_len, d_model)
        // Phase 7-3: matmul_t2 で W1 の転置 materialize を回避
        dl_dz1.matmul_t2(&self.w1)
    }

    pub fn zero_grad(&mut self) {
        self.grad_w1.data_mut().fill(0.0);
        self.grad_b1.fill(0.0);
        self.grad_w2.data_mut().fill(0.0);
        self.grad_b2.fill(0.0);
    }

    pub fn apply_gradients(&mut self, opt: &mut AdamW, prefix: &str) {
        opt.step_matrix_flat(&format!("{prefix}.w1"), &mut self.w1, &self.grad_w1);
        opt.step_matrix_flat(&format!("{prefix}.w2"), &mut self.w2, &self.grad_w2);
        opt.step_vector(&format!("{prefix}.b1"), &mut self.b1, &self.grad_b1);
        opt.step_vector(&format!("{prefix}.b2"), &mut self.b2, &self.grad_b2);
    }
}

impl FeedForward for FeedForwardNetwork {
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

impl Checkpointable for FeedForwardNetwork {
    fn to_weight_map(&self) -> WeightMap {
        let mut map = WeightMap::new();
        map.insert_scalar("d_model", self.d_model as u64);
        map.insert_scalar("d_ff", self.d_ff as u64);
        map.insert_matrix("w1", self.w1.to_jagged());
        map.insert_vector("b1", self.b1.clone());
        map.insert_matrix("w2", self.w2.to_jagged());
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
        self.w1 = Matrix::from_jagged(map.get_matrix("w1")?);
        self.b1 = map.get_vector("b1")?.clone();
        self.w2 = Matrix::from_jagged(map.get_matrix("w2")?);
        self.b2 = map.get_vector("b2")?.clone();
        Ok(())
    }
}
