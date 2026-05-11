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

pub struct SwiGluFeedForwardNetwork {
    w_gate: Matrix, // (d_model x d_ff_g)
    w_up: Matrix,   // (d_model x d_ff_g)
    w_down: Matrix, // (d_ff_g x d_model)
    // bias なし (LLaMA 流)
    grad_w_gate: Matrix,
    grad_w_up: Matrix,
    grad_w_down: Matrix,

    cache_x: Matrix,    // (seq x d_model)
    cache_gate: Matrix, // (seq x d_ff_g) Swish 前の値
    cache_up: Matrix,   // (seq x d_ff_g)
    cache_a: Matrix,    // (seq x d_ff_g)　Swish(gate) ⊙ up

    d_model: usize,
    d_ff_g: usize,
}

impl SwiGluFeedForwardNetwork {
    pub fn new(d_model: usize, d_ff: usize) -> Self {
        let d_ff_g = (d_ff * 2) / 3;
        let mut rng = rng();
        let scale_model = (2.0 / d_model as f32).sqrt();
        let scale_ff = (2.0 / d_ff_g as f32).sqrt();
        let mut rand_matrix = |rows: usize, cols: usize, scale: f32| -> Matrix {
            let mut m = Matrix::zeros(rows, cols);
            for v in m.data_mut() {
                *v = rng.random_range(-scale..scale);
            }
            m
        };
        Self {
            w_gate: rand_matrix(d_model, d_ff_g, scale_model),
            w_up: rand_matrix(d_model, d_ff_g, scale_model),
            w_down: rand_matrix(d_ff_g, d_model, scale_ff),
            grad_w_gate: Matrix::zeros(d_model, d_ff_g),
            grad_w_up: Matrix::zeros(d_model, d_ff_g),
            grad_w_down: Matrix::zeros(d_ff_g, d_model),
            cache_x: Matrix::zeros(0, 0),
            cache_gate: Matrix::zeros(0, 0),
            cache_up: Matrix::zeros(0, 0),
            cache_a: Matrix::zeros(0, 0),
            d_model,
            d_ff_g,
        }
    }

    fn swish(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    fn swish_grad(x: f32) -> f32 {
        let s = 1.0 / (1.0 + (-x).exp());
        s + x * s * (1.0 - s)
    }

    pub fn forward(&mut self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let x_m = Matrix::from_jagged(x);
        let gate = x_m.matmul(&self.w_gate);
        let up = x_m.matmul(&self.w_up);

        // a = swish(gate) * up
        let a = gate.elementwise_with(&up, |g, u| Self::swish(g) * u);

        let y = a.matmul(&self.w_down);

        self.cache_x = x_m;
        self.cache_gate = gate;
        self.cache_up = up;
        self.cache_a = a;

        y.to_jagged()
    }

    pub fn backward(&mut self, dl_dy: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let dl_dy_m = Matrix::from_jagged(dl_dy);

        // W_down の grad
        let g_w_down = self.cache_a.transpose().matmul(&dl_dy_m);
        self.grad_w_down.add_in_place(&g_w_down);

        // dL/da
        let dl_da = dl_dy_m.matmul(&self.w_down.transpose());

        // dL/dgate, dL/dup を作る (要素積 + Swish' 適用)
        let dl_da_du = dl_da.elementwise_with(&self.cache_up, |da, u| da * u);
        let dl_dup = dl_da.elementwise_with(&self.cache_gate, |da, g| da * Self::swish(g));
        let dl_dgate =
            dl_da_du.elementwise_with(&self.cache_gate, |da_du, g| da_du * Self::swish_grad(g));

        // W_gate / W_up の grad
        let g_w_gate = self.cache_x.transpose().matmul(&dl_dgate);
        let g_w_up = self.cache_x.transpose().matmul(&dl_dup);
        self.grad_w_gate.add_in_place(&g_w_gate);
        self.grad_w_up.add_in_place(&g_w_up);

        // dL/dx = dL/dgate @ W_gate^T + dL/dup @ W_up^T
        let dl_dx_gate = dl_dgate.matmul(&self.w_gate.transpose());
        let dl_dx_up = dl_dup.matmul(&self.w_up.transpose());
        let mut dl_dx = dl_dx_gate;
        dl_dx.add_in_place(&dl_dx_up);

        dl_dx.to_jagged()
    }

    pub fn zero_grad(&mut self) {
        self.grad_w_gate.data_mut().fill(0.0);
        self.grad_w_up.data_mut().fill(0.0);
        self.grad_w_down.data_mut().fill(0.0);
    }

    pub fn apply_gradients(&mut self, opt: &mut AdamW, prefix: &str) {
        opt.step_matrix_flat(
            &format!("{prefix}.w_gate"),
            &mut self.w_gate,
            &self.grad_w_gate,
        );
        opt.step_matrix_flat(&format!("{prefix}.w_up"), &mut self.w_up, &self.grad_w_up);
        opt.step_matrix_flat(
            &format!("{prefix}.w_down"),
            &mut self.w_down,
            &self.grad_w_down,
        );
    }
}

impl FeedForward for SwiGluFeedForwardNetwork {
    fn forward(&mut self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        self.forward(x)
    }

    fn backward(&mut self, dl_dy: &[Vec<f32>]) -> Vec<Vec<f32>> {
        self.backward(dl_dy)
    }

    fn zero_grad(&mut self) {
        self.zero_grad();
    }

    fn apply_gradients(&mut self, opt: &mut AdamW, prefix: &str) {
        self.apply_gradients(opt, prefix);
    }
}

impl Checkpointable for SwiGluFeedForwardNetwork {
    fn to_weight_map(&self) -> WeightMap {
        let mut map = WeightMap::new();
        map.insert_scalar("d_model", self.d_model as u64);
        map.insert_scalar("d_ff_g", self.d_ff_g as u64);
        map.insert_matrix("w_gate", self.w_gate.to_jagged());
        map.insert_matrix("w_up", self.w_up.to_jagged());
        map.insert_matrix("w_down", self.w_down.to_jagged());
        map
    }

    fn from_weight_map(&mut self, map: &WeightMap) -> Result<()> {
        let d_model = map.get_scalar("d_model")? as usize;
        let d_ff_g = map.get_scalar("d_ff_g")? as usize;
        if d_model != self.d_model || d_ff_g != self.d_ff_g {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "Feed forward network config mismatch",
            ));
        }
        self.w_gate = Matrix::from_jagged(map.get_matrix("w_gate")?);
        self.w_up = Matrix::from_jagged(map.get_matrix("w_up")?);
        self.w_down = Matrix::from_jagged(map.get_matrix("w_down")?);
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn swish_known_values() {
        // swish(x) = x · sigmoid(x)
        // swish(0)  = 0 · 0.5 = 0
        // swish(1)  = 1 · sigmoid(1) ≈ 1 · 0.7311 = 0.7311
        // swish(-1) = -1 · sigmoid(-1) ≈ -1 · 0.2689 = -0.2689
        // swish(2)  = 2 · sigmoid(2) ≈ 2 · 0.8808 = 1.7616
        assert!((SwiGluFeedForwardNetwork::swish(0.0) - 0.0).abs() < 1e-5);
        assert!((SwiGluFeedForwardNetwork::swish(1.0) - 0.7311).abs() < 1e-3);
        assert!((SwiGluFeedForwardNetwork::swish(-1.0) - (-0.2689)).abs() < 1e-3);
        assert!((SwiGluFeedForwardNetwork::swish(2.0) - 1.7616).abs() < 1e-3);
    }

    #[test]
    fn swish_grad_known_values() {
        // swish'(x) = sigmoid(x) + x · sigmoid(x) · (1 - sigmoid(x))
        // swish'(0) = 0.5 + 0 = 0.5
        // swish'(1) = 0.7311 + 1 · 0.7311 · 0.2689 ≈ 0.9277
        // swish'(-1) ≈ 0.2689 + (-1) · 0.2689 · 0.7311 ≈ 0.0723
        assert!((SwiGluFeedForwardNetwork::swish_grad(0.0) - 0.5).abs() < 1e-5);
        assert!((SwiGluFeedForwardNetwork::swish_grad(1.0) - 0.9277).abs() < 1e-3);
        assert!((SwiGluFeedForwardNetwork::swish_grad(-1.0) - 0.0723).abs() < 1e-3);
    }

    #[test]
    fn forward_with_fixed_weights() {
        // 最小構成: d_model=1, d_ff=2 → d_ff_g=1, seq=1
        let mut sw = SwiGluFeedForwardNetwork::new(1, 2);

        // 全行列を 1.0 で固定 (Matrix::from_flat or data_mut().fill(1.0))
        sw.w_gate.data_mut().fill(1.0);
        sw.w_up.data_mut().fill(1.0);
        sw.w_down.data_mut().fill(1.0);

        // x = [[2.0]]
        let x = vec![vec![2.0]];
        let y = sw.forward(&x);

        // gate = x @ w_gate = [[2.0]]
        // up   = x @ w_up   = [[2.0]]
        // a    = swish(2.0) * 2.0 = 1.7616 * 2.0 = 3.5233
        // y    = a @ w_down = [[3.5233]]
        assert!((y[0][0] - 3.5233).abs() < 1e-3);
    }

    #[test]
    fn forward_param_matched_d_ff_g() {
        // (d_ff * 2) / 3 (整数除算で切り捨て)
        let sw = SwiGluFeedForwardNetwork::new(384, 1536);
        assert_eq!(sw.d_ff_g, 1024); // 1536 * 2 / 3 = 1024 ぴったり

        let sw = SwiGluFeedForwardNetwork::new(8, 12);
        assert_eq!(sw.d_ff_g, 8); // 12 * 2 / 3 = 8

        let sw = SwiGluFeedForwardNetwork::new(8, 10);
        assert_eq!(sw.d_ff_g, 6); // 10 * 2 / 3 = 20/3 = 6 (切り捨て)

        // 行列の shape も確認
        assert_eq!(sw.w_gate.shape(), (8, 6));
        assert_eq!(sw.w_up.shape(), (8, 6));
        assert_eq!(sw.w_down.shape(), (6, 8));
    }

    #[test]
    fn backward_matches_numerical_gradient() {
        use rand::{RngExt, SeedableRng, rngs::SmallRng};

        // 小さい構成 — 3 行列で数値微分するので少しでも軽くする
        let d_model = 4;
        let d_ff = 6; // d_ff_g = 4
        let seq_len = 2;

        // 重みもテスト用に決定的に作る (rng を渡せれば良いが、 ここでは new() のランダムを許容)
        let mut sw = SwiGluFeedForwardNetwork::new(d_model, d_ff);

        // 入力もランダム (seed 固定)
        let mut rng = SmallRng::seed_from_u64(42);
        let x: Vec<Vec<f32>> = (0..seq_len)
            .map(|_| (0..d_model).map(|_| rng.random_range(-1.0..1.0)).collect())
            .collect();

        // ★ 解析的勾配を最初に 1 回だけ取得 ★
        let _ = sw.forward(&x);
        let dl_dy = vec![vec![1.0; d_model]; seq_len];
        let dl_dx_analytic = sw.backward(&dl_dy);

        // 数値微分: L = sum(y) と置く (∂L/∂y_ij = 1)
        let h = 1e-3;
        for i in 0..seq_len {
            for j in 0..d_model {
                let mut x_p = x.clone();
                x_p[i][j] += h;
                let y_p = sw.forward(&x_p);
                let l_p: f32 = y_p.iter().flatten().sum();

                let mut x_m = x.clone();
                x_m[i][j] -= h;
                let y_m = sw.forward(&x_m);
                let l_m: f32 = y_m.iter().flatten().sum();

                let num = (l_p - l_m) / (2.0 * h);

                assert!(
                    (dl_dx_analytic[i][j] - num).abs() < 1e-2,
                    "mismatch at ({i},{j}): analytic={}, numerical={}",
                    dl_dx_analytic[i][j],
                    num
                );
            }
        }
    }

    #[test]
    fn checkpoint_roundtrip_preserves_outputs() {
        let d_model = 4;
        let d_ff = 6; // d_ff_g = 4

        let mut a = SwiGluFeedForwardNetwork::new(d_model, d_ff);
        // 重みを上書き (固定値で OK、 デフォルトのランダムでも roundtrip 確認には十分)
        a.w_gate.data_mut().fill(0.5);
        a.w_up.data_mut().fill(-0.3);
        a.w_down.data_mut().fill(0.2);

        let map = a.to_weight_map();

        let mut b = SwiGluFeedForwardNetwork::new(d_model, d_ff);
        b.from_weight_map(&map).unwrap();

        let x = vec![vec![1.0, 2.0, 3.0, 4.0]; 2];
        let y_a = a.forward(&x);
        let y_b = b.forward(&x);

        // 完全一致 (同じ重み × 同じ入力 × 同じ計算経路)
        assert_eq!(y_a, y_b);
    }
}
