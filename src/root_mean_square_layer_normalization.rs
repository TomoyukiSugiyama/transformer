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

pub struct RootMeanSquareLayerNormalization {
    gain: Vec<f32>, // [d_model] — γ 相当 (init 1.0)、 RMSNorm では β は持たない

    eps: f32, // 1e-6 で OK (LayerNorm と同じ)

    grad_gain: Vec<f32>, // [d_model]

    cache_x_hat: Matrix,     // (seq_len, d_model) — x * inv_rms
    cache_inv_rms: Vec<f32>, // (seq_len,)
}

impl RootMeanSquareLayerNormalization {
    pub fn new(d_model: usize) -> Self {
        Self {
            gain: vec![1.0; d_model],
            eps: 1e-6,
            grad_gain: vec![0.0; d_model],
            cache_x_hat: Matrix::zeros(0, 0),
            cache_inv_rms: Vec::new(),
        }
    }

    /// Matrix 直叩き forward。 行ごとに rayon で並列化。
    pub fn forward_matrix(&mut self, x: &Matrix) -> Matrix {
        let (seq_len, d_model) = x.shape();
        let n = d_model as f32;
        let eps = self.eps;
        let gain = &self.gain;

        let mut y_data = vec![0.0f32; seq_len * d_model];
        let mut xh_data = vec![0.0f32; seq_len * d_model];
        let mut inv_rms = vec![0.0f32; seq_len];

        y_data
            .par_chunks_mut(d_model)
            .zip(xh_data.par_chunks_mut(d_model))
            .zip(inv_rms.par_iter_mut())
            .enumerate()
            .for_each(|(i, ((y_row, xh_row), inv_r_slot))| {
                let xi = x.row(i);
                let ms = xi.iter().map(|v| v.powi(2)).sum::<f32>() / n;
                let inv_r = 1.0 / (ms + eps).sqrt();
                *inv_r_slot = inv_r;
                for j in 0..d_model {
                    let xh = xi[j] * inv_r;
                    xh_row[j] = xh;
                    y_row[j] = gain[j] * xh;
                }
            });

        self.cache_x_hat = Matrix::from_flat(xh_data, seq_len, d_model);
        self.cache_inv_rms = inv_rms;
        Matrix::from_flat(y_data, seq_len, d_model)
    }

    /// Matrix 直叩き backward。 grad_gain は加算。
    pub fn backward_matrix(&mut self, dl_dy: &Matrix) -> Matrix {
        let (seq_len, d_model) = dl_dy.shape();
        let d = d_model as f32;
        let gain = &self.gain;
        let xh = &self.cache_x_hat;

        // grad_gain の集計を rayon の fold で並列化
        let sum_gain = (0..seq_len)
            .into_par_iter()
            .fold(
                || vec![0.0f32; d_model],
                |mut acc, i| {
                    let dy_row = dl_dy.row(i);
                    let xh_row = xh.row(i);
                    for j in 0..d_model {
                        acc[j] += dy_row[j] * xh_row[j];
                    }
                    acc
                },
            )
            .reduce(
                || vec![0.0f32; d_model],
                |mut a, b| {
                    for j in 0..d_model {
                        a[j] += b[j];
                    }
                    a
                },
            );
        for j in 0..d_model {
            self.grad_gain[j] += sum_gain[j];
        }

        // dl/dx を行ごとに計算
        let inv_rms = &self.cache_inv_rms;
        let mut dx_data = vec![0.0f32; seq_len * d_model];
        dx_data
            .par_chunks_mut(d_model)
            .enumerate()
            .for_each(|(i, dx_row)| {
                let dy_row = dl_dy.row(i);
                let xh_row = xh.row(i);
                let ir = inv_rms[i];

                let mut sum_g_xh = 0.0f32;
                let mut g = vec![0.0f32; d_model];
                for j in 0..d_model {
                    let gv = gain[j] * dy_row[j];
                    g[j] = gv;
                    sum_g_xh += gv * xh_row[j];
                }
                let coef = ir / d;
                for j in 0..d_model {
                    dx_row[j] = coef * (d * g[j] - xh_row[j] * sum_g_xh);
                }
            });
        Matrix::from_flat(dx_data, seq_len, d_model)
    }

    /// 旧 API: jagged → Matrix 経由。
    pub fn forward(&mut self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let xm = Matrix::from_jagged(x);
        self.forward_matrix(&xm).to_jagged()
    }

    /// 旧 API: jagged → Matrix 経由。
    pub fn backward(&mut self, dl_dy: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let dy = Matrix::from_jagged(dl_dy);
        self.backward_matrix(&dy).to_jagged()
    }

    pub fn zero_grad(&mut self) {
        self.grad_gain.fill(0.0);
    }

    pub fn apply_gradients(&mut self, opt: &mut AdamW, prefix: &str) {
        opt.step_vector(&format!("{prefix}.gain"), &mut self.gain, &self.grad_gain);
    }
}

impl Normalization for RootMeanSquareLayerNormalization {
    fn forward(&mut self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        self.forward(x)
    }

    fn backward(&mut self, dl_dy: &[Vec<f32>]) -> Vec<Vec<f32>> {
        self.backward(dl_dy)
    }

    fn forward_matrix(&mut self, x: &Matrix) -> Matrix {
        self.forward_matrix(x)
    }

    fn backward_matrix(&mut self, dl_dy: &Matrix) -> Matrix {
        self.backward_matrix(dl_dy)
    }

    fn zero_grad(&mut self) {
        self.zero_grad();
    }

    fn apply_gradients(&mut self, opt: &mut AdamW, prefix: &str) {
        self.apply_gradients(opt, prefix);
    }
}
impl Checkpointable for RootMeanSquareLayerNormalization {
    fn to_weight_map(&self) -> WeightMap {
        let mut map = WeightMap::new();
        map.insert_vector("gain", self.gain.clone());
        map
    }

    fn from_weight_map(&mut self, map: &WeightMap) -> Result<()> {
        let gain = map.get_vector("gain")?.clone();
        if gain.len() != self.gain.len() {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "Layer normalization shape mismatch",
            ));
        }
        self.gain = gain;
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn forward_normalizes_known_input_to_expected_values() {
        let x: Vec<Vec<f32>> = vec![vec![3.0, 4.0]; 1];
        let d_model = x[0].len();
        let mut rms = RootMeanSquareLayerNormalization::new(d_model);

        // ms = (9 + 16)/2 = 12.5
        // rms = sqrt(12.5 + 1e-6) ≈ 3.5355
        // x_hat = [3/3.5355, 4/3.5355] ≈ [0.8485, 1.1314]
        // y = [0.8485, 1.1314] (gain=1.0 なので変化なし)
        let y = rms.forward(&x);
        assert!((y[0][0] - 0.8485).abs() < 1e-3);
        assert!((y[0][1] - 1.1314).abs() < 1e-3);
    }

    #[test]
    fn forward_handle_small_values() {
        let x: Vec<Vec<f32>> = vec![vec![0.001, 0.002]; 1];
        let d_model = x[0].len();
        let mut rms = RootMeanSquareLayerNormalization::new(d_model);

        // ms = (0.001^2 + 0.002-2)/2 = 2.5e-6
        // rms = sqrt(2.5e-6 + 1e-6) ≈ 1.871e-3
        // x_hat = [0.001/1.871e-3, 0.002/1.871e-3] ≈ [0.5345, 1.0690]
        // y = [0.5345, 1.0690] (gain=1.0 なので変化なし)
        let y = rms.forward(&x);
        assert!((y[0][0] - 0.5345).abs() < 1e-3);
        assert!((y[0][1] - 1.0690).abs() < 1e-3);
    }

    #[test]
    fn foward_handles_constant_input() {
        let x: Vec<Vec<f32>> = vec![vec![2.0, 2.0]; 1];
        let d_model = x[0].len();
        let mut rms = RootMeanSquareLayerNormalization::new(d_model);

        // ms = (2^2 + 2^2)/2 = 4.0
        // rms = sqrt(4.0 + 1e-6) ≈ 2.0000
        // x_hat = [2/2, 2/2] ≈ [1.0000, 1.0000]
        // y = [1.0000, 1.0000] (gain=1.0 なので変化なし)
        let y = rms.forward(&x);
        assert!((y[0][0] - 1.0000).abs() < 1e-3);
        assert!((y[0][1] - 1.0000).abs() < 1e-3);
    }

    #[test]
    fn forward_applies_gain_per_dimention() {
        let x: Vec<Vec<f32>> = vec![vec![3.0, 4.0]; 1];
        let d_model = x[0].len();
        let mut rms = RootMeanSquareLayerNormalization::new(d_model);

        let gain: Vec<f32> = vec![2.0, 0.5];
        rms.gain = gain;

        // y = [0.8485 * 2.0, 1.1314 * 0.5] (gain=[2.0, 0.5])
        let y = rms.forward(&x);
        assert!((y[0][0] - 0.8485 * 2.0).abs() < 1e-3);
        assert!((y[0][1] - 1.1314 * 0.5).abs() < 1e-3);
    }

    #[test]
    fn backward_matches_numerical_gradient() {
        use rand::{RngExt, SeedableRng, rngs::SmallRng};

        let d_model = 8;
        let seq_len = 3;
        let mut rng = SmallRng::seed_from_u64(42);

        let x: Vec<Vec<f32>> = (0..seq_len)
            .map(|_| (0..d_model).map(|_| rng.random_range(-1.0..1.0)).collect())
            .collect();

        let mut rms = RootMeanSquareLayerNormalization::new(d_model);

        let _ = rms.forward(&x);

        let dl_dy = vec![vec![1.0; d_model]; seq_len];
        let dl_dx_analytics = rms.backward(&dl_dy);

        // 数値微分で各 x_ij の勾配を計算: 中心差分 (f(x+h) - f(x-h)) / (2h)
        let h = 1e-3;
        for i in 0..seq_len {
            for j in 0..d_model {
                let mut x_plus = x.clone();
                x_plus[i][j] += h;
                let y_plus = rms.forward(&x_plus);
                let l_plus: f32 = y_plus.iter().flatten().sum();

                let mut x_minus = x.clone();
                x_minus[i][j] -= h;
                let y_minus = rms.forward(&x_minus);
                let l_minus: f32 = y_minus.iter().flatten().sum();

                let dl_dx_numerical = (l_plus - l_minus) / (2.0 * h);

                assert!(
                    (dl_dx_analytics[i][j] - dl_dx_numerical).abs() < 1e-2,
                    "mismatch at ({i}, {j}: analytic={}, numerical={})",
                    dl_dx_analytics[i][j],
                    dl_dx_numerical
                );
            }
        }
    }

    #[test]
    fn checkpoint_roundtrip_preserves_outputs() {
        let d_model = 8;

        let mut a = RootMeanSquareLayerNormalization::new(d_model);
        a.gain = vec![1.5; d_model];

        let map = a.to_weight_map();

        let mut b = RootMeanSquareLayerNormalization::new(d_model);
        b.from_weight_map(&map).unwrap();

        let x: Vec<Vec<f32>> = vec![vec![3.0, 4.0]; 1];
        let y_a = a.forward(&x);
        let y_b = b.forward(&x);

        assert_eq!(y_a, y_b)
    }
}
