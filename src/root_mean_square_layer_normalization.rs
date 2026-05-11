use std::io::Error;
use std::io::ErrorKind;
use std::io::Result;

use crate::normalization::Normalization;
use crate::{
    adam_w::AdamW,
    checkpoint::{Checkpointable, WeightMap},
};

pub struct RootMeanSquareLayerNormalization {
    gain: Vec<f32>, // [d_model] — γ 相当 (init 1.0)、 RMSNorm では β は持たない

    eps: f32, // 1e-6 で OK (LayerNorm と同じ)

    grad_gain: Vec<f32>, // [d_model]

    cache_x_hat: Vec<Vec<f32>>, // [seq_len, d_model] — x / (rms + eps)
    cache_inv_rms: Vec<f32>,    // [seq_len] — 1.0 / (rms + eps)
}

impl RootMeanSquareLayerNormalization {
    pub fn new(d_model: usize) -> Self {
        Self {
            gain: vec![1.0; d_model],
            eps: 1e-6,
            grad_gain: vec![0.0; d_model],
            cache_x_hat: Vec::new(),
            cache_inv_rms: Vec::new(),
        }
    }

    fn normalization(&mut self, x: &[f32]) -> Vec<f32> {
        let d = x.len() as f32;

        let ms = x.iter().map(|v| v.powi(2)).sum::<f32>() / d;

        let inv_rms = 1.0 / (ms + self.eps).sqrt();

        let x_hat: Vec<f32> = x.iter().map(|v| v * inv_rms).collect();

        let y: Vec<f32> = x_hat
            .iter()
            .enumerate()
            .map(|(i, xh)| self.gain[i] * xh)
            .collect();

        self.cache_x_hat.push(x_hat);
        self.cache_inv_rms.push(inv_rms);

        y
    }

    pub fn forward(&mut self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        self.cache_x_hat.clear();
        self.cache_inv_rms.clear();

        x.iter().map(|row| self.normalization(row)).collect()
    }

    /// dl_dy: [seq_len, d_model]
    pub fn backward(&mut self, dl_dy: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let d = self.gain.len() as f32;

        for (dy_row, xh_row) in dl_dy.iter().zip(self.cache_x_hat.iter()) {
            for j in 0..self.gain.len() {
                self.grad_gain[j] += dy_row[j] * xh_row[j];
            }
        }

        let mut dl_dx = Vec::with_capacity(dl_dy.len());
        for (i, dy_row) in dl_dy.iter().enumerate() {
            let inv_rms = self.cache_inv_rms[i];
            let xh_row = &self.cache_x_hat[i];

            // g_j = γ_j * dy_j
            let g: Vec<f32> = dy_row
                .iter()
                .zip(self.gain.iter())
                .map(|(dy, gam)| dy * gam)
                .collect();

            let sum_g_xh: f32 = g.iter().zip(xh_row).map(|(gj, xhj)| gj * xhj).sum();

            let dx_row: Vec<f32> = g
                .iter()
                .zip(xh_row.iter())
                .map(|(gi, xhi)| inv_rms / d * (d * gi - xhi * sum_g_xh))
                .collect();
            dl_dx.push(dx_row);
        }

        dl_dx
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
