//! Inverted dropout (PyTorch / nanoGPT と同じ仕様)。
//!
//! - 学習中: 各要素を確率 `p` でゼロ化、 残った要素は `1/(1-p)` で
//!   スケールアップして期待値を維持する。
//! - 推論中: 入力をそのまま素通しさせる (mask 情報は保持しない)。
//!
//! `forward` 時の mask を保存し、 `backward` で同じ mask を勾配に再適用する。

use rand::{RngExt, rng};

use crate::matrix::Matrix;

pub struct Dropout {
    p: f32,
    training: bool,
    /// 直近の forward で適用した mask (training 時のみ作成)。
    /// flat 表現で持ち、 `mask_shape` (rows, cols) で形を覚える。
    /// 各要素は 0.0 (drop) もしくは `1/(1-p)` (keep)。
    mask: Vec<f32>,
    mask_shape: (usize, usize),
}

impl Dropout {
    pub fn new(p: f32) -> Self {
        assert!(
            (0.0..1.0).contains(&p),
            "dropout p must be in [0.0, 1.0), got {p}"
        );
        Self {
            p,
            training: true,
            mask: Vec::new(),
            mask_shape: (0, 0),
        }
    }

    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    /// Matrix 直叩き forward。 学習時はマスクを掛けて scale-up、 推論時は素通し (clone)。
    pub fn forward_matrix(&mut self, x: &Matrix) -> Matrix {
        if !self.training || self.p == 0.0 {
            self.mask.clear();
            self.mask_shape = (0, 0);
            return x.clone();
        }
        let keep = 1.0 - self.p;
        let scale = 1.0 / keep;
        let n = x.data().len();
        let mut rng = rng();
        let mut mask = vec![0.0f32; n];
        let mut out = vec![0.0f32; n];
        for i in 0..n {
            let r: f32 = rng.random_range(0.0..1.0);
            let m = if r < keep { scale } else { 0.0 };
            mask[i] = m;
            out[i] = x.data()[i] * m;
        }
        self.mask = mask;
        self.mask_shape = x.shape();
        Matrix::from_flat(out, x.rows(), x.cols())
    }

    /// Matrix 直叩き backward。
    pub fn backward_matrix(&self, dl_dy: &Matrix) -> Matrix {
        if self.mask.is_empty() {
            return dl_dy.clone();
        }
        assert_eq!(self.mask_shape, dl_dy.shape());
        let n = dl_dy.data().len();
        let mut out = vec![0.0f32; n];
        for i in 0..n {
            out[i] = dl_dy.data()[i] * self.mask[i];
        }
        Matrix::from_flat(out, dl_dy.rows(), dl_dy.cols())
    }

    /// 旧 API: jagged → Matrix 経由。
    pub fn forward(&mut self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let xm = Matrix::from_jagged(x);
        self.forward_matrix(&xm).to_jagged()
    }

    /// 旧 API: jagged → Matrix 経由。
    pub fn backward(&self, dl_dy: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let dy = Matrix::from_jagged(dl_dy);
        self.backward_matrix(&dy).to_jagged()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn p_zero_is_identity_in_training() {
        let mut d = Dropout::new(0.0);
        let x = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let y = d.forward(&x);
        assert_eq!(y, x);
        let g = vec![vec![0.5; 3]; 2];
        assert_eq!(d.backward(&g), g);
    }

    #[test]
    fn inference_mode_is_identity() {
        let mut d = Dropout::new(0.5);
        d.set_training(false);
        let x = vec![vec![1.0; 100]];
        let y = d.forward(&x);
        assert_eq!(y, x);
        // mask が空なので backward も identity
        let g = vec![vec![0.7; 100]];
        assert_eq!(d.backward(&g), g);
    }

    #[test]
    fn training_mode_drops_some_and_scales_others() {
        let mut d = Dropout::new(0.5);
        let x = vec![vec![1.0; 10000]];
        let y = d.forward(&x);
        let row = &y[0];
        // 各要素は 0.0 か 2.0 (= 1/0.5) のどちらか
        for &v in row {
            assert!(v == 0.0 || (v - 2.0).abs() < 1e-6);
        }
        // ドロップ率がだいたい 0.5 ± 0.05 の範囲に収まる
        let dropped = row.iter().filter(|&&v| v == 0.0).count();
        let dropped_ratio = dropped as f32 / row.len() as f32;
        assert!(
            (0.45..0.55).contains(&dropped_ratio),
            "dropped_ratio={dropped_ratio}"
        );
    }

    #[test]
    fn backward_applies_same_mask() {
        let mut d = Dropout::new(0.3);
        let x = vec![vec![10.0; 50]];
        let y = d.forward(&x);
        let g = vec![vec![1.0; 50]];
        let dx = d.backward(&g);
        // forward で 0.0 になった位置は backward でも 0.0 (mask は同じ)
        for i in 0..50 {
            if y[0][i] == 0.0 {
                assert_eq!(dx[0][i], 0.0);
            } else {
                // 通った位置は scale = 1/(1-0.3) = 1.4286...
                let scale = 1.0 / 0.7;
                assert!((dx[0][i] - scale).abs() < 1e-5);
            }
        }
    }
}
