//! Inverted dropout (PyTorch / nanoGPT と同じ仕様)。
//!
//! - 学習中: 各要素を確率 `p` でゼロ化、 残った要素は `1/(1-p)` で
//!   スケールアップして期待値を維持する。
//! - 推論中: 入力をそのまま素通しさせる (mask 情報は保持しない)。
//!
//! `forward` 時の mask を保存し、 `backward` で同じ mask を勾配に再適用する。

use rand::{RngExt, rng};

pub struct Dropout {
    p: f32,
    training: bool,
    /// 直近の forward で適用した mask (training 時のみ作成)。
    /// 各要素は 0.0 (drop) もしくは `1/(1-p)` (keep)。
    mask: Vec<Vec<f32>>,
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
        }
    }

    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    /// 学習時はマスクを掛けて scale-up、 推論時は素通し。
    pub fn forward(&mut self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        if !self.training || self.p == 0.0 {
            self.mask.clear();
            return x.to_vec();
        }
        let keep = 1.0 - self.p;
        let scale = 1.0 / keep;
        let mut rng = rng();
        let mut mask: Vec<Vec<f32>> = Vec::with_capacity(x.len());
        let mut out: Vec<Vec<f32>> = Vec::with_capacity(x.len());
        for row in x {
            let mut mask_row = Vec::with_capacity(row.len());
            let mut out_row = Vec::with_capacity(row.len());
            for &v in row {
                let r: f32 = rng.random_range(0.0..1.0);
                let m = if r < keep { scale } else { 0.0 };
                mask_row.push(m);
                out_row.push(v * m);
            }
            mask.push(mask_row);
            out.push(out_row);
        }
        self.mask = mask;
        out
    }

    /// forward で保存した mask を勾配にも適用する。
    /// mask が空 (= 推論モードで forward した) のときは素通し。
    pub fn backward(&self, dl_dy: &[Vec<f32>]) -> Vec<Vec<f32>> {
        if self.mask.is_empty() {
            return dl_dy.to_vec();
        }
        assert_eq!(self.mask.len(), dl_dy.len());
        dl_dy
            .iter()
            .zip(self.mask.iter())
            .map(|(dy_row, m_row)| {
                debug_assert_eq!(dy_row.len(), m_row.len());
                dy_row.iter().zip(m_row.iter()).map(|(g, m)| g * m).collect()
            })
            .collect()
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
