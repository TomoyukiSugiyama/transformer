use std::io::Error;
use std::io::ErrorKind;
use std::io::Result;

use rand::RngExt;
use rand::rng;

use crate::adam_w::AdamW;
use crate::checkpoint::Checkpointable;
use crate::checkpoint::WeightMap;
use crate::matrix::Matrix;

pub struct OutputHead {
    w: Matrix,            // (d_model, vocab_size)
    grad_w: Matrix,       // (d_model, vocab_size)
    cache_hidden: Matrix, // (seq_len, d_model)
    vocab_size: usize,
    d_model: usize,
}

impl OutputHead {
    pub fn new(d_model: usize, vocab_size: usize) -> Self {
        let mut rng = rng();
        let scale = (1.0 / d_model as f32).sqrt();
        let mut w = Matrix::zeros(d_model, vocab_size);
        for v in w.data_mut() {
            *v = rng.random_range(-scale..scale);
        }
        Self {
            w,
            grad_w: Matrix::zeros(d_model, vocab_size),
            cache_hidden: Matrix::zeros(0, 0),
            vocab_size,
            d_model,
        }
    }

    #[allow(dead_code)]
    pub fn logits_last(&self, last_hidden: &[f32]) -> Vec<f32> {
        use rayon::prelude::*;
        let cols = self.w.cols();
        (0..cols)
            .into_par_iter()
            .map(|j| {
                last_hidden
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| x * self.w.get(i, j))
                    .sum::<f32>()
            })
            .collect()
    }

    pub fn softmax(logits: &[f32]) -> Vec<f32> {
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        exps.iter().map(|e| e / sum).collect()
    }

    #[allow(dead_code)]
    pub fn greedy(probs: &[f32]) -> usize {
        probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    }

    pub fn top_k_sample(logits: &[f32], k: usize, temperature: f32) -> usize {
        let mut indexed: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed.truncate(k);

        let top_logits: Vec<f32> = indexed.iter().map(|(_, l)| l / temperature).collect();
        let probes = Self::softmax(&top_logits);
        let mut rng = rng();
        let r: f32 = rng.random_range(0.0..1.0);
        let mut custom = 0.0;

        for (idx, &p) in probes.iter().enumerate() {
            custom += p;
            if r < custom {
                return indexed[idx].0;
            }
        }
        indexed[0].0
    }

    /// top-p (nucleus) sampling の候補生成 (テスト・サンプリングで共用)。
    ///
    /// 累積確率が `p` を初めて超える位置までを候補として残し、 候補内で確率を再正規化する。
    /// 戻り値の各 `(vocab_id, prob)` の prob は候補集合内の正規化済み確率で、 合計は 1.0 (浮動小数誤差除く)。
    ///
    /// `p = 1.0` で全候補、 `p` が極小で確信度最大の 1 候補のみが返る。
    /// 必ず最低 1 候補を返す (空集合にならない)。
    pub fn top_p_candidates(logits: &[f32], p: f32, temperature: f32) -> Vec<(usize, f32)> {
        assert!(temperature > 0.0, "temperature must be > 0, got {temperature}");
        assert!(
            (0.0..=1.0).contains(&p),
            "top-p must be in [0.0, 1.0], got {p}"
        );

        let scaled: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();
        let probs = Self::softmax(&scaled);

        let mut indexed: Vec<(usize, f32)> = probs.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // 累積確率が初めて p を超える index で打ち切り (inclusive)。
        // p=1.0 のときは浮動小数誤差で 1.0 に達しない可能性があるので、 デフォルトで全件を残す。
        let mut cutoff = indexed.len();
        let mut cum = 0.0;
        for (i, &(_, prob)) in indexed.iter().enumerate() {
            cum += prob;
            if cum >= p {
                cutoff = i + 1;
                break;
            }
        }

        let mut kept: Vec<(usize, f32)> = indexed.into_iter().take(cutoff).collect();
        let sum: f32 = kept.iter().map(|&(_, q)| q).sum();
        for entry in kept.iter_mut() {
            entry.1 /= sum;
        }
        kept
    }

    pub fn top_p_sample(logits: &[f32], p: f32, temperature: f32) -> usize {
        let kept = Self::top_p_candidates(logits, p, temperature);
        let mut rng = rng();
        let r: f32 = rng.random_range(0.0..1.0);
        let mut acc = 0.0;
        for &(idx, prob) in &kept {
            acc += prob;
            if r < acc {
                return idx;
            }
        }
        kept[0].0
    }

    /// Matrix 直叩き forward (Phase 7 高速化で導入)。
    /// (seq_len, d_model) → (seq_len, vocab_size)
    pub fn forward_matrix(&mut self, hidden: &Matrix) -> Matrix {
        self.cache_hidden = hidden.clone();
        // logits = hidden @ W   shape: (seq_len, vocab_size)
        self.cache_hidden.matmul(&self.w)
    }

    /// Matrix 直叩き backward (Phase 7 高速化で導入)。
    /// dL/d_logits (seq_len, vocab_size) → dL/d_hidden (seq_len, d_model)
    pub fn backward_matrix(&mut self, dl_dlogits: &Matrix) -> Matrix {
        // grad_w += cache_hidden^T @ dl_dlogits   shape: (d_model, vocab_size)
        let g_w = self.cache_hidden.transpose().matmul(dl_dlogits);
        self.grad_w.add_in_place(&g_w);

        // dl_dhidden = dl_dlogits @ W^T   shape: (seq_len, d_model)
        dl_dlogits.matmul(&self.w.transpose())
    }

    /// 旧 API: jagged → Matrix 経由。
    pub fn forward(&mut self, hidden: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let xm = Matrix::from_jagged(hidden);
        self.forward_matrix(&xm).to_jagged()
    }

    /// 旧 API: jagged → Matrix 経由。
    pub fn backward(&mut self, dl_dlogits: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let dy = Matrix::from_jagged(dl_dlogits);
        self.backward_matrix(&dy).to_jagged()
    }

    pub fn zero_grad(&mut self) {
        self.grad_w.data_mut().fill(0.0);
    }

    pub fn apply_gradients(&mut self, opt: &mut AdamW, prefix: &str) {
        opt.step_matrix_flat(&format!("{prefix}.w"), &mut self.w, &self.grad_w);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// p=1.0 は全候補を残す (確率の合計が 1.0 に正規化される)。
    #[test]
    fn top_p_one_keeps_all_candidates() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 0.5];
        let kept = OutputHead::top_p_candidates(&logits, 1.0, 1.0);
        assert_eq!(kept.len(), logits.len());
        let sum: f32 = kept.iter().map(|&(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum should be 1.0, got {sum}");

        // 元の vocab id が全て揃っているかチェック
        let mut ids: Vec<usize> = kept.iter().map(|&(i, _)| i).collect();
        ids.sort();
        assert_eq!(ids, vec![0, 1, 2, 3, 4]);
    }

    /// p が極小だと top-1 のみが残り、 確率は 1.0 になる。
    #[test]
    fn top_p_zero_keeps_top_one() {
        let logits = vec![1.0, 5.0, 2.0, 0.5]; // top-1 は index=1
        let kept = OutputHead::top_p_candidates(&logits, 0.0, 1.0);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].0, 1);
        assert!((kept[0].1 - 1.0).abs() < 1e-6);
    }

    /// 累積確率が p を「初めて超えた」 index で打ち切る (inclusive)。
    /// 確率が [0.5, 0.3, 0.15, 0.05] となるような logits を構成し、 p=0.7 で 2 候補だけ残るか確認。
    #[test]
    fn top_p_cutoff_is_inclusive_at_first_overshoot() {
        // softmax(logits) ≈ [0.5, 0.3, 0.15, 0.05] となる logits を逆算:
        // log(p_i / p_0) を logits 差として与える
        let probs = [0.5_f32, 0.3, 0.15, 0.05];
        let logits: Vec<f32> = probs.iter().map(|p| p.ln()).collect();

        // p=0.7: 累積 0.5 (1 件) → 0.8 (2 件超過) で打ち切り、 残るのは 2 件
        let kept = OutputHead::top_p_candidates(&logits, 0.7, 1.0);
        assert_eq!(kept.len(), 2);
        // 元 idx 0 と 1 のはず (top-2)
        let mut kept_ids: Vec<usize> = kept.iter().map(|&(i, _)| i).collect();
        kept_ids.sort();
        assert_eq!(kept_ids, vec![0, 1]);
        // 再正規化後の合計
        let sum: f32 = kept.iter().map(|&(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-5);
        // 比率は元の 0.5:0.3 を保つ (再正規化されただけ)
        // 上位が 0.5/0.8 = 0.625, 次が 0.3/0.8 = 0.375
        let top = kept.iter().find(|&&(i, _)| i == 0).unwrap().1;
        let next = kept.iter().find(|&&(i, _)| i == 1).unwrap().1;
        assert!((top - 0.625).abs() < 1e-5);
        assert!((next - 0.375).abs() < 1e-5);
    }

    /// temperature が確率分布の鋭さを正しく変える:
    /// 同じ logits でも t<1 で分布が尖り、 t>1 で平坦化する。
    /// p を固定したとき、 t<1 では候補が減り、 t>1 では増える方向のはず。
    #[test]
    fn top_p_temperature_affects_distribution_sharpness() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sharp = OutputHead::top_p_candidates(&logits, 0.9, 0.5);
        let flat = OutputHead::top_p_candidates(&logits, 0.9, 2.0);
        // 鋭いほど少数候補で 90% に到達、 平坦ほど多数必要
        assert!(
            sharp.len() <= flat.len(),
            "sharp(t=0.5)={} flat(t=2.0)={}",
            sharp.len(),
            flat.len()
        );
        // 同じ logits で順位は保たれる: top-1 は idx=4 (logit=5.0)
        assert_eq!(sharp[0].0, 4);
        assert_eq!(flat[0].0, 4);
    }

    /// 浮動小数誤差で p=1.0 でも cumsum が 0.999... になっても全候補が残る。
    #[test]
    fn top_p_one_robust_to_float_error() {
        let logits = vec![0.0_f32; 100]; // 一様分布
        let kept = OutputHead::top_p_candidates(&logits, 1.0, 1.0);
        assert_eq!(kept.len(), 100);
        let sum: f32 = kept.iter().map(|&(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-4);
    }

    /// 同じ vocab id は重複しない。
    #[test]
    fn top_p_returns_unique_ids() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let kept = OutputHead::top_p_candidates(&logits, 0.8, 1.0);
        let mut ids: Vec<usize> = kept.iter().map(|&(i, _)| i).collect();
        let before = ids.len();
        ids.sort();
        ids.dedup();
        assert_eq!(before, ids.len(), "ids must be unique");
    }

    /// top_p_sample は kept 集合から id を返す。
    #[test]
    fn top_p_sample_returns_id_from_kept_set() {
        let logits = vec![1.0, 10.0, 2.0]; // top-1 は idx=1 (圧倒的)
        // p=0.5 ならほぼ確実に top-1 のみが kept になる
        let id = OutputHead::top_p_sample(&logits, 0.5, 1.0);
        assert_eq!(id, 1);
    }
}

impl Checkpointable for OutputHead {
    fn to_weight_map(&self) -> WeightMap {
        let mut map = WeightMap::new();
        map.insert_scalar("vocab_size", self.vocab_size as u64);
        map.insert_scalar("d_model", self.d_model as u64);
        map.insert_matrix("w", self.w.to_jagged());
        map
    }

    fn from_weight_map(&mut self, map: &WeightMap) -> Result<()> {
        let vocab_size = map.get_scalar("vocab_size")? as usize;
        let d_model = map.get_scalar("d_model")? as usize;
        if vocab_size != self.vocab_size || d_model != self.d_model {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "Output head config mismatch",
            ));
        }
        self.w = Matrix::from_jagged(map.get_matrix("w")?);
        Ok(())
    }
}
