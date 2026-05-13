use crate::matrix::Matrix;

pub struct CrossEntropyLoss;

impl CrossEntropyLoss {
    pub fn forward(logits: &[f32], target: usize) -> (f32, Vec<f32>) {
        // Softmax
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();

        // Loss
        let loss = -(probs[target] + 1e-10).ln();

        // 勾配: dL/d_logit_i = probs[i] - 1(i == target)
        let grad: Vec<f32> = probs
            .iter()
            .enumerate()
            .map(|(i, &p)| if i == target { p - 1.0 } else { p })
            .collect();

        (loss, grad)
    }

    /// `logits` は (seq_len, vocab) の Matrix、 戻り値 grads も同形。
    /// rayon で行ごとに並列化、 grads の内部バッファは flat。
    pub fn forward_sequence(
        logits: &Matrix,
        targets: &[usize],
        mask: &[u8],
    ) -> (f32, Matrix) {
        use rayon::prelude::*;

        let (seq_len, vocab) = logits.shape();
        assert_eq!(seq_len, targets.len(), "targets length mismatch");
        assert_eq!(seq_len, mask.len(), "mask length mismatch");

        let valid_count = mask.iter().filter(|&&m| m == 1).count() as f32;
        let inv_count = 1.0 / valid_count.max(1.0);

        let mut grad_data = vec![0.0f32; seq_len * vocab];
        // 行ごとに loss + grad を埋めて、 loss を sum
        let total_loss: f32 = grad_data
            .par_chunks_mut(vocab)
            .enumerate()
            .map(|(t, g_row)| {
                if mask[t] == 0 {
                    return 0.0;
                }
                let logits_row = logits.row(t);
                let (loss, grad) = Self::forward(logits_row, targets[t]);
                for j in 0..vocab {
                    g_row[j] = grad[j] * inv_count;
                }
                loss
            })
            .sum();
        let avg_loss = total_loss * inv_count;
        (avg_loss, Matrix::from_flat(grad_data, seq_len, vocab))
    }
}
