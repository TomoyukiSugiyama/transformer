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

    pub fn forward_sequence(
        logits_seq: &[Vec<f32>],
        targets: &[usize],
        mask: &[u8],
    ) -> (f32, Vec<Vec<f32>>) {
        use rayon::prelude::*;

        let valid_count = mask.iter().filter(|&&m| m == 1).count() as f32;
        let inv_count = 1.0 / valid_count.max(1.0);
        let vocab = logits_seq[0].len();

        // 各 token 独立に (loss, scaled_grad) を計算（masked は zero）
        let results: Vec<(f32, Vec<f32>)> = logits_seq
            .par_iter()
            .zip(targets.par_iter())
            .zip(mask.par_iter())
            .map(|((logits, &target), &m)| {
                if m == 0 {
                    (0.0, vec![0.0f32; vocab])
                } else {
                    let (loss, grad) = Self::forward(logits, target);
                    let scaled: Vec<f32> = grad.into_iter().map(|v| v * inv_count).collect();
                    (loss, scaled)
                }
            })
            .collect();

        let total_loss: f32 = results.iter().map(|(l, _)| *l).sum();
        let avg_loss = total_loss * inv_count;
        let grads: Vec<Vec<f32>> = results.into_iter().map(|(_, g)| g).collect();

        (avg_loss, grads)
    }
}
