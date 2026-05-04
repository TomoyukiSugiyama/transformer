pub struct CrossEntropyLoss;

impl CrossEntropyLoss {
    pub fn forward(logits: &[f32], target: usize) -> (f32, Vec<f32>) {
        // Softmax
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();

        // Loss
        let loss = -(probs[target] - 1e-10).ln();

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
        let mut total_loss = 0.0f32;
        let mut grads = vec![vec![0.0f32; logits_seq[0].len()]; logits_seq.len()];
        let valid_count = mask.iter().filter(|&&m| m == 1).count() as f32;

        for (i, ((logits, &target), &m)) in logits_seq
            .iter()
            .zip(targets.iter())
            .zip(mask.iter())
            .enumerate()
        {
            if m == 0 {
                continue;
            }
            let (loss, grad) = Self::forward(logits, target);
            total_loss += loss;
            grads[i] = grad;
        }

        let avg_loss = total_loss / valid_count.max(1.0);

        let scaled_grad: Vec<Vec<f32>> = grads
            .iter()
            .map(|g| g.iter().map(|&v| v / valid_count.max(1.0)).collect())
            .collect();

        (avg_loss, scaled_grad)
    }
}
