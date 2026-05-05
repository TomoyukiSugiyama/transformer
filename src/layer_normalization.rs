pub struct LayerNormalization {
    gamma: Vec<f32>,
    beta: Vec<f32>,
    eps: f32,
}

impl LayerNormalization {
    pub fn new(d_model: usize) -> Self {
        Self {
            gamma: vec![1.0; d_model],
            beta: vec![0.0; d_model],
            eps: 1e-6,
        }
    }

    fn normalization(&self, x: &[f32]) -> Vec<f32> {
        let n = x.len() as f32;
        let mean = x.iter().sum::<f32>() / n;
        let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>();

        x.iter()
            .enumerate()
            .map(|(i, v)| {
                let x_hat = (v - mean) / (var + self.eps).sqrt();
                self.gamma[i] * x_hat + self.beta[i]
            })
            .collect()
    }

    pub fn forward(&self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        x.iter().map(|row| self.normalization(row)).collect()
    }
}
