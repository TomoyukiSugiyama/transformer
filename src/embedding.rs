use rand::RngExt;
use rand::rng;

pub struct Embedding {
    weight: Vec<Vec<f32>>,      // [vocab_size, d_model]
    grad_weight: Vec<Vec<f32>>, // [vocab_size, d_model]
    cache_ids: Vec<usize>,
    vocab_size: usize,
    d_model: usize,
    pad_id: Option<usize>,
}

impl Embedding {
    pub fn new(vocab_size: usize, d_model: usize, pad_id: Option<usize>) -> Self {
        let mut rng = rng();
        let scale = (1.0 / d_model as f32).sqrt();
        let mut weight: Vec<Vec<f32>> = (0..vocab_size)
            .map(|_| {
                (0..d_model)
                    .map(|_| rng.random_range(-scale..scale))
                    .collect()
            })
            .collect();

        if let Some(pad) = pad_id {
            weight[pad] = vec![0.0; d_model];
        }

        Self {
            weight,
            grad_weight: vec![vec![0.0; d_model]; vocab_size],
            cache_ids: Vec::new(),
            vocab_size,
            d_model,
            pad_id,
        }
    }

    fn lookup(&self, token_id: usize) -> &[f32] {
        assert!(token_id < self.vocab_size, "token_id out of vocab range");
        &self.weight[token_id]
    }

    pub fn forward(&mut self, token_ids: &[usize]) -> Vec<Vec<f32>> {
        self.cache_ids = token_ids.to_vec();
        let scale = (self.d_model as f32).sqrt();

        token_ids
            .iter()
            .map(|&id| self.lookup(id).iter().map(|&v| v * scale).collect())
            .collect()
    }

    pub fn backward(&mut self, dl_dx: &[Vec<f32>]) {
        self.grad_weight = vec![vec![0.0; self.d_model]; self.vocab_size];

        for (i, &id) in self.cache_ids.iter().enumerate() {
            if let Some(pad) = self.pad_id {
                if id == pad {
                    continue;
                }
            }

            for j in 0..self.d_model {
                // 学習の安定性のため scale をかけない
                self.grad_weight[i][j] += dl_dx[i][j];
            }
        }
    }

    pub fn apply_gradients(&mut self, lr: f32) {
        for id in 0..self.vocab_size {
            if let Some(pad) = self.pad_id {
                if id == pad {
                    continue;
                }
            }

            for j in 0..self.d_model {
                self.weight[id][j] -= lr * self.grad_weight[id][j];
            }
        }
    }

    pub fn weight_norm(&self, token_id: usize) -> f32 {
        self.weight[token_id]
            .iter()
            .map(|v| v.powi(2))
            .sum::<f32>()
            .sqrt()
    }
}
