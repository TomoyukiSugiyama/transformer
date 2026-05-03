use rand::RngExt;
use rand::rng;

pub struct Embedding {
    weight: Vec<Vec<f32>>,
    vocab_size: usize,
    d_model: usize,
}

impl Embedding {
    pub fn new(vocab_size: usize, d_model: usize) -> Self {
        let mut rng = rng();
        let scale = (1.0 / d_model as f32).sqrt();
        let weight: Vec<Vec<f32>> = (0..vocab_size)
            .map(|_| {
                (0..d_model)
                    .map(|_| rng.random_range(-scale..scale))
                    .collect()
            })
            .collect();
        Self {
            weight,
            vocab_size,
            d_model,
        }
    }

    fn lookup(&self, token_id: usize) -> &[f32] {
        assert!(token_id < self.vocab_size, "token_id out of vocab range");
        &self.weight[token_id]
    }

    pub fn forward(&self, token_ids: &[usize]) -> Vec<Vec<f32>> {
        let scale = (self.d_model as f32).sqrt();

        token_ids
            .iter()
            .map(|&id| self.lookup(id).iter().map(|&v| v * scale).collect())
            .collect()
    }
}
