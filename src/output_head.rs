use rand::RngExt;
use rand::rng;

pub struct OutputHead {
    pub w: Vec<Vec<f32>>,
    vocab_size: usize,
    d_model: usize,
}

impl OutputHead {
    pub fn new(d_model: usize, vocab_size: usize) -> Self {
        let mut rng = rng();
        let scale = (1.0 / d_model as f32).sqrt();
        let w = (0..d_model)
            .map(|_| {
                (0..vocab_size)
                    .map(|_| rng.random_range(-scale..scale))
                    .collect()
            })
            .collect();
        Self {
            w,
            vocab_size,
            d_model,
        }
    }

    pub fn logits_last(&self, last_hidden: &[f32]) -> Vec<f32> {
        (0..self.vocab_size)
            .map(|j| {
                last_hidden
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| x * self.w[i][j])
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

    pub fn forward(&self, hidden: &[Vec<f32>]) -> Vec<Vec<f32>> {
        hidden
            .iter()
            .map(|row: &Vec<f32>| self.logits_last(row))
            .collect()
    }
}
