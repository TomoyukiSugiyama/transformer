use rand::RngExt;
use rand::rng;

pub struct FeedForwardNetwork {
    w1: Vec<Vec<f32>>,
    b1: Vec<f32>,
    w2: Vec<Vec<f32>>,
    b2: Vec<f32>,
    d_model: usize,
    d_ff: usize,
}

impl FeedForwardNetwork {
    pub fn new(d_model: usize, d_ff: usize) -> Self {
        let mut rng = rng();
        let scale_model = (2.0 / d_model as f32).sqrt();
        let scale_ff = (2.0 / d_ff as f32).sqrt();
        let mut rand_matrix = |raws: usize, cols: usize, scale: f32| -> Vec<Vec<f32>> {
            (0..raws)
                .map(|_| (0..cols).map(|_| rng.random_range(-scale..scale)).collect())
                .collect()
        };
        Self {
            w1: rand_matrix(d_model, d_ff, scale_model),
            b1: vec![0.0; d_ff],
            w2: rand_matrix(d_ff, d_model, scale_ff),
            b2: vec![0.0; d_model],
            d_model,
            d_ff,
        }
    }
    fn gelu(x: f32) -> f32 {
        let c = (2.0_f32 / std::f32::consts::PI).sqrt();
        0.5 * x * (1.0 + (c * (x + 0.44715 * x.powi(3))).tanh())
    }
    
    fn forward_one(&self, x: &[f32]) -> Vec<f32> {
        // Layer 1: (d_model,) × W1(d_model, d_ff) + b1 → (d_ff,)
        let mut h = vec![0.0f32;self.d_ff];
        for j in 0..self.d_ff {
            h[j] = self.b1[j] + x.iter().enumerate().map(|(i,&xi)| xi*self.w1[i][j]).sum::<f32>();
            h[j] = Self::gelu(h[j]);
        }

        // Layer 2: (d_ff,) × W2(d_ff, d_model) + b2 → (d_model,)
        let mut out = vec![0.0f32;self.d_model];
        for j in 0..self.d_model {
            out[j] = self.b2[j] + h.iter().enumerate().map(|(i,&hi)| hi*self.w2[i][j]).sum::<f32>();
        }
        out        
    }

    pub fn forward(&self, x:&[Vec<f32>]) -> Vec<Vec<f32>> {
        x.iter().map(|row| self.forward_one(row)).collect()
    }
}


