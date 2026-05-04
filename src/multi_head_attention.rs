use rand::RngExt;
use rand::rng;

use crate::utility::linear;

pub struct MultiHeadAttention {
    w_q: Vec<Vec<f32>>,
    w_k: Vec<Vec<f32>>,
    w_v: Vec<Vec<f32>>,
    w_o: Vec<Vec<f32>>,
    n_heads: usize,
    d_model: usize,
    d_head: usize,
}

impl MultiHeadAttention {
    pub fn new(d_model: usize, n_heads: usize) -> Self {
        assert!(
            d_model % n_heads == 0,
            "d_model must to dibisible by n_heads"
        );
        let d_head = d_model / n_heads;
        let mut rng = rng();
        let scale = (1.0 / d_model as f32).sqrt();
        let mut rand_matrix = |raws: usize, cols: usize| -> Vec<Vec<f32>> {
            (0..raws)
                .map(|_| (0..cols).map(|_| rng.random_range(-scale..scale)).collect())
                .collect()
        };

        Self {
            w_q: rand_matrix(d_model, d_model),
            w_k: rand_matrix(d_model, d_model),
            w_v: rand_matrix(d_model, d_model),
            w_o: rand_matrix(d_model, d_model),
            n_heads,
            d_model,
            d_head,
        }
    }

    fn split_heads(&self, x: &[Vec<f32>]) -> Vec<Vec<Vec<f32>>> {
        (0..self.n_heads)
            .map(|h| {
                let start = h * self.n_heads;
                x.iter()
                    .map(|row| row[start..start + self.d_head].to_vec())
                    .collect()
            })
            .collect()
    }

    fn concat_heads(&self, heads: &[Vec<Vec<f32>>]) -> Vec<Vec<f32>> {
        let seq_len = heads[0].len();
        (0..seq_len)
            .map(|i| heads.iter().flat_map(|h| h[i].iter().cloned()).collect())
            .collect()
    }

    pub fn forward(
        &self,
        x: &[Vec<f32>],
        mask: Option<&Vec<Vec<bool>>>,
    ) -> (Vec<Vec<f32>>, Vec<Vec<Vec<f32>>>) {
        // Q, K, V を射影
        let q = linear(x, &self.w_q);
        let k = linear(x, &self.w_k);
        let v = linear(x, &self.w_v);

        let q_heads = self.split_heads(&q);
        let k_heads = self.split_heads(&k);
        let v_heads = self.split_heads(&v);

        let mut all_waights = Vec::new();
        let head_outputs: Vec<Vec<Vec<f32>>> = (0..self.n_heads)
            .map(|h| {
                let (out, w) =
                    scaled_dot_product_attention(&q_heads[h], &k_heads[h], &v_heads[h], mask);
                all_waights.push(w);
                out
            })
            .collect();
        let concat = self.concat_heads(&head_outputs);
        let output = linear(&concat, &self.w_o);
        (output, all_waights)
    }
}

fn scaled_dot_product_attention(
    q: &[Vec<f32>],
    k: &[Vec<f32>],
    v: &[Vec<f32>],
    mask: Option<&Vec<Vec<bool>>>,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    use crate::utility::*;
    let d_k = q[0].len() as f32;
    let scale = d_k.sqrt();

    // QK^T / √d_k
    let k_t = transpose(k);
    let mut scores = matmul(q, &k_t);
    scores.iter_mut().flatten().for_each(|s| *s /= scale);

    if let Some(m) = mask {
        for (i, raw) in scores.iter_mut().enumerate() {
            for (j, s) in raw.iter_mut().enumerate() {
                if m[i][j] {
                    *s = f32::NEG_INFINITY;
                }
            }
        }
    }

    // Attention(Q,K,V) = softmax(QK^T / √d_k)V
    softmax_rows(&mut scores);
    let attention_weights = scores.clone();
    let output = matmul(&scores, v);
    (output, attention_weights)
}

pub fn causal_mask(seq_len: usize) -> Vec<Vec<bool>> {
    (0..seq_len)
        .map(|i| (0..seq_len).map(|j| j > i).collect())
        .collect()
}
