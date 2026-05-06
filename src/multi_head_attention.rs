use rand::RngExt;
use rand::rng;

use crate::adam_w::AdamW;
use crate::utility::linear;

pub struct MultiHeadAttention {
    w_q: Vec<Vec<f32>>,
    w_k: Vec<Vec<f32>>,
    w_v: Vec<Vec<f32>>,
    w_o: Vec<Vec<f32>>,

    grad_w_q: Vec<Vec<f32>>,
    grad_w_k: Vec<Vec<f32>>,
    grad_w_v: Vec<Vec<f32>>,
    grad_w_o: Vec<Vec<f32>>,

    cache_x: Vec<Vec<f32>>,
    cache_q: Vec<Vec<f32>>,
    cache_k: Vec<Vec<f32>>,
    cache_v: Vec<Vec<f32>>,
    cache_concat: Vec<Vec<f32>>,
    cache_att_w: Vec<Vec<Vec<f32>>>,
    cache_v_head: Vec<Vec<Vec<f32>>>,

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
        let zeros = |r: usize, c: usize| vec![vec![0.0f32; c]; r];

        Self {
            w_q: rand_matrix(d_model, d_model),
            w_k: rand_matrix(d_model, d_model),
            w_v: rand_matrix(d_model, d_model),
            w_o: rand_matrix(d_model, d_model),
            grad_w_q: zeros(d_model, d_model),
            grad_w_k: zeros(d_model, d_model),
            grad_w_v: zeros(d_model, d_model),
            grad_w_o: zeros(d_model, d_model),
            cache_x: Vec::new(),
            cache_q: Vec::new(),
            cache_k: Vec::new(),
            cache_v: Vec::new(),
            cache_concat: Vec::new(),
            cache_att_w: Vec::new(),
            cache_v_head: Vec::new(),
            n_heads,
            d_model,
            d_head,
        }
    }

    fn split_heads(&self, x: &[Vec<f32>]) -> Vec<Vec<Vec<f32>>> {
        (0..self.n_heads)
            .map(|h| {
                let start = h * self.d_head;
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
        &mut self,
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

        self.cache_x = x.to_vec();
        self.cache_q = q;
        self.cache_k = k;
        self.cache_v = v;
        self.cache_concat = concat;
        self.cache_att_w = all_waights.clone();
        self.cache_v_head = head_outputs;

        (output, all_waights)
    }

    pub fn backward(&mut self, dl_dout: &[Vec<f32>]) -> Vec<Vec<f32>> {
        use crate::utility::*;

        // W_O backward
        // grad_w_o = concat^T @ dl_dout
        self.grad_w_o = matmul(&transpose(&self.cache_concat), dl_dout);
        // dl_dconcat = dl_dout @ W_O^T
        let dl_dconcat = matmul(dl_dout, &transpose(&self.w_o));

        // concat_heads backward → head ごとに dl_dconcat をスライス
        let dl_dhead_outs: Vec<Vec<Vec<f32>>> = (0..self.n_heads)
            .map(|h| {
                let start = h * self.d_head;
                dl_dconcat
                    .iter()
                    .map(|row| row[start..start + self.d_head].to_vec())
                    .collect()
            })
            .collect();
        // scaled_dot_product_attention backward（head ごと）
        let q_heads = self.split_heads(&self.cache_q);
        let k_heads = self.split_heads(&self.cache_k);

        let mut dl_dq_heads =
            vec![vec![vec![0.0f32; self.d_head]; self.cache_x.len()]; self.n_heads];
        let mut dl_dk_heads =
            vec![vec![vec![0.0f32; self.d_head]; self.cache_x.len()]; self.n_heads];
        let mut dl_dv_heads =
            vec![vec![vec![0.0f32; self.d_head]; self.cache_x.len()]; self.n_heads];

        for h in 0..self.n_heads {
            let (dq, dk, dv) = scaled_dot_product_attention_backward(
                &q_heads[h],
                &k_heads[h],
                &self.cache_v_head[h],
                &self.cache_att_w[h],
                &dl_dhead_outs[h],
            );
            dl_dq_heads[h] = dq;
            dl_dk_heads[h] = dk;
            dl_dv_heads[h] = dv;
        }

        // split_heads backward → head の勾配を結合して [seq, d_model] に戻す
        let dl_dq = self.concat_heads(&dl_dq_heads);
        let dl_dk = self.concat_heads(&dl_dk_heads);
        let dl_dv = self.concat_heads(&dl_dv_heads);

        // W_Q/K/V backward
        self.grad_w_q = matmul(&transpose(&self.cache_x), &dl_dq);
        self.grad_w_k = matmul(&transpose(&self.cache_x), &dl_dk);
        self.grad_w_v = matmul(&transpose(&self.cache_x), &dl_dv);

        // dl_dx = dQ @ W_Q^T + dK @ W_K^T + dV @ W_V^T
        let dx_q = matmul(&dl_dq, &transpose(&self.w_q));
        let dx_k = matmul(&dl_dk, &transpose(&self.w_k));
        let dx_v = matmul(&dl_dv, &transpose(&self.w_v));

        let seq = self.cache_x.len();
        (0..seq)
            .map(|i| {
                (0..self.d_model)
                    .map(|j| dx_q[i][j] + dx_k[i][j] + dx_v[i][j])
                    .collect()
            })
            .collect()
    }

    pub fn apply_gradients(&mut self, opt: &mut AdamW, prefix: &str) {
        opt.step_matrix(
            &format!("{prefix}.w_q"),
            &mut self.w_q,
            &self.grad_w_q.clone(),
        );
        opt.step_matrix(
            &format!("{prefix}.w_k"),
            &mut self.w_k,
            &self.grad_w_k.clone(),
        );
        opt.step_matrix(
            &format!("{prefix}.w_v"),
            &mut self.w_v,
            &self.grad_w_v.clone(),
        );
        opt.step_matrix(
            &format!("{prefix}.w_o"),
            &mut self.w_o,
            &self.grad_w_o.clone(),
        );
    }
}

fn scaled_dot_product_attention_backward(
    q: &[Vec<f32>],
    k: &[Vec<f32>],
    v: &[Vec<f32>],
    att_w: &[Vec<f32>],
    dl_dout: &[Vec<f32>],
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>) {
    use crate::utility::*;
    let d_k = q[0].len() as f32;
    let scale = d_k.sqrt();

    // dl_dv = P^T @ dl_dout  [seq, d_head]
    let dl_dv = matmul(att_w, dl_dout);

    // dl_dP = dl_dout @ V^T  [seq, seq]
    let dl_dp = matmul(dl_dout, &transpose(v));

    // softmax backward: dl_dS[i][j] = P[i][j] * (dl_dP[i][j] - Σ_k P[i][k]*dl_dP[i][k])
    let seq = att_w.len();
    let mut dl_ds: Vec<Vec<f32>> = vec![vec![0.0; seq]; seq];
    for i in 0..seq {
        let dot: f32 = (0..seq).map(|k| att_w[i][k] * dl_dp[i][k]).sum();
        for j in 0..seq {
            dl_ds[i][j] = att_w[i][j] * (dl_dp[i][j] - dot);
        }
    }

    // スケールを戻す: dl_dS /= √d_k
    dl_ds.iter_mut().flatten().for_each(|s| *s /= scale);

    // dl_dq = dl_dS @ K  [seq, d_head]
    let dl_dq = matmul(&dl_ds, k);
    // dl_dk = dl_dS^T @ Q  [seq, d_head]
    let dl_dk = matmul(&transpose(&dl_ds), q);

    (dl_dq, dl_dk, dl_dv)
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
