use std::io::Error;
use std::io::ErrorKind;
use std::io::Result;

use rand::RngExt;
use rand::rng;

use crate::adam_w::AdamW;
use crate::checkpoint::Checkpointable;
use crate::checkpoint::WeightMap;
use crate::kv_cache::KvCache;
use crate::matrix::Matrix;
use crate::rope::Rope;

pub struct MultiHeadAttention {
    w_q: Matrix, // (d_model, d_model)
    w_k: Matrix,
    w_v: Matrix,
    w_o: Matrix,

    grad_w_q: Matrix,
    grad_w_k: Matrix,
    grad_w_v: Matrix,
    grad_w_o: Matrix,

    cache_x: Matrix,           // (seq, d_model)
    cache_q: Matrix,           // (seq, d_model) ※未回転
    cache_k: Matrix,           // (seq, d_model) ※未回転
    cache_v: Matrix,           // (seq, d_model)
    cache_concat: Matrix,      // (seq, d_model)
    cache_att_w: Vec<Matrix>,  // n_heads × (seq, seq)
    cache_v_head: Vec<Matrix>, // n_heads × (seq, d_head)

    n_heads: usize,
    d_model: usize,
    d_head: usize,

    /// Some の場合は Q, K に回転位置エンコーディングを適用する。
    /// V には掛けない (RoPE の規約)。
    rope: Option<Rope>,
}

impl MultiHeadAttention {
    pub fn new(d_model: usize, n_heads: usize, rope: Option<Rope>) -> Self {
        assert!(
            d_model % n_heads == 0,
            "d_model must to dibisible by n_heads"
        );
        let d_head = d_model / n_heads;
        if let Some(r) = &rope {
            assert_eq!(
                r.d_head(),
                d_head,
                "Rope d_head ({}) must match MHA d_head ({})",
                r.d_head(),
                d_head
            );
        }
        let mut rng = rng();
        let scale = (1.0 / d_model as f32).sqrt();
        let mut rand_matrix = |rows: usize, cols: usize| -> Matrix {
            let mut m = Matrix::zeros(rows, cols);
            for v in m.data_mut() {
                *v = rng.random_range(-scale..scale);
            }
            m
        };

        Self {
            w_q: rand_matrix(d_model, d_model),
            w_k: rand_matrix(d_model, d_model),
            w_v: rand_matrix(d_model, d_model),
            w_o: rand_matrix(d_model, d_model),
            grad_w_q: Matrix::zeros(d_model, d_model),
            grad_w_k: Matrix::zeros(d_model, d_model),
            grad_w_v: Matrix::zeros(d_model, d_model),
            grad_w_o: Matrix::zeros(d_model, d_model),
            cache_x: Matrix::zeros(0, 0),
            cache_q: Matrix::zeros(0, 0),
            cache_k: Matrix::zeros(0, 0),
            cache_v: Matrix::zeros(0, 0),
            cache_concat: Matrix::zeros(0, 0),
            cache_att_w: Vec::new(),
            cache_v_head: Vec::new(),
            n_heads,
            d_model,
            d_head,
            rope,
        }
    }

    pub fn forward(
        &mut self,
        x: &[Vec<f32>],
        mask: Option<&Vec<Vec<bool>>>,
    ) -> Vec<Vec<f32>> {
        let x_m = Matrix::from_jagged(x);

        // Q, K, V を射影
        let q = x_m.matmul(&self.w_q);
        let k = x_m.matmul(&self.w_k);
        let v = x_m.matmul(&self.w_v);

        let mut q_heads = q.split_columns(self.n_heads);
        let mut k_heads = k.split_columns(self.n_heads);
        let v_heads = v.split_columns(self.n_heads);

        // RoPE: Q と K のみに位置回転を適用 (V には適用しない)。
        // backward でも同じ回転が必要なので、 ここでの結果は捨てて backward で再計算する。
        if let Some(rope) = &self.rope {
            for h in 0..self.n_heads {
                rope.apply_in_place(&mut q_heads[h]);
                rope.apply_in_place(&mut k_heads[h]);
            }
        }

        let mut all_weights = Vec::with_capacity(self.n_heads);
        let mut head_outputs = Vec::with_capacity(self.n_heads);
        for h in 0..self.n_heads {
            let (out, w) =
                scaled_dot_product_attention(&q_heads[h], &k_heads[h], &v_heads[h], mask);
            head_outputs.push(out);
            all_weights.push(w);
        }
        let concat = Matrix::concat_columns(&head_outputs);
        let output = concat.matmul(&self.w_o);

        let output_jagged = output.to_jagged();

        self.cache_x = x_m;
        self.cache_q = q;
        self.cache_k = k;
        self.cache_v = v;
        self.cache_concat = concat;
        self.cache_att_w = all_weights;
        self.cache_v_head = v_heads;

        output_jagged
    }

    pub fn backward(&mut self, dl_dout: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let dl_dout_m = Matrix::from_jagged(dl_dout);

        // W_O backward
        // grad_w_o += concat^T @ dl_dout
        let g_w_o = self.cache_concat.transpose().matmul(&dl_dout_m);
        self.grad_w_o.add_in_place(&g_w_o);
        // dl_dconcat = dl_dout @ W_O^T
        let dl_dconcat = dl_dout_m.matmul(&self.w_o.transpose());

        // concat_heads backward → head ごとに dl_dconcat をスライス
        let dl_dhead_outs = dl_dconcat.split_columns(self.n_heads);

        // scaled_dot_product_attention backward（head ごと）
        // RoPE 使用時、 attention は **回転後** の Q', K' に対して計算されたので、
        // backward の入力にも回転後の値が必要。 cache は未回転なので再回転する。
        let mut q_heads = self.cache_q.split_columns(self.n_heads);
        let mut k_heads = self.cache_k.split_columns(self.n_heads);
        if let Some(rope) = &self.rope {
            for h in 0..self.n_heads {
                rope.apply_in_place(&mut q_heads[h]);
                rope.apply_in_place(&mut k_heads[h]);
            }
        }

        let mut dl_dq_heads = Vec::with_capacity(self.n_heads);
        let mut dl_dk_heads = Vec::with_capacity(self.n_heads);
        let mut dl_dv_heads = Vec::with_capacity(self.n_heads);
        for h in 0..self.n_heads {
            let (dq, dk, dv) = scaled_dot_product_attention_backward(
                &q_heads[h],
                &k_heads[h],
                &self.cache_v_head[h],
                &self.cache_att_w[h],
                &dl_dhead_outs[h],
            );
            dl_dq_heads.push(dq);
            dl_dk_heads.push(dk);
            dl_dv_heads.push(dv);
        }

        // RoPE backward: dL/dQ' から dL/dQ へ (= 逆回転)。 V は未回転なので不要。
        if let Some(rope) = &self.rope {
            for h in 0..self.n_heads {
                rope.apply_backward_in_place(&mut dl_dq_heads[h]);
                rope.apply_backward_in_place(&mut dl_dk_heads[h]);
            }
        }

        // split_heads backward → head の勾配を結合して [seq, d_model] に戻す
        let dl_dq = Matrix::concat_columns(&dl_dq_heads);
        let dl_dk = Matrix::concat_columns(&dl_dk_heads);
        let dl_dv = Matrix::concat_columns(&dl_dv_heads);

        // W_Q/K/V backward (累積)
        let cache_x_t = self.cache_x.transpose();
        let g_w_q = cache_x_t.matmul(&dl_dq);
        let g_w_k = cache_x_t.matmul(&dl_dk);
        let g_w_v = cache_x_t.matmul(&dl_dv);
        self.grad_w_q.add_in_place(&g_w_q);
        self.grad_w_k.add_in_place(&g_w_k);
        self.grad_w_v.add_in_place(&g_w_v);

        // dl_dx = dQ @ W_Q^T + dK @ W_K^T + dV @ W_V^T
        let mut dl_dx = dl_dq.matmul(&self.w_q.transpose());
        dl_dx.add_in_place(&dl_dk.matmul(&self.w_k.transpose()));
        dl_dx.add_in_place(&dl_dv.matmul(&self.w_v.transpose()));
        dl_dx.to_jagged()
    }

    /// 推論専用 (KV cache あり) の 1 token 前進。
    ///
    /// `x_new` は単一 token の hidden state (`d_model` 長)。 内部キャッシュ (training 用)
    /// は **触らない** ので、 学習・validation の途中に呼んでも副作用なし。
    ///
    /// 計算量 (1 step):
    /// - Q/K/V projection: `O(d_model²)`
    /// - attention: `O(cur_len · d_model)` (累積したキャッシュとの内積)
    /// - 出力 projection: `O(d_model²)`
    ///
    /// 旧 `forward(seq_n+1)` は `O((n+1)² · d + (n+1) · d²)` だったので、
    /// step 単位で **約 (n+1) 倍** 高速化される。
    pub fn forward_step(&self, x_new: &[f32], cache: &mut KvCache) -> Vec<f32> {
        assert_eq!(
            x_new.len(),
            self.d_model,
            "MHA::forward_step: x_new len {} != d_model {}",
            x_new.len(),
            self.d_model
        );

        let dh = self.d_head;
        let h_n = self.n_heads;

        // 1 token を 1-row Matrix にして Q, K, V 投影。
        // matmul は (1, d) × (d, d) なので m=1 の細長い形状だが、
        // BLAS sgemm はこの形でも問題なく動く。
        let x_m = Matrix::from_flat(x_new.to_vec(), 1, self.d_model);
        let q = x_m.matmul(&self.w_q);
        let k = x_m.matmul(&self.w_k);
        let v = x_m.matmul(&self.w_v);

        let mut q_data = q.data().to_vec();
        let mut k_data = k.data().to_vec();
        let v_data: &[f32] = v.data();

        // RoPE: 新規 token の Q, K を **位置 cur_len** で head ごとに回転。
        // 過去の K (cache.k) は append 時点で当時の位置の角度で回転済なので不変。
        let pos = cache.cur_len();
        if let Some(rope) = &self.rope {
            for h in 0..h_n {
                rope.apply_at_position(&mut q_data[h * dh..(h + 1) * dh], pos);
                rope.apply_at_position(&mut k_data[h * dh..(h + 1) * dh], pos);
            }
        }

        // 新しい K (回転済) と V を cache に追記。 これで cache.cur_len() == pos + 1。
        cache.append(&k_data, v_data);

        let n = cache.cur_len(); // = pos + 1
        let cache_k = cache.k();
        let cache_v = cache.v();
        let scale = (dh as f32).sqrt();

        // 各 head ごとに 「1 query × n key」 の attention を計算する。
        // causal mask は **不要** — cache に入っているのは過去 + 自分自身だけ。
        let mut concat = vec![0.0f32; self.d_model];
        for h in 0..h_n {
            let q_head: &[f32] = &q_data[h * dh..(h + 1) * dh];

            // scores[j] = (q · K_j) / √d_head, j ∈ [0, n)
            let mut scores = vec![0.0f32; n];
            for j in 0..n {
                let k_row = cache_k.row(j);
                let k_slice = &k_row[h * dh..(h + 1) * dh];
                let mut s = 0.0f32;
                for i in 0..dh {
                    s += q_head[i] * k_slice[i];
                }
                scores[j] = s / scale;
            }

            // softmax (n 要素のみ。 行列 softmax を呼び出すより手書きの方が割安)
            let mut max = f32::NEG_INFINITY;
            for &s in &scores {
                if s > max {
                    max = s;
                }
            }
            let mut sum = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - max).exp();
                sum += *s;
            }
            let inv_sum = 1.0 / sum;
            for s in scores.iter_mut() {
                *s *= inv_sum;
            }

            // out_head[i] = Σ_j probs[j] * V_j[i]
            let dst = &mut concat[h * dh..(h + 1) * dh];
            for j in 0..n {
                let p = scores[j];
                let v_row = cache_v.row(j);
                let v_slice = &v_row[h * dh..(h + 1) * dh];
                for i in 0..dh {
                    dst[i] += p * v_slice[i];
                }
            }
        }

        // Output projection: (1, d_model) × (d_model, d_model) = (1, d_model)
        let concat_m = Matrix::from_flat(concat, 1, self.d_model);
        let output = concat_m.matmul(&self.w_o);
        output.data().to_vec()
    }

    pub fn zero_grad(&mut self) {
        self.grad_w_q.data_mut().fill(0.0);
        self.grad_w_k.data_mut().fill(0.0);
        self.grad_w_v.data_mut().fill(0.0);
        self.grad_w_o.data_mut().fill(0.0);
    }

    pub fn apply_gradients(&mut self, opt: &mut AdamW, prefix: &str) {
        opt.step_matrix_flat(&format!("{prefix}.w_q"), &mut self.w_q, &self.grad_w_q);
        opt.step_matrix_flat(&format!("{prefix}.w_k"), &mut self.w_k, &self.grad_w_k);
        opt.step_matrix_flat(&format!("{prefix}.w_v"), &mut self.w_v, &self.grad_w_v);
        opt.step_matrix_flat(&format!("{prefix}.w_o"), &mut self.w_o, &self.grad_w_o);
    }
}

fn scaled_dot_product_attention_backward(
    q: &Matrix,
    k: &Matrix,
    v: &Matrix,
    att_w: &Matrix,
    dl_dout: &Matrix,
) -> (Matrix, Matrix, Matrix) {
    let d_k = q.cols() as f32;
    let scale = d_k.sqrt();

    // dl_dv = P^T @ dl_dout  [seq, d_head]
    let dl_dv = att_w.transpose().matmul(dl_dout);

    // dl_dP = dl_dout @ V^T  [seq, seq]
    let dl_dp = dl_dout.matmul(&v.transpose());

    // softmax backward: dl_dS[i][j] = P[i][j] * (dl_dP[i][j] - Σ_k P[i][k]*dl_dP[i][k])
    let seq = att_w.rows();
    let mut dl_ds = Matrix::zeros(seq, seq);
    for i in 0..seq {
        let p_row = att_w.row(i);
        let dp_row = dl_dp.row(i);
        let dot: f32 = p_row.iter().zip(dp_row).map(|(p, dp)| p * dp).sum();
        let ds_row = dl_ds.row_mut(i);
        for j in 0..seq {
            ds_row[j] = p_row[j] * (dp_row[j] - dot);
        }
    }

    // スケールを戻す: dl_dS /= √d_k
    for v in dl_ds.data_mut() {
        *v /= scale;
    }

    // dl_dq = dl_dS @ K  [seq, d_head]
    let dl_dq = dl_ds.matmul(k);
    // dl_dk = dl_dS^T @ Q  [seq, d_head]
    let dl_dk = dl_ds.transpose().matmul(q);

    (dl_dq, dl_dk, dl_dv)
}

fn scaled_dot_product_attention(
    q: &Matrix,
    k: &Matrix,
    v: &Matrix,
    mask: Option<&Vec<Vec<bool>>>,
) -> (Matrix, Matrix) {
    let d_k = q.cols() as f32;
    let scale = d_k.sqrt();

    // QK^T / √d_k
    let mut scores = q.matmul(&k.transpose());
    for s in scores.data_mut() {
        *s /= scale;
    }

    if let Some(m) = mask {
        let cols = scores.cols();
        for i in 0..scores.rows() {
            let row = scores.row_mut(i);
            for j in 0..cols {
                if m[i][j] {
                    row[j] = f32::NEG_INFINITY;
                }
            }
        }
    }

    // Attention(Q,K,V) = softmax(QK^T / √d_k)V
    scores.softmax_rows_in_place();
    let attention_weights = scores.clone();
    let output = scores.matmul(v);
    (output, attention_weights)
}

pub fn causal_mask(seq_len: usize) -> Vec<Vec<bool>> {
    (0..seq_len)
        .map(|i| (0..seq_len).map(|j| j > i).collect())
        .collect()
}

impl Checkpointable for MultiHeadAttention {
    fn to_weight_map(&self) -> WeightMap {
        let mut map = WeightMap::new();
        map.insert_scalar("n_heads", self.n_heads as u64);
        map.insert_scalar("d_model", self.d_model as u64);
        map.insert_scalar("d_head", self.d_head as u64);
        map.insert_matrix("w_q", self.w_q.to_jagged());
        map.insert_matrix("w_k", self.w_k.to_jagged());
        map.insert_matrix("w_v", self.w_v.to_jagged());
        map.insert_matrix("w_o", self.w_o.to_jagged());
        map
    }

    fn from_weight_map(&mut self, map: &WeightMap) -> Result<()> {
        let n_heads = map.get_scalar("n_heads")? as usize;
        let d_model = map.get_scalar("d_model")? as usize;
        let d_head = map.get_scalar("d_head")? as usize;
        if n_heads != self.n_heads || d_model != self.d_model || d_head != self.d_head {
            return Err(Error::new(ErrorKind::InvalidData, "mha config mismatch"));
        }
        self.w_q = Matrix::from_jagged(map.get_matrix("w_q")?);
        self.w_k = Matrix::from_jagged(map.get_matrix("w_k")?);
        self.w_v = Matrix::from_jagged(map.get_matrix("w_v")?);
        self.w_o = Matrix::from_jagged(map.get_matrix("w_o")?);
        Ok(())
    }
}
