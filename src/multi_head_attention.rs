use std::io::Error;
use std::io::ErrorKind;
use std::io::Result;

use rand::RngExt;
use rand::rng;
use rayon::prelude::*;

use crate::adam_w::AdamW;
use crate::checkpoint::Checkpointable;
use crate::checkpoint::WeightMap;
use crate::kv_cache::KvCache;
use crate::matrix::Matrix;
use crate::rope::Rope;

pub struct MultiHeadAttention {
    /// Q/K/V を 1 つに融合した重み: (d_model, 3*d_model)。 列方向に [Q | K | V] と並ぶ。
    /// Phase 7 高速化で導入: matmul を 3 回 → 1 回に削減 (BLAS 効率も向上)。
    /// チェックポイントでは互換性のため `w_q`, `w_k`, `w_v` の 3 つに分割して保存する。
    w_qkv: Matrix,
    w_o: Matrix,

    grad_w_qkv: Matrix,
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
            w_qkv: rand_matrix(d_model, 3 * d_model),
            w_o: rand_matrix(d_model, d_model),
            grad_w_qkv: Matrix::zeros(d_model, 3 * d_model),
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

    /// W_QKV から W_Q (列 0..d_model) のビューに相当する Matrix を切り出す。
    /// チェックポイント保存・KV cache の per-token forward で利用。
    fn slice_w_q(&self) -> Matrix {
        let cols = self.w_qkv.cols();
        let mut data = Vec::with_capacity(self.d_model * self.d_model);
        for i in 0..self.d_model {
            data.extend_from_slice(&self.w_qkv.data()[i * cols..i * cols + self.d_model]);
        }
        Matrix::from_flat(data, self.d_model, self.d_model)
    }
    fn slice_w_k(&self) -> Matrix {
        let cols = self.w_qkv.cols();
        let mut data = Vec::with_capacity(self.d_model * self.d_model);
        for i in 0..self.d_model {
            let start = i * cols + self.d_model;
            data.extend_from_slice(&self.w_qkv.data()[start..start + self.d_model]);
        }
        Matrix::from_flat(data, self.d_model, self.d_model)
    }
    fn slice_w_v(&self) -> Matrix {
        let cols = self.w_qkv.cols();
        let mut data = Vec::with_capacity(self.d_model * self.d_model);
        for i in 0..self.d_model {
            let start = i * cols + 2 * self.d_model;
            data.extend_from_slice(&self.w_qkv.data()[start..start + self.d_model]);
        }
        Matrix::from_flat(data, self.d_model, self.d_model)
    }

    /// 旧形式 (W_Q | W_K | W_V) から W_QKV を再構成する。 checkpoint ロード時に使う。
    fn assemble_w_qkv(&mut self, q: &Matrix, k: &Matrix, v: &Matrix) {
        let d = self.d_model;
        let mut w = Matrix::zeros(d, 3 * d);
        for i in 0..d {
            let dst = w.row_mut(i);
            dst[..d].copy_from_slice(&q.row(i)[..d]);
            dst[d..2 * d].copy_from_slice(&k.row(i)[..d]);
            dst[2 * d..3 * d].copy_from_slice(&v.row(i)[..d]);
        }
        self.w_qkv = w;
    }

    pub fn forward(&mut self, x: &Matrix, mask: Option<&Vec<Vec<bool>>>) -> Matrix {
        // Q, K, V を **1 つの matmul に融合**: x @ W_QKV → split。
        // 旧 3 matmul (x @ W_Q, x @ W_K, x @ W_V) より BLAS 効率が良い。
        let qkv = x.matmul(&self.w_qkv);
        // Phase 7-4: split_columns の戻り Vec を IntoIter で move して .clone() 3 回 (3 × 2 MB)
        // を排除。
        let mut parts_iter = qkv.split_columns(3).into_iter();
        let q = parts_iter.next().unwrap();
        let k = parts_iter.next().unwrap();
        let v = parts_iter.next().unwrap();

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

        self.cache_x = x.clone();
        self.cache_q = q;
        self.cache_k = k;
        self.cache_v = v;
        self.cache_concat = concat;
        self.cache_att_w = all_weights;
        self.cache_v_head = v_heads;

        output
    }

    pub fn backward(&mut self, dl_dout: &Matrix) -> Matrix {
        // W_O backward
        // grad_w_o += concat^T @ dl_dout  (Phase 7-4: fused matmul-add で temp alloc + sweep 圧縮)
        self.cache_concat
            .matmul_t1_add_into(dl_dout, &mut self.grad_w_o);
        // dl_dconcat = dl_dout @ W_O^T  (Phase 7-3: matmul_t2 で transpose アロケを回避)
        let dl_dconcat = dl_dout.matmul_t2(&self.w_o);

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

        // QKV 融合 backward: dQ/dK/dV を列方向に concat → 1 matmul で W_QKV と x の勾配を作る。
        let dl_dqkv = Matrix::concat_columns(&[dl_dq, dl_dk, dl_dv]);

        // grad_w_qkv += cache_x^T @ dl_dqkv  shape: (d_model, 3*d_model)
        // Phase 7-4: fused matmul-add で d_model × 3*d_model の temp + sweep を 1 sgemm に圧縮。
        self.cache_x
            .matmul_t1_add_into(&dl_dqkv, &mut self.grad_w_qkv);

        // dl_dx = dl_dqkv @ W_QKV^T  shape: (seq, d_model)
        // Phase 7-3: matmul_t2 で W_QKV (d, 3d) の transpose を materialize しない (省 6 MB)
        dl_dqkv.matmul_t2(&self.w_qkv)
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

        // 1 token を 1-row Matrix にして Q, K, V を **1 つの matmul で射影**。
        // x @ W_QKV: (1, d) × (d, 3d) → (1, 3d)
        let x_m = Matrix::from_flat(x_new.to_vec(), 1, self.d_model);
        let qkv = x_m.matmul(&self.w_qkv);
        let d = self.d_model;
        let mut q_data: Vec<f32> = qkv.data()[0..d].to_vec();
        let mut k_data: Vec<f32> = qkv.data()[d..2 * d].to_vec();
        let v_owned: Vec<f32> = qkv.data()[2 * d..3 * d].to_vec();
        let v_data: &[f32] = &v_owned;

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
        self.grad_w_qkv.data_mut().fill(0.0);
        self.grad_w_o.data_mut().fill(0.0);
    }

    pub fn apply_gradients(&mut self, opt: &mut AdamW, prefix: &str) {
        // 重みの命名は **互換性のため w_q/w_k/w_v** に分割した形に保つ。
        // (Adam の moments を含む checkpoint 形式が変わらない)
        // 内部 W_QKV は (d_model, 3*d_model) のまま、 列 0..d, d..2d, 2d..3d を
        // 仮想的な W_Q/W_K/W_V として個別更新する。 まずは融合のままステップして
        // (シンプル & 1 step) 後で必要なら 3 分割に切り分ける。
        opt.step_matrix_flat(
            &format!("{prefix}.w_qkv"),
            &mut self.w_qkv,
            &self.grad_w_qkv,
        );
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
    let inv_scale = 1.0 / d_k.sqrt();

    // dl_dv = P^T @ dl_dout  [seq, d_head]  (Phase 7-3: matmul_t1)
    let dl_dv = att_w.matmul_t1(dl_dout);

    // dl_dS = scaled softmax-backward(P, dl_dP) を **dl_dP の buffer に in-place** で
    // 計算する (Phase 7-4)。 旧来は dl_dP (seq×seq=4 MB) と dl_dS (4 MB) を別々に確保
    // していたが、 dl_dS[i][j] は dp_row[j] のみに依存するため (dot は j 全体の集約で
    // 1 行の書き換え前に確定する)、 同じ buffer に書き戻して問題ない。
    // /√d_k スケーリングも同じループに融合 (旧来の 2 度目の sweep を削減)。
    let mut dl_ds = dl_dout.matmul_t2(v); // 元 dl_dP, in-place で dl_dS に上書き
    let cols = dl_ds.cols();
    dl_ds
        .data_mut()
        .par_chunks_mut(cols)
        .enumerate()
        .for_each(|(i, ds_row)| {
            let p_row = att_w.row(i);
            let dot: f32 = p_row
                .iter()
                .zip(ds_row.iter())
                .map(|(p, dp)| p * dp)
                .sum();
            for j in 0..cols {
                ds_row[j] = p_row[j] * (ds_row[j] - dot) * inv_scale;
            }
        });

    // dl_dq = dl_dS @ K  [seq, d_head]
    let dl_dq = dl_ds.matmul(k);
    // dl_dk = dl_dS^T @ Q  [seq, d_head]  (Phase 7-3: matmul_t1)
    let dl_dk = dl_ds.matmul_t1(q);

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

    // QK^T / √d_k  (Phase 7-3: matmul_t2 で K の transpose を materialize しない)
    let mut scores = q.matmul_t2(k);
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
    // Phase 7-4: 旧コードでは softmax 後に scores.clone() で attention weights を確保していたが、
    // matmul は &self しか借りないので、 そのまま `scores` を返せば clone を省略できる
    // (4 MB × n_heads × n_layers ぶんの per-step アロケを削減)。
    let output = scores.matmul(v);
    (output, scores)
}

pub fn causal_mask(seq_len: usize) -> Vec<Vec<bool>> {
    (0..seq_len)
        .map(|i| (0..seq_len).map(|j| j > i).collect())
        .collect()
}

impl Checkpointable for MultiHeadAttention {
    fn to_weight_map(&self) -> WeightMap {
        // 旧形式互換のため w_q/w_k/w_v に分割して保存する。
        // (Phase 6-c までの best.bin が w_q/w_k/w_v を期待しているため、
        //  Phase 7 移行後も load 時の互換性を担保する)
        let mut map = WeightMap::new();
        map.insert_scalar("n_heads", self.n_heads as u64);
        map.insert_scalar("d_model", self.d_model as u64);
        map.insert_scalar("d_head", self.d_head as u64);
        map.insert_matrix("w_q", self.slice_w_q().to_jagged());
        map.insert_matrix("w_k", self.slice_w_k().to_jagged());
        map.insert_matrix("w_v", self.slice_w_v().to_jagged());
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
        // 旧形式 (w_q, w_k, w_v) → W_QKV に組み立てる
        let q = Matrix::from_jagged(map.get_matrix("w_q")?);
        let k = Matrix::from_jagged(map.get_matrix("w_k")?);
        let v = Matrix::from_jagged(map.get_matrix("w_v")?);
        self.assemble_w_qkv(&q, &k, &v);
        self.w_o = Matrix::from_jagged(map.get_matrix("w_o")?);
        Ok(())
    }
}
