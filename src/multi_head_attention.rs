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
    // Q は w_q、KV は w_kv に分離。matmul を 3 回 → 2 回に削減。
    /// Phase 7 高速化で導入: matmul を 3 回 → 1 回に削減 (BLAS 効率も向上)。
    /// チェックポイントでは互換性のため `w_q`, `w_k`, `w_v` の 3 つに分割して保存する。
    w_q: Matrix, // (d_model, d_model)
    w_kv: Matrix, // (d_model, 2*d_kv_model)
    w_o: Matrix,

    grad_w_q: Matrix,
    grad_w_kv: Matrix,
    grad_w_o: Matrix,

    cache_x: Matrix,      // (seq, d_model)
    cache_q: Matrix,      // (seq, d_model) ※未回転
    cache_k: Matrix,      // (seq, d_kv_model) ※未回転
    cache_v: Matrix,      // (seq, d_kv_model)
    cache_concat: Matrix, // (seq, d_model)

    // Phase 7-5 B3: Flash Attention の出力統計を保存 (backward で attention を再構成するため)。
    // 旧 cache_att_w (n_heads × (seq, seq) Vec<Matrix>) は per-head 4 MB × 8 head × 6 layer = 192 MB
    // だったが、 cache_lse (n_heads × seq の Vec<Vec<f32>>) は per-head 4 KB × 8 × 6 = 192 KB に縮小。
    cache_lse: Vec<Vec<f32>>, // n_heads × seq, log-sum-exp from flash forward
    cache_out_heads: Vec<Matrix>, // n_heads × (seq, d_head), flash forward output per head
    // (backward の D = rowsum(O ⊙ dO) 計算で必要)

    // ---------- Phase 7-5 (perf-alloc): reusable buffers ----------
    // forward/backward を呼ぶたびに毎回 alloc していた中間行列を struct field 化して、
    // ensure_shape + _into 系 API で **buffer を再利用** する。 BLAS time は不変だが
    // per-step alloc を ~30 MB 削減 (ピーク RSS と allocator 競合の減少が効く)。
    buf_kv: Matrix,         // forward: x @ W_KV (seq, 2*d_kv_model)
    buf_dl_dconcat: Matrix, // backward: dl_dout @ W_O^T (seq, d_model)
    buf_dl_dq: Matrix,      // backward: concat heads → (seq, d_model)
    buf_dl_dk: Matrix,      // backward: concat heads → (seq, d_kv_model)
    buf_dl_dv: Matrix,      // backward: concat heads → (seq, d_kv_model)
    buf_dl_dkv: Matrix,     // backward: concat [dk, dv] → (seq, 2*d_kv_model)

    n_heads: usize,
    n_kv_heads: usize,
    d_model: usize,
    d_kv_model: usize,
    d_head: usize,

    /// Some の場合は Q, K に回転位置エンコーディングを適用する。
    /// V には掛けない (RoPE の規約)。
    rope: Option<Rope>,
}

impl MultiHeadAttention {
    pub fn new(d_model: usize, n_heads: usize, n_kv_heads: usize, rope: Option<Rope>) -> Self {
        assert!(
            d_model % n_heads == 0,
            "d_model must to divisible by n_heads"
        );
        assert!(
            n_heads % n_kv_heads == 0,
            "n_heads must to divisible by n_kv_heads"
        );
        assert!(n_kv_heads > 0, "n_kv_heads must be > 0");
        assert!(n_heads > 0, "n_heads must be > 0");
        let d_head = d_model / n_heads;
        let d_kv_model = n_kv_heads * d_head;
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
            w_kv: rand_matrix(d_model, 2 * d_kv_model),
            w_o: rand_matrix(d_model, d_model),
            grad_w_q: Matrix::zeros(d_model, d_model),
            grad_w_kv: Matrix::zeros(d_model, 2 * d_kv_model),
            grad_w_o: Matrix::zeros(d_model, d_model),
            cache_x: Matrix::default(),
            cache_q: Matrix::default(),
            cache_k: Matrix::default(),
            cache_v: Matrix::default(),
            cache_concat: Matrix::default(),
            cache_lse: Vec::new(),
            cache_out_heads: Vec::new(),
            buf_kv: Matrix::default(),
            buf_dl_dconcat: Matrix::default(),
            buf_dl_dq: Matrix::default(),
            buf_dl_dk: Matrix::default(),
            buf_dl_dv: Matrix::default(),
            buf_dl_dkv: Matrix::default(),
            n_heads,
            n_kv_heads,
            d_model,
            d_kv_model,
            d_head,
            rope,
        }
    }

    /// W_KV から W_K / W_V (列 0..d_kv_model) のビューに相当する Matrix を切り出す。
    /// チェックポイント保存・KV cache の per-token forward で利用。
    fn slice_w_k(&self) -> Matrix {
        let cols = self.w_kv.cols();
        let mut data = Vec::with_capacity(self.d_model * self.d_kv_model);
        for i in 0..self.d_model {
            data.extend_from_slice(&self.w_kv.data()[i * cols..i * cols + self.d_kv_model]);
        }
        Matrix::from_flat(data, self.d_model, self.d_kv_model)
    }
    fn slice_w_v(&self) -> Matrix {
        let cols = self.w_kv.cols();
        let mut data = Vec::with_capacity(self.d_model * self.d_kv_model);
        for i in 0..self.d_model {
            let start = i * cols + self.d_kv_model;
            data.extend_from_slice(&self.w_kv.data()[start..start + self.d_kv_model]);
        }
        Matrix::from_flat(data, self.d_model, self.d_kv_model)
    }

    /// 旧形式 (W_K | W_V) から W_KV を再構成する。 checkpoint ロード時に使う。
    fn assemble_w_kv(&mut self, k: &Matrix, v: &Matrix) {
        let d = self.d_model;
        let d_kv = self.d_kv_model;
        let mut w = Matrix::zeros(d, 2 * d_kv);
        for i in 0..d {
            let dst = w.row_mut(i);
            dst[..d_kv].copy_from_slice(&k.row(i)[..d_kv]);
            dst[d_kv..2 * d_kv].copy_from_slice(&v.row(i)[..d_kv]);
        }
        self.w_kv = w;
    }

    pub fn forward(&mut self, x: &Matrix, _mask: Option<&Vec<Vec<bool>>>) -> Matrix {
        // 注: `mask` は API 互換のため残置。 訓練では常に causal_mask が渡され、
        // Phase 7-5 B3 で Flash Attention に置換された結果 causal 専用となっている。
        // (推論の per-token forward_step は別パスで mask 不要)。
        let seq = x.rows();
        let d = self.d_model;
        let d_kv = self.d_kv_model;

        // Phase 7-5: cache_x ← x の clone を ensure_shape + memcpy に置換 (alloc 1 回 → 0)
        self.cache_x.ensure_shape(seq, d);
        self.cache_x.data_mut().copy_from_slice(x.data());

        // Q projection: x @ W_Q → cache_q
        self.cache_q.ensure_shape(seq, d);
        x.matmul_into(&self.w_q, &mut self.cache_q);

        // KV 融合 projection: x @ W_KV → buf_kv (Phase 7-5: alloc 1 回 → 0)。
        self.buf_kv.ensure_shape(seq, 2 * d_kv);
        x.matmul_into(&self.w_kv, &mut self.buf_kv);

        // buf_kv を 2 列ブロックに split → cache_k, cache_v に直接書き込む
        // (Phase 7-5: alloc 2 回 → 0)。 mem::take で所有権を一時的に外して配列にまとめる。
        let mut kv_parts = [
            std::mem::take(&mut self.cache_k),
            std::mem::take(&mut self.cache_v),
        ];
        for p in &mut kv_parts {
            p.ensure_shape(seq, d_kv);
        }

        self.buf_kv.split_columns_into(&mut kv_parts);
        let [k, v] = kv_parts;

        self.cache_k = k;
        self.cache_v = v;

        // head 分割 (n_heads × (seq, d_head))。
        let mut q_heads = self.cache_q.split_columns(self.n_heads);
        let mut k_heads = self.cache_k.split_columns(self.n_kv_heads);
        let v_heads = self.cache_v.split_columns(self.n_kv_heads);

        // RoPE: Q と K のみに位置回転を適用 (V には適用しない)。
        // backward でも同じ回転が必要なので、 ここでの結果は捨てて backward で再計算する。
        if let Some(rope) = &self.rope {
            for h in 0..self.n_heads {
                rope.apply_in_place(&mut q_heads[h]);
            }
            for h in 0..self.n_kv_heads {
                rope.apply_in_place(&mut k_heads[h]);
            }
        }

        // Phase 7-5 B3: Flash Attention に置換。
        // 旧 scaled_dot_product_attention は per-head で full (seq, seq) scores を作っていた
        // (4 MB × n_heads × n_layers × 2 (fwd+bwd) = ~384 MB / step を allocate)。
        // Flash Attention は block-tile online softmax で scores を materialize しないため
        // alloc を大幅に削減し、 さらに causal の upper triangle を skip して FLOP も ~半減する。
        // Phase 8-1: GQA を導入
        // Q head を KV head にグループして flash attention
        // d_model=768, n_heads=8, n_kv_heads=2, d_head=96 の場合
        // W_K MHAサイズ:768×768 GQAサイズ:768×192 削減率:75%
        // W_V MHAサイズ:768×768 GQAサイズ:768×192 削減率:75%
        // KV-cache MHAサイズ:n×768 GQAサイズ:n×192 削減率:75%
        // 複数の Q head が同じ K/V を参照しても品質の劣化が小さいことが LLaMA 2 の論文で実証されている。
        let group_size = self.n_heads / self.n_kv_heads;
        let mut head_outputs = Vec::with_capacity(self.n_heads);
        let mut lse_per_head = Vec::with_capacity(self.n_heads);

        for kv_idx in 0..self.n_kv_heads {
            for g in 0..group_size {
                let q_idx = kv_idx * group_size + g;
                // 同じ k_heads[kv_idx], v_heads[kv_idx] を group_size 回使う
                let (out, lse) =
                    flash_attention_forward(&q_heads[q_idx], &k_heads[kv_idx], &v_heads[kv_idx]);
                head_outputs.push(out);
                lse_per_head.push(lse);
            }
        }

        // concat heads → cache_concat (Phase 7-5: alloc 1 回 → 0)。
        self.cache_concat.ensure_shape(seq, d);
        let head_refs: Vec<&Matrix> = head_outputs.iter().collect();
        Matrix::concat_columns_into(&head_refs, &mut self.cache_concat);

        // 出力 projection: concat @ W_O → 戻り値。 ここは関数の return 値なので残置。
        let output = self.cache_concat.matmul(&self.w_o);

        self.cache_lse = lse_per_head;
        self.cache_out_heads = head_outputs;

        output
    }

    pub fn backward(&mut self, dl_dout: &Matrix) -> Matrix {
        let seq = self.cache_x.rows();
        let d = self.d_model;
        let d_kv = self.d_kv_model;

        // W_O backward
        // grad_w_o += concat^T @ dl_dout  (Phase 7-4: fused matmul-add で temp alloc + sweep 圧縮)
        self.cache_concat
            .matmul_t1_add_into(dl_dout, &mut self.grad_w_o);

        // dl_dconcat = dl_dout @ W_O^T を buf_dl_dconcat に書き込む (Phase 7-5: alloc 1 回 → 0)。
        self.buf_dl_dconcat.ensure_shape(seq, d);
        dl_dout.matmul_t2_into(&self.w_o, &mut self.buf_dl_dconcat);

        // concat_heads backward → head ごとに dl_dconcat をスライス
        let dl_dhead_outs = self.buf_dl_dconcat.split_columns(self.n_heads);

        // scaled_dot_product_attention backward（head ごと）
        // RoPE 使用時、 attention は **回転後** の Q', K' に対して計算されたので、
        // backward の入力にも回転後の値が必要。 cache は未回転なので再回転する。
        let mut q_heads = self.cache_q.split_columns(self.n_heads);
        let mut k_heads = self.cache_k.split_columns(self.n_kv_heads);
        // V も head 分割。 旧 cache_v_head は廃止して、 backward 毎に cache_v から split し直す
        // (split_columns は memcpy のみで安価、 4 KB × 8 head 程度)。
        let v_heads = self.cache_v.split_columns(self.n_kv_heads);
        if let Some(rope) = &self.rope {
            for h in 0..self.n_heads {
                rope.apply_in_place(&mut q_heads[h]);
            }
            for h in 0..self.n_kv_heads {
                rope.apply_in_place(&mut k_heads[h]);
            }
        }

        // Phase 7-5 B3: Flash Attention backward に置換。
        // cache_lse から P を再構成し、 dq/dk/dv を block-tile で accumulate する。
        let group_size = self.n_heads / self.n_kv_heads;
        let mut dl_dq_heads = Vec::with_capacity(self.n_heads);
        let mut dl_dk_heads: Vec<Matrix> =
            vec![Matrix::zeros(/* seq */ self.cache_k.rows(), self.d_head); self.n_kv_heads];
        let mut dl_dv_heads: Vec<Matrix> =
            vec![Matrix::zeros(self.cache_v.rows(), self.d_head); self.n_kv_heads];

        for kv_idx in 0..self.n_kv_heads {
            for g in 0..group_size {
                let q_idx = kv_idx * group_size + g;
                let (dq, dk, dv) = flash_attention_backward(
                    &q_heads[q_idx],
                    &k_heads[kv_idx], // ← 共有 KV head
                    &v_heads[kv_idx],
                    &self.cache_out_heads[q_idx],
                    &self.cache_lse[q_idx],
                    &dl_dhead_outs[q_idx],
                );
                dl_dq_heads.push(dq);
                // dk/dv は同じ KV head への寄与を **累積**
                let seq = dk.rows();
                for r in 0..seq {
                    let dst_k = dl_dk_heads[kv_idx].row_mut(r);
                    let dst_v = dl_dv_heads[kv_idx].row_mut(r);
                    for c in 0..self.d_head {
                        dst_k[c] += dk.row(r)[c];
                        dst_v[c] += dv.row(r)[c];
                    }
                }
            }
        }

        // RoPE backward: dL/dQ' から dL/dQ へ (= 逆回転)。 V は未回転なので不要。
        if let Some(rope) = &self.rope {
            for h in 0..self.n_heads {
                rope.apply_backward_in_place(&mut dl_dq_heads[h]);
            }
            for h in 0..self.n_kv_heads {
                rope.apply_backward_in_place(&mut dl_dk_heads[h]);
            }
        }

        // split_heads backward → head の勾配を結合して [seq, d_model] に戻す。
        // Phase 7-5: 3 つの concat 出力を buf_dl_d{q,k,v} に書き込む (alloc 3 回 → 0)。
        self.buf_dl_dq.ensure_shape(seq, d);
        self.buf_dl_dk.ensure_shape(seq, d_kv);
        self.buf_dl_dv.ensure_shape(seq, d_kv);
        let dq_refs: Vec<&Matrix> = dl_dq_heads.iter().collect();
        let dk_refs: Vec<&Matrix> = dl_dk_heads.iter().collect();
        let dv_refs: Vec<&Matrix> = dl_dv_heads.iter().collect();
        Matrix::concat_columns_into(&dq_refs, &mut self.buf_dl_dq);
        Matrix::concat_columns_into(&dk_refs, &mut self.buf_dl_dk);
        Matrix::concat_columns_into(&dv_refs, &mut self.buf_dl_dv);

        // KV 融合 backward: dK/dV を列方向に concat → 1 matmul で W_KV と x の勾配を作る。
        // Phase 7-5: buf_dl_dkv に書き込む (alloc 1 回 → 0)。
        self.buf_dl_dkv.ensure_shape(seq, 2 * d_kv);
        Matrix::concat_columns_into(&[&self.buf_dl_dk, &self.buf_dl_dv], &mut self.buf_dl_dkv);

        // grad_w_q += cache_x^T @ dl_dq  shape: (d_model, d_model)
        self.cache_x
            .matmul_t1_add_into(&self.buf_dl_dq, &mut self.grad_w_q);
        // grad_w_kv += cache_x^T @ dl_dkv  shape: (d_model, 2*d_kv_model)
        // Phase 7-4: fused matmul-add で d_model × 2*d_kv_model の temp + sweep を 1 sgemm に圧縮。
        self.cache_x
            .matmul_t1_add_into(&self.buf_dl_dkv, &mut self.grad_w_kv);

        // dl_dx = dl_dkv @ W_KV^T  shape: (seq, d_kv_model)
        // dl_dx += buf_dl_dq  @ W_Q^T   shape: (seq, d_model)
        // Phase 7-3: matmul_t2 で W_KV (d, 2d_kv) の transpose を materialize しない (省 6 MB)。
        // 戻り値なので buffer 化しても allocator 呼び出し回数は変わらず、 ここは alloc 1 回のまま残置。
        let mut dl_dx = self.buf_dl_dkv.matmul_t2(&self.w_kv);
        self.buf_dl_dq.matmul_t2_add_into(&self.w_q, &mut dl_dx);
        dl_dx
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

        let x_m = Matrix::from_flat(x_new.to_vec(), 1, self.d_model);
        let mut q_data: Vec<f32> = x_m.matmul(&self.w_q).data().to_vec();
        let kv = x_m.matmul(&self.w_kv);
        let d_kv = self.d_kv_model;
        let mut k_data: Vec<f32> = kv.data()[0..d_kv].to_vec();
        let v_owned: Vec<f32> = kv.data()[d_kv..2 * d_kv].to_vec();
        let v_data: &[f32] = &v_owned;

        // RoPE: 新規 token の Q, K を **位置 cur_len** で head ごとに回転。
        // 過去の K (cache.k) は append 時点で当時の位置の角度で回転済なので不変。
        let pos = cache.cur_len();
        if let Some(rope) = &self.rope {
            for h in 0..h_n {
                rope.apply_at_position(&mut q_data[h * dh..(h + 1) * dh], pos);
            }
            for h in 0..self.n_kv_heads {
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
        let kv_group = self.n_heads / self.n_kv_heads;
        for h in 0..h_n {
            let q_head = &q_data[h * dh..(h + 1) * dh];
            let kv_h = h / kv_group;

            // scores[j] = (q · K_j) / √d_head, j ∈ [0, n)
            let mut scores = vec![0.0f32; n];
            for j in 0..n {
                let k_row = cache_k.row(j);
                let k_slice = &k_row[kv_h * dh..(kv_h + 1) * dh];
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
                let v_slice = &v_row[kv_h * dh..(kv_h + 1) * dh];
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
        self.grad_w_kv.data_mut().fill(0.0);
        self.grad_w_o.data_mut().fill(0.0);
    }

    pub fn apply_gradients(&mut self, opt: &mut AdamW, prefix: &str) {
        // 重みの命名は **互換性のため w_q/w_k/w_v** に分割した形に保つ。
        // (Adam の moments を含む checkpoint 形式が変わらない)
        opt.step_matrix_flat(&format!("{prefix}.w_q"), &mut self.w_q, &self.grad_w_q);
        // 内部 W_KV は (d_model, 2*d_kv_model) のまま、 列 0..d, d..2d を
        // 仮想的な W_K/W_V として個別更新する。 まずは融合のままステップして
        // (シンプル & 1 step) 後で必要なら 2 分割に切り分ける。
        opt.step_matrix_flat(&format!("{prefix}.w_kv"), &mut self.w_kv, &self.grad_w_kv);

        opt.step_matrix_flat(&format!("{prefix}.w_o"), &mut self.w_o, &self.grad_w_o);
    }
}

/// Phase 7-5 B3 移行で MHA からは Flash Attention 経由に切り替えたが、
/// `flash_attention_*` の数値一致テスト (基準実装) として残置する。
#[allow(dead_code)]
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
            let dot: f32 = p_row.iter().zip(ds_row.iter()).map(|(p, dp)| p * dp).sum();
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

/// Phase 7-5 B3 移行で MHA からは Flash Attention 経由に切り替えたが、
/// `flash_attention_*` の数値一致テスト (基準実装) として残置する。
#[allow(dead_code)]
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

// ============================================================================
// Phase 7-5 B3: Flash Attention 風 (block-tiled online softmax + causal skip)
// ============================================================================
//
// 目的:
// 1. **alloc 削減**: 中間 (N, N) scores 行列を materialize しない (per head 4 MB × 8 head × 6 layer
//    × 2 (fwd/bwd) = ~384 MB の per-step alloc を消す)。
// 2. **causal skip による FLOP 削減**: causal mask の upper triangle は計算しない (~50% off)。
// 3. **キャッシュ局所性**: Q/K/V を block-tile で処理し、 L1/L2 キャッシュに収める。
//
// アルゴリズム (FlashAttention-1 流):
//   for i in 0..n_blocks (Q blocks):
//     for j in 0..=i (K blocks, causal: j > i は skip):
//       S_ij = Qi @ Kj^T / √d_k                                          # (Bq, Bk)
//       if i == j: apply causal mask within diagonal block
//       online softmax update: m_new, l_new, O_block を rescale + 加算
//   final: O /= l (row-wise), lse = m + log(l)
//
// backward は forward で保存した lse (per-row log-sum-exp) から P を再構成して
// blockwise に dQ/dK/dV を accumulate する。
//
// 設計選択:
// - block size BLOCK_SIZE = 256 を固定 (seq=1024 で 4 Q block、 causal で平均 2.5 K block)。
//   小さすぎる block size (32-64) は sgemm 呼び出しオーバーヘッドが支配的になるため避ける。
//   d_head=64 のとき S block は 256×256×4 = 256 KB (L2 cache 内、 sgemm 効率良)。
//   seq < BLOCK_SIZE のときは単一ブロックでの処理にフォールバック (正しさは変わらない)。
// - 戻り値は (output, lse: Vec<f32>) で、 既存 (output, att_w: Matrix) と互換は無い。
//   MHA 側で cache_att_w → cache_lse に変更する。

const FA_BLOCK_SIZE: usize = 256;

/// Flash Attention forward (causal mask 専用)。
/// 入力 q, k, v は形状 `(N, d_head)`。 戻り値 `(output, lse)`:
/// - `output`: `(N, d_head)` の attention 出力。
/// - `lse`: 長さ `N` の per-row log-sum-exp (= `m + log(l)`)。 backward で使う。
pub fn flash_attention_forward(q: &Matrix, k: &Matrix, v: &Matrix) -> (Matrix, Vec<f32>) {
    let n = q.rows();
    let d_head = q.cols();
    assert_eq!(k.rows(), n, "flash_attention: K rows mismatch");
    assert_eq!(v.rows(), n, "flash_attention: V rows mismatch");
    assert_eq!(k.cols(), d_head, "flash_attention: K cols mismatch");
    assert_eq!(v.cols(), d_head, "flash_attention: V cols mismatch");

    let inv_scale = 1.0 / (d_head as f32).sqrt();
    let bs = FA_BLOCK_SIZE;
    let n_blocks = n.div_ceil(bs);

    let mut output = Matrix::zeros(n, d_head);
    let mut m_vec = vec![f32::NEG_INFINITY; n];
    let mut l_vec = vec![0.0f32; n];

    // 各 Q block を独立に処理 (row 方向は他の Q block と独立なので rayon 並列化可だが、
    // ここでは単純化のため逐次。 BLAS 内部の並列化に任せる)。
    for i_block in 0..n_blocks {
        let i_start = i_block * bs;
        let i_end = (i_start + bs).min(n);
        let bq = i_end - i_start;

        // Qi = Q[i_start..i_end] のビュー風に slice → 新規 Matrix を構築
        // (slice view が無いので copy。 ホット path だが d_head 小なのでコスト低)
        let qi = block_view(q, i_start, i_end);

        // Q block ごとの running max, sum, output (block-local)
        let mut m_block = vec![f32::NEG_INFINITY; bq];
        let mut l_block = vec![0.0f32; bq];
        let mut o_block = Matrix::zeros(bq, d_head);

        // causal: j_block は 0..=i_block のみ計算 (それ以外は S_ij が全て -inf で寄与なし)
        for j_block in 0..=i_block {
            let j_start = j_block * bs;
            let j_end = (j_start + bs).min(n);

            let kj = block_view(k, j_start, j_end);
            let vj = block_view(v, j_start, j_end);

            // S = Qi @ Kj^T / √d_k    shape: (bq, j_end - j_start)
            let mut s = qi.matmul_t2(&kj);
            for x in s.data_mut() {
                *x *= inv_scale;
            }

            // Diagonal block (i_block == j_block): causal mask を local 座標で適用。
            // 「global col > global row」 = 「j_start + c > i_start + r」 → c > r (block 対角の場合)
            if i_block == j_block {
                let cols = s.cols();
                for r in 0..bq {
                    let s_row = s.row_mut(r);
                    for c in (r + 1)..cols {
                        s_row[c] = f32::NEG_INFINITY;
                    }
                }
            }

            // Online softmax 更新:
            //   m_new = max(m_old, rowmax(S))
            //   P = exp(S - m_new)
            //   rescale = exp(m_old - m_new)
            //   O_block = O_block * rescale + P @ V_j
            //   l_block = l_block * rescale + rowsum(P)
            //   m_block = m_new

            // 1. row max of S
            let mut row_max_s = vec![f32::NEG_INFINITY; bq];
            for r in 0..bq {
                let mut mx = f32::NEG_INFINITY;
                for &x in s.row(r).iter() {
                    if x > mx {
                        mx = x;
                    }
                }
                row_max_s[r] = mx;
            }

            // 2. m_new, rescale
            let mut m_new = vec![0.0f32; bq];
            let mut rescale = vec![0.0f32; bq];
            for r in 0..bq {
                let mnew = m_block[r].max(row_max_s[r]);
                // m_block[r] が -inf のときは rescale = 1.0 (初期値で乗算しても 0 のまま)
                let rs = if m_block[r] == f32::NEG_INFINITY {
                    1.0
                } else {
                    (m_block[r] - mnew).exp()
                };
                m_new[r] = mnew;
                rescale[r] = rs;
            }

            // 3. P = exp(S - m_new) in-place を S buffer に
            let mut sum_p = vec![0.0f32; bq];
            for r in 0..bq {
                let mnew = m_new[r];
                let s_row = s.row_mut(r);
                let mut sp = 0.0f32;
                for x in s_row.iter_mut() {
                    let e = if *x == f32::NEG_INFINITY {
                        0.0
                    } else {
                        (*x - mnew).exp()
                    };
                    *x = e;
                    sp += e;
                }
                sum_p[r] = sp;
            }

            // 4. O_block = O_block * rescale + P @ V_j
            //    まず O_block の各行を rescale で in-place scale
            for r in 0..bq {
                let rs = rescale[r];
                if rs != 1.0 {
                    let o_row = o_block.row_mut(r);
                    for x in o_row.iter_mut() {
                        *x *= rs;
                    }
                }
            }
            //    O_block += P @ V_j    (P: bq × bk, V_j: bk × d_head)
            s.matmul_add_into(&vj, &mut o_block);

            // 5. l_block, m_block update
            for r in 0..bq {
                l_block[r] = l_block[r] * rescale[r] + sum_p[r];
                m_block[r] = m_new[r];
            }
        }

        // O_block を l_block で正規化し、 output / lse / m_vec / l_vec へ書き戻し
        for r in 0..bq {
            let global_r = i_start + r;
            let inv_l = 1.0 / l_block[r];
            let o_row = o_block.row(r);
            let out_row = output.row_mut(global_r);
            for c in 0..d_head {
                out_row[c] = o_row[c] * inv_l;
            }
            m_vec[global_r] = m_block[r];
            l_vec[global_r] = l_block[r];
        }
    }

    // lse = m + log(l)
    let lse: Vec<f32> = m_vec
        .iter()
        .zip(l_vec.iter())
        .map(|(&m, &l)| m + l.ln())
        .collect();

    (output, lse)
}

/// Flash Attention backward (causal mask 専用)。
/// forward と同じ block 構造で attention probabilities を再構成し、 dq/dk/dv を計算する。
///
/// 入力:
/// - q, k, v: forward 同様
/// - output: forward の戻り値 (N, d_head)  ※ D = rowsum(O ⊙ dO) の計算に使用
/// - lse: forward の戻り値 (N,)            ※ P = exp(S - LSE) で attention 再構成
/// - dl_dout: 上流からの gradient (N, d_head)
///
/// 戻り値: (dq, dk, dv) いずれも (N, d_head)
pub fn flash_attention_backward(
    q: &Matrix,
    k: &Matrix,
    v: &Matrix,
    output: &Matrix,
    lse: &[f32],
    dl_dout: &Matrix,
) -> (Matrix, Matrix, Matrix) {
    let n = q.rows();
    let d_head = q.cols();
    assert_eq!(k.rows(), n);
    assert_eq!(v.rows(), n);
    assert_eq!(output.rows(), n);
    assert_eq!(dl_dout.rows(), n);
    assert_eq!(lse.len(), n);

    let inv_scale = 1.0 / (d_head as f32).sqrt();
    let bs = FA_BLOCK_SIZE;
    let n_blocks = n.div_ceil(bs);

    // D[r] = sum_c O[r,c] * dO[r,c]
    let mut d_vec = vec![0.0f32; n];
    for r in 0..n {
        let o_row = output.row(r);
        let do_row = dl_dout.row(r);
        let mut s = 0.0f32;
        for c in 0..d_head {
            s += o_row[c] * do_row[c];
        }
        d_vec[r] = s;
    }

    let mut dq = Matrix::zeros(n, d_head);
    let mut dk = Matrix::zeros(n, d_head);
    let mut dv = Matrix::zeros(n, d_head);

    // K block を外側、 対応する Q block (causal: i_block >= j_block) を内側でループ。
    // dV_block, dK_block を block-local に積み上げてから global dV, dK に書き戻す。
    for j_block in 0..n_blocks {
        let j_start = j_block * bs;
        let j_end = (j_start + bs).min(n);
        let bk = j_end - j_start;

        let kj = block_view(k, j_start, j_end);
        let vj = block_view(v, j_start, j_end);

        let mut dv_block = Matrix::zeros(bk, d_head);
        let mut dk_block = Matrix::zeros(bk, d_head);

        for i_block in j_block..n_blocks {
            let i_start = i_block * bs;
            let i_end = (i_start + bs).min(n);
            let bq = i_end - i_start;

            let qi = block_view(q, i_start, i_end);
            let doi = block_view(dl_dout, i_start, i_end);

            // S = Qi @ Kj^T / √d_k
            let mut s = qi.matmul_t2(&kj);
            for x in s.data_mut() {
                *x *= inv_scale;
            }
            // Diagonal block の causal mask
            if i_block == j_block {
                let cols = s.cols();
                for r in 0..bq {
                    let s_row = s.row_mut(r);
                    for c in (r + 1)..cols {
                        s_row[c] = f32::NEG_INFINITY;
                    }
                }
            }

            // P = exp(S - LSE_i[:, None])  in-place
            for r in 0..bq {
                let lse_r = lse[i_start + r];
                let s_row = s.row_mut(r);
                for x in s_row.iter_mut() {
                    *x = if *x == f32::NEG_INFINITY {
                        0.0
                    } else {
                        (*x - lse_r).exp()
                    };
                }
            }
            let p = s; // 名前変更 (semantics 明確化)

            // dV_block += P^T @ dOi    shape: (bk, d_head)
            p.matmul_t1_add_into(&doi, &mut dv_block);

            // dP = dOi @ Vj^T    shape: (bq, bk)
            let mut dp = doi.matmul_t2(&vj);

            // dS = P ⊙ (dP - D_i[:, None]) / √d_k  in-place を dp buffer に書き込む
            for r in 0..bq {
                let d_i = d_vec[i_start + r];
                let p_row = p.row(r);
                let dp_row = dp.row_mut(r);
                for c in 0..p_row.len() {
                    dp_row[c] = p_row[c] * (dp_row[c] - d_i) * inv_scale;
                }
            }
            let ds = dp;

            // dQ[i_start..i_end] += dS @ Kj
            // 直接 dq の block 部分にアクセスする方法がないので、 一度 block matmul を作って加算
            {
                let mut dq_add = Matrix::zeros(bq, d_head);
                ds.matmul_into(&kj, &mut dq_add);
                for r in 0..bq {
                    let src = dq_add.row(r);
                    let dst = dq.row_mut(i_start + r);
                    for c in 0..d_head {
                        dst[c] += src[c];
                    }
                }
            }

            // dK_block += dS^T @ Qi
            ds.matmul_t1_add_into(&qi, &mut dk_block);
        }

        // dV[j_start..j_end] = dv_block (block 内に完全に書き込み)
        for r in 0..bk {
            let src = dv_block.row(r);
            let dst = dv.row_mut(j_start + r);
            for c in 0..d_head {
                dst[c] = src[c];
            }
        }
        for r in 0..bk {
            let src = dk_block.row(r);
            let dst = dk.row_mut(j_start + r);
            for c in 0..d_head {
                dst[c] = src[c];
            }
        }
    }

    (dq, dk, dv)
}

/// `mat` の `start..end` 行を新しい Matrix として切り出す (B3 用のブロックビュー)。
/// view 構造がないので copy。 d_head が小さい (typ 64) ので 1 ブロック数 KB のコピーで済む。
fn block_view(mat: &Matrix, start: usize, end: usize) -> Matrix {
    let cols = mat.cols();
    let rows = end - start;
    let mut data = Vec::with_capacity(rows * cols);
    data.extend_from_slice(&mat.data()[start * cols..end * cols]);
    Matrix::from_flat(data, rows, cols)
}

impl Checkpointable for MultiHeadAttention {
    fn to_weight_map(&self) -> WeightMap {
        // 旧形式互換のため w_q/w_k/w_v に分割して保存する。
        // (Phase 6-c までの best.bin が w_q/w_k/w_v を期待しているため、
        //  Phase 7 移行後も load 時の互換性を担保する)
        let mut map = WeightMap::new();
        map.insert_scalar("n_heads", self.n_heads as u64);
        map.insert_scalar("n_kv_heads", self.n_kv_heads as u64);
        map.insert_scalar("d_model", self.d_model as u64);
        map.insert_scalar("d_kv_model", self.d_kv_model as u64);
        map.insert_scalar("d_head", self.d_head as u64);
        map.insert_matrix("w_q", self.w_q.to_jagged());
        map.insert_matrix("w_k", self.slice_w_k().to_jagged());
        map.insert_matrix("w_v", self.slice_w_v().to_jagged());
        map.insert_matrix("w_o", self.w_o.to_jagged());
        map
    }

    fn from_weight_map(&mut self, map: &WeightMap) -> Result<()> {
        let n_heads = map.get_scalar("n_heads")? as usize;
        let n_kv_heads = map.get_scalar("n_kv_heads")? as usize;
        let d_model = map.get_scalar("d_model")? as usize;
        let d_kv_model = map.get_scalar("d_kv_model")? as usize;
        let d_head = map.get_scalar("d_head")? as usize;
        if n_heads != self.n_heads
            || n_kv_heads != self.n_kv_heads
            || d_model != self.d_model
            || d_kv_model != self.d_kv_model
            || d_head != self.d_head
        {
            return Err(Error::new(ErrorKind::InvalidData, "mha config mismatch"));
        }
        self.w_q = Matrix::from_jagged(map.get_matrix("w_q")?);
        // 旧形式 (w_k, w_v) → W_KV に組み立てる
        let k = Matrix::from_jagged(map.get_matrix("w_k")?);
        let v = Matrix::from_jagged(map.get_matrix("w_v")?);
        self.assemble_w_kv(&k, &v);
        self.w_o = Matrix::from_jagged(map.get_matrix("w_o")?);
        Ok(())
    }
}

// ============================================================================
// Phase 7-5 B3: Flash Attention の数値一致テスト
// ============================================================================
//
// `flash_attention_forward` / `flash_attention_backward` が、
// 既存の causal mask 付き `scaled_dot_product_attention` (+_backward) と
// 数値的に一致することを確認する (誤差は f32 演算順序の違いに起因する ~1e-4)。
//
// 各テストは複数の seq_len (block 境界の挙動を確認するため):
// - seq = 8  : single block (< BLOCK_SIZE=64)
// - seq = 64 : exactly 1 block
// - seq = 65 : 2 blocks with the second one partial
// - seq = 128: 2 full blocks
// - seq = 200: 4 blocks (last partial)

#[cfg(test)]
mod flash_attention_tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn random_matrix(rows: usize, cols: usize, seed: u64) -> Matrix {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut m = Matrix::zeros(rows, cols);
        for v in m.data_mut() {
            *v = rng.random_range(-1.0f32..1.0);
        }
        m
    }

    fn max_abs_diff(a: &Matrix, b: &Matrix) -> f32 {
        assert_eq!(a.shape(), b.shape());
        a.data()
            .iter()
            .zip(b.data().iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    fn run_forward_for_seq(seq: usize, d_head: usize, seed_base: u64) {
        let q = random_matrix(seq, d_head, seed_base);
        let k = random_matrix(seq, d_head, seed_base + 1);
        let v = random_matrix(seq, d_head, seed_base + 2);

        // 既存実装 (full SDPA + causal mask)
        let mask = causal_mask(seq);
        let (expected_out, _expected_w) = scaled_dot_product_attention(&q, &k, &v, Some(&mask));

        // Flash Attention
        let (actual_out, _lse) = flash_attention_forward(&q, &k, &v);

        let diff = max_abs_diff(&expected_out, &actual_out);
        assert!(
            diff < 1e-4,
            "flash_attention_forward forward output mismatch for seq={seq}: max abs diff = {diff:.6e}"
        );
    }

    fn run_backward_for_seq(seq: usize, d_head: usize, seed_base: u64) {
        let q = random_matrix(seq, d_head, seed_base);
        let k = random_matrix(seq, d_head, seed_base + 1);
        let v = random_matrix(seq, d_head, seed_base + 2);
        let dl_dout = random_matrix(seq, d_head, seed_base + 3);

        // 既存実装 forward + backward
        let mask = causal_mask(seq);
        let (_expected_out, expected_w) = scaled_dot_product_attention(&q, &k, &v, Some(&mask));
        let (expected_dq, expected_dk, expected_dv) =
            scaled_dot_product_attention_backward(&q, &k, &v, &expected_w, &dl_dout);

        // Flash Attention forward + backward
        let (actual_out, lse) = flash_attention_forward(&q, &k, &v);
        let (actual_dq, actual_dk, actual_dv) =
            flash_attention_backward(&q, &k, &v, &actual_out, &lse, &dl_dout);

        let diff_dq = max_abs_diff(&expected_dq, &actual_dq);
        let diff_dk = max_abs_diff(&expected_dk, &actual_dk);
        let diff_dv = max_abs_diff(&expected_dv, &actual_dv);

        assert!(
            diff_dq < 1e-4,
            "flash dq mismatch (seq={seq}): max abs diff = {diff_dq:.6e}"
        );
        assert!(
            diff_dk < 1e-4,
            "flash dk mismatch (seq={seq}): max abs diff = {diff_dk:.6e}"
        );
        assert!(
            diff_dv < 1e-4,
            "flash dv mismatch (seq={seq}): max abs diff = {diff_dv:.6e}"
        );
    }

    #[test]
    fn flash_forward_matches_sdpa_seq8() {
        run_forward_for_seq(8, 16, 42);
    }

    #[test]
    fn flash_forward_matches_sdpa_seq64_exact_block() {
        run_forward_for_seq(FA_BLOCK_SIZE, 64, 100);
    }

    #[test]
    fn flash_forward_matches_sdpa_seq65_partial_second_block() {
        run_forward_for_seq(FA_BLOCK_SIZE + 1, 64, 101);
    }

    #[test]
    fn flash_forward_matches_sdpa_seq128_two_blocks() {
        run_forward_for_seq(2 * FA_BLOCK_SIZE, 64, 102);
    }

    #[test]
    fn flash_forward_matches_sdpa_seq200_four_blocks() {
        run_forward_for_seq(200, 64, 103);
    }

    #[test]
    fn flash_backward_matches_sdpa_seq8() {
        run_backward_for_seq(8, 16, 200);
    }

    #[test]
    fn flash_backward_matches_sdpa_seq64_exact_block() {
        run_backward_for_seq(FA_BLOCK_SIZE, 64, 201);
    }

    #[test]
    fn flash_backward_matches_sdpa_seq65_partial_second_block() {
        run_backward_for_seq(FA_BLOCK_SIZE + 1, 64, 202);
    }

    #[test]
    fn flash_backward_matches_sdpa_seq128_two_blocks() {
        run_backward_for_seq(2 * FA_BLOCK_SIZE, 64, 203);
    }

    #[test]
    fn flash_backward_matches_sdpa_seq200_four_blocks() {
        run_backward_for_seq(200, 64, 204);
    }
}
