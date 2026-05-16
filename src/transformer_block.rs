use crate::adam_w::AdamW;
use crate::checkpoint::Checkpointable;
use crate::checkpoint::WeightMap;
use crate::dropout::Dropout;
use crate::feed_forward::FeedForward;
use crate::feed_forward::FeedForwardKind;
use crate::feed_forward::load_feed_forward;
use crate::kv_cache::KvCache;
use crate::matrix::Matrix;
use crate::multi_head_attention::MultiHeadAttention;
use crate::normalization::Normalization;
use crate::normalization::NormalizationKind;
use crate::normalization::load_normalization;
use crate::rope::Rope;

/// Attention → Add&Norm → FFN → Add&Norm
///
/// Pre-LN 構成。 dropout は **residual の直前** (= 各 sublayer の出力に対して)
/// 2 箇所で適用する。 これは GPT-2 / nanoGPT と同じ位置取り。
pub struct TransformerBlock {
    mha: MultiHeadAttention,
    norm1: Box<dyn Normalization>,
    drop_attn: Dropout,
    ffn: Box<dyn FeedForward>,
    norm2: Box<dyn Normalization>,
    drop_ffn: Dropout,
    cache_x: Matrix,
    cache_x2: Matrix,
}

impl TransformerBlock {
    pub fn new(
        d_model: usize,
        n_heads: usize,
        n_kv_heads: usize,
        d_ff: usize,
        dropout_p: f32,
        normalization_kind: NormalizationKind,
        feed_forward_kind: FeedForwardKind,
        rope: Option<Rope>,
    ) -> Self {
        Self {
            mha: MultiHeadAttention::new(d_model, n_heads, n_kv_heads, rope),
            norm1: load_normalization(normalization_kind, d_model),
            drop_attn: Dropout::new(dropout_p),
            ffn: load_feed_forward(feed_forward_kind, d_model, d_ff),
            norm2: load_normalization(normalization_kind, d_model),
            drop_ffn: Dropout::new(dropout_p),
            cache_x: Matrix::zeros(0, 0),
            cache_x2: Matrix::zeros(0, 0),
        }
    }

    pub fn set_training(&mut self, training: bool) {
        self.drop_attn.set_training(training);
        self.drop_ffn.set_training(training);
    }

    /// Pre-LN: Norm → Sublayer → Dropout → Residual
    pub fn forward(&mut self, x: &Matrix, mask: Option<&Vec<Vec<bool>>>) -> Matrix {
        self.cache_x = x.clone();
        let norm1 = self.norm1.forward(x);
        let attn_out = self.mha.forward(&norm1, mask);
        let attn_dropped = self.drop_attn.forward(&attn_out);
        let mut x2 = x.clone();
        x2.add_in_place(&attn_dropped);

        self.cache_x2 = x2.clone();
        let norm2 = self.norm2.forward(&x2);
        let ffn_out = self.ffn.forward(&norm2);
        let ffn_dropped = self.drop_ffn.forward(&ffn_out);
        let mut out = x2;
        out.add_in_place(&ffn_dropped);
        out
    }

    pub fn backward(&mut self, dl_dout: &Matrix) -> Matrix {
        // FFN side
        let dl_dffn_out = self.drop_ffn.backward(dl_dout);
        let dl_dnorm2 = self.ffn.backward(&dl_dffn_out);
        let dl_dx2_from_ffn = self.norm2.backward(&dl_dnorm2);

        let mut dl_dx2 = dl_dout.clone();
        dl_dx2.add_in_place(&dl_dx2_from_ffn);

        // MHA side
        let dl_dattn_out = self.drop_attn.backward(&dl_dx2);
        let dl_dnorm1 = self.mha.backward(&dl_dattn_out);
        let dl_dx_from_mha = self.norm1.backward(&dl_dnorm1);

        let mut dl_dx = dl_dx2;
        dl_dx.add_in_place(&dl_dx_from_mha);
        dl_dx
    }

    /// 推論専用 (KV cache あり) の 1 token 前進。
    /// `x_new` は単一 token (`d_model` 長)、 戻り値も `d_model` 長。
    /// dropout は **無効化済み** であることを呼び出し側で保証 (`set_training(false)`)。
    /// 内部では Matrix 直叩きパスを 1-row Matrix で呼び出す。
    /// 学習用 cache (cache_x / cache_x2 / cache_x_hat 等) は上書きされるが、
    /// backward は呼ばれない前提なので無害。
    pub fn forward_step(&mut self, x_new: &[f32], cache: &mut KvCache) -> Vec<f32> {
        let d = x_new.len();
        let x_m = Matrix::from_flat(x_new.to_vec(), 1, d);

        let norm1 = self.norm1.forward(&x_m);
        let attn_out_vec = self.mha.forward_step(norm1.row(0), cache);
        let attn_out_m = Matrix::from_flat(attn_out_vec, 1, d);
        let attn_dropped = self.drop_attn.forward(&attn_out_m);

        let mut x2 = x_m;
        x2.add_in_place(&attn_dropped);

        let norm2 = self.norm2.forward(&x2);
        let ffn_out = self.ffn.forward(&norm2);
        let ffn_dropped = self.drop_ffn.forward(&ffn_out);

        let mut out = x2;
        out.add_in_place(&ffn_dropped);
        out.row(0).to_vec()
    }

    pub fn zero_grad(&mut self) {
        self.mha.zero_grad();
        self.norm1.zero_grad();
        self.ffn.zero_grad();
        self.norm2.zero_grad();
    }

    pub fn apply_gradients(&mut self, opt: &mut AdamW, prefix: &str) {
        self.mha.apply_gradients(opt, &format!("{prefix}.mha"));
        self.norm1.apply_gradients(opt, &format!("{prefix}.norm1"));
        self.ffn.apply_gradients(opt, &format!("{prefix}.ffn"));
        self.norm2.apply_gradients(opt, &format!("{prefix}.norm2"));
    }
}

impl Checkpointable for TransformerBlock {
    fn to_weight_map(&self) -> WeightMap {
        let mut map = WeightMap::new();
        map.merge("mha", self.mha.to_weight_map());
        map.merge("norm1", self.norm1.to_weight_map());
        map.merge("ffn", self.ffn.to_weight_map());
        map.merge("norm2", self.norm2.to_weight_map());
        map
    }

    fn from_weight_map(&mut self, map: &WeightMap) -> std::io::Result<()> {
        self.mha.from_weight_map(&map.scoped("mha"))?;
        self.norm1.from_weight_map(&map.scoped("norm1"))?;
        self.ffn.from_weight_map(&map.scoped("ffn"))?;
        self.norm2.from_weight_map(&map.scoped("norm2"))?;
        Ok(())
    }
}
