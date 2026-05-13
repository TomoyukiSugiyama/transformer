use crate::MultiHeadAttention;
use crate::adam_w::AdamW;
use crate::checkpoint::Checkpointable;
use crate::checkpoint::WeightMap;
use crate::dropout::Dropout;
use crate::feed_forward::FeedForward;
use crate::feed_forward::FeedForwardKind;
use crate::feed_forward::load_feed_forward;
use crate::kv_cache::KvCache;
use crate::matrix::Matrix;
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
        d_ff: usize,
        dropout_p: f32,
        normalization_kind: NormalizationKind,
        feed_forward_kind: FeedForwardKind,
        rope: Option<Rope>,
    ) -> Self {
        Self {
            mha: MultiHeadAttention::new(d_model, n_heads, rope),
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

    /// Matrix 直叩き forward (Phase 7 高速化で導入)。
    /// Pre-LN: Norm → Sublayer → Dropout → Residual
    pub fn forward_matrix(&mut self, x: &Matrix, mask: Option<&Vec<Vec<bool>>>) -> Matrix {
        self.cache_x = x.clone();
        let norm1 = self.norm1.forward_matrix(x);
        let attn_out = self.mha.forward_matrix(&norm1, mask);
        let attn_dropped = self.drop_attn.forward_matrix(&attn_out);
        let mut x2 = x.clone();
        x2.add_in_place(&attn_dropped);

        self.cache_x2 = x2.clone();
        let norm2 = self.norm2.forward_matrix(&x2);
        let ffn_out = self.ffn.forward_matrix(&norm2);
        let ffn_dropped = self.drop_ffn.forward_matrix(&ffn_out);
        let mut out = x2;
        out.add_in_place(&ffn_dropped);
        out
    }

    /// Matrix 直叩き backward (Phase 7 高速化で導入)。
    pub fn backward_matrix(&mut self, dl_dout: &Matrix) -> Matrix {
        // FFN side
        let dl_dffn_out = self.drop_ffn.backward_matrix(dl_dout);
        let dl_dnorm2 = self.ffn.backward_matrix(&dl_dffn_out);
        let dl_dx2_from_ffn = self.norm2.backward_matrix(&dl_dnorm2);

        let mut dl_dx2 = dl_dout.clone();
        dl_dx2.add_in_place(&dl_dx2_from_ffn);

        // MHA side
        let dl_dattn_out = self.drop_attn.backward_matrix(&dl_dx2);
        let dl_dnorm1 = self.mha.backward_matrix(&dl_dattn_out);
        let dl_dx_from_mha = self.norm1.backward_matrix(&dl_dnorm1);

        let mut dl_dx = dl_dx2;
        dl_dx.add_in_place(&dl_dx_from_mha);
        dl_dx
    }

    /// 旧 API: jagged → Matrix 経由。
    pub fn forward(&mut self, x: &[Vec<f32>], mask: Option<&Vec<Vec<bool>>>) -> Vec<Vec<f32>> {
        let xm = Matrix::from_jagged(x);
        self.forward_matrix(&xm, mask).to_jagged()
    }

    /// 旧 API: jagged → Matrix 経由。
    pub fn backward(&mut self, dl_dout: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let dy = Matrix::from_jagged(dl_dout);
        self.backward_matrix(&dy).to_jagged()
    }

    /// 推論専用 (KV cache あり) の 1 token 前進。
    /// `x_new` は単一 token (`d_model` 長)、 戻り値も `d_model` 長。
    /// dropout は **無効化済み** であることを呼び出し側で保証 (`set_training(false)`)。
    pub fn forward_step(&mut self, x_new: &[f32], cache: &mut KvCache) -> Vec<f32> {
        // norm/ffn は既存の `forward(&[Vec<f32>])` を 1-row Vec で呼び出して再利用。
        // 内部の training cache は上書きされるが、 backward は呼ばれない前提なので無害。
        let single = vec![x_new.to_vec()];

        let norm1 = self.norm1.forward(&single);
        let attn_out = self.mha.forward_step(&norm1[0], cache);
        // dropout は eval モードなら identity (drop_attn.set_training(false) 済)
        let attn_dropped = self.drop_attn.forward(&[attn_out]);
        let x2: Vec<f32> = x_new
            .iter()
            .zip(attn_dropped[0].iter())
            .map(|(a, b)| a + b)
            .collect();

        let single_x2 = vec![x2.clone()];
        let norm2 = self.norm2.forward(&single_x2);
        let ffn_out = self.ffn.forward(&norm2);
        let ffn_dropped = self.drop_ffn.forward(&ffn_out);

        x2.iter()
            .zip(ffn_dropped[0].iter())
            .map(|(a, b)| a + b)
            .collect()
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
