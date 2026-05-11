use crate::MultiHeadAttention;
use crate::adam_w::AdamW;
use crate::checkpoint::Checkpointable;
use crate::checkpoint::WeightMap;
use crate::dropout::Dropout;
use crate::feed_forward::FeedForward;
use crate::feed_forward::FeedForwardKind;
use crate::feed_forward::load_feed_forward;
use crate::normalization::Normalization;
use crate::normalization::NormalizationKind;
use crate::normalization::load_normalization;

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
    cache_x: Vec<Vec<f32>>,
    cache_x2: Vec<Vec<f32>>,
}

impl TransformerBlock {
    pub fn new(
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        dropout_p: f32,
        normalization_kind: NormalizationKind,
        feed_forward_kind: FeedForwardKind,
    ) -> Self {
        Self {
            mha: MultiHeadAttention::new(d_model, n_heads),
            norm1: load_normalization(normalization_kind, d_model),
            drop_attn: Dropout::new(dropout_p),
            ffn: load_feed_forward(feed_forward_kind, d_model, d_ff),
            norm2: load_normalization(normalization_kind, d_model),
            drop_ffn: Dropout::new(dropout_p),
            cache_x: Vec::new(),
            cache_x2: Vec::new(),
        }
    }

    pub fn set_training(&mut self, training: bool) {
        self.drop_attn.set_training(training);
        self.drop_ffn.set_training(training);
    }

    /// Pre-LN: Norm → Sublayer → Dropout → Residual
    pub fn forward(&mut self, x: &[Vec<f32>], mask: Option<&Vec<Vec<bool>>>) -> Vec<Vec<f32>> {
        self.cache_x = x.to_vec();
        let norm1 = self.norm1.forward(x);
        let attn_out = self.mha.forward(&norm1, mask);
        let attn_dropped = self.drop_attn.forward(&attn_out);
        let x2 = residual_add(x, &attn_dropped);

        self.cache_x2 = x2.clone();
        let norm2 = self.norm2.forward(&x2);
        let ffn_out = self.ffn.forward(&norm2);
        let ffn_dropped = self.drop_ffn.forward(&ffn_out);
        let out = residual_add(&x2, &ffn_dropped);

        out
    }

    pub fn backward(&mut self, dl_dout: &[Vec<f32>]) -> Vec<Vec<f32>> {
        // FFN side
        // out = x2 + drop_ffn(ffn(norm2(x2)))
        // dl_dout は x2 と ffn_dropped の両方に流れる (residual)
        let dl_dffn_dropped = dl_dout;
        let dl_dx2_from_res = dl_dout.to_vec();

        let dl_dffn_out = self.drop_ffn.backward(dl_dffn_dropped);
        let dl_dnorm2 = self.ffn.backward(&dl_dffn_out);
        let dl_dx2_from_ffn = self.norm2.backward(&dl_dnorm2);

        let dl_dx2 = residual_add(&dl_dx2_from_res, &dl_dx2_from_ffn);

        // MHA side
        // x2 = x + drop_attn(mha(norm1(x)))
        let dl_dattn_dropped = &dl_dx2;
        let dl_dx_from_res = dl_dx2.clone();

        let dl_dattn_out = self.drop_attn.backward(dl_dattn_dropped);
        let dl_dnorm1 = self.mha.backward(&dl_dattn_out);
        let dl_dx_from_mha = self.norm1.backward(&dl_dnorm1);

        residual_add(&dl_dx_from_res, &dl_dx_from_mha)
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

fn residual_add(x: &[Vec<f32>], sublayer_out: &[Vec<f32>]) -> Vec<Vec<f32>> {
    x.iter()
        .zip(sublayer_out.iter())
        .map(|(xi, si)| xi.iter().zip(si.iter()).map(|(a, b)| a + b).collect())
        .collect()
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
