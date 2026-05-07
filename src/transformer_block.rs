use crate::FeedForwardNetwork;
use crate::LayerNormalization;
use crate::MultiHeadAttention;
use crate::adam_w::AdamW;
use crate::checkpoint::Checkpointable;
use crate::checkpoint::WeightMap;

/// Attention → Add&Norm → FFN → Add&Norm
pub struct TransformerBlock {
    mha: MultiHeadAttention,
    norm1: LayerNormalization,
    ffn: FeedForwardNetwork,
    norm2: LayerNormalization,
    cache_x: Vec<Vec<f32>>,
    cache_x2: Vec<Vec<f32>>,
}

impl TransformerBlock {
    pub fn new(d_model: usize, n_heads: usize, d_ff: usize) -> Self {
        Self {
            mha: MultiHeadAttention::new(d_model, n_heads),
            norm1: LayerNormalization::new(d_model),
            ffn: FeedForwardNetwork::new(d_model, d_ff),
            norm2: LayerNormalization::new(d_model),
            cache_x: Vec::new(),
            cache_x2: Vec::new(),
        }
    }

    /// Pre-LN: Norm → Sublayer → Residual
    pub fn forward(&mut self, x: &[Vec<f32>], mask: Option<&Vec<Vec<bool>>>) -> Vec<Vec<f32>> {
        self.cache_x = x.to_vec();
        let norm1 = self.norm1.forward(x);
        let (attn_out, _) = self.mha.forward(&norm1, mask);
        let x2 = residual_add(x, &attn_out);

        self.cache_x2 = x2.clone();
        let norm2 = self.norm2.forward(&x2);
        let ffn_out = self.ffn.forward(&norm2);
        let out = residual_add(&x2, &ffn_out);

        out
    }

    pub fn backward(&mut self, dl_dout: &[Vec<f32>]) -> Vec<Vec<f32>> {
        // FFN
        // out = x2 + ffn(norm2(x2))
        // dl_dout は x2 と ffn_out の両方に流れる（residual）
        let dl_dffn_dout = dl_dout;
        let dl_dx2_from_res = dl_dout.to_vec();

        let dl_dnorm2 = self.ffn.backward(dl_dffn_dout);
        let dl_dx2_from_ffn = self.norm2.backward(&dl_dnorm2);

        let dl_dx2 = residual_add(&dl_dx2_from_res, &dl_dx2_from_ffn);

        // MHA
        // x2 = x + mha(norm1(x))
        let dl_dattn_out = &dl_dx2;
        let dl_dx_from_res = dl_dx2.clone();

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
