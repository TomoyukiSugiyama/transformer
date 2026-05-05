use crate::FeedForwardNetwork;
use crate::LayerNormalization;
use crate::MultiHeadAttention;

/// Attention → Add&Norm → FFN → Add&Norm
pub struct TransformerBlock {
    mha: MultiHeadAttention,
    norm1: LayerNormalization,
    ffn: FeedForwardNetwork,
    norm2: LayerNormalization,
}

impl TransformerBlock {
    pub fn new(d_model: usize, n_heads: usize, d_ff: usize) -> Self {
        Self {
            mha: MultiHeadAttention::new(d_model, n_heads),
            norm1: LayerNormalization::new(d_model),
            ffn: FeedForwardNetwork::new(d_model, d_ff),
            norm2: LayerNormalization::new(d_model),
        }
    }

    /// Pre-LN方式
    pub fn forward(&self, x: &[Vec<f32>], mask: Option<&Vec<Vec<bool>>>) -> Vec<Vec<f32>> {
        let norm1 = self.norm1.forward(x);
        let (attn_out, _) = self.mha.forward(&norm1, mask);
        let x = residual_add(x, &attn_out);

        let norm2 = self.norm2.forward(&x);
        let ffn_out = self.ffn.forward(&norm2);
        let x = residual_add(&x, &ffn_out);

        x
    }
}

fn residual_add(x: &[Vec<f32>], sublayer_out: &[Vec<f32>]) -> Vec<Vec<f32>> {
    x.iter()
        .zip(sublayer_out.iter())
        .map(|(xi, si)| xi.iter().zip(si.iter()).map(|(a, b)| a + b).collect())
        .collect()
}
