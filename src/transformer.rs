use crate::{
    adam_w::AdamW, layer_normalization::LayerNormalization, transformer_block::TransformerBlock,
};

pub struct Transformer {
    blocks: Vec<TransformerBlock>,
    final_norm: LayerNormalization,
}

impl Transformer {
    pub fn new(n_layers: usize, d_model: usize, n_heads: usize, d_ff: usize) -> Self {
        Self {
            blocks: (0..n_layers)
                .map(|_| TransformerBlock::new(d_model, n_heads, d_ff))
                .collect(),
            final_norm: LayerNormalization::new(d_model),
        }
    }

    pub fn forward(&mut self, x: &[Vec<f32>], mask: Option<&Vec<Vec<bool>>>) -> Vec<Vec<f32>> {
        let mut x = x.to_vec();
        for i in 0..self.blocks.len() {
            x = self.blocks[i].forward(&x, mask);
        }
        self.final_norm.forward(&x)
    }

    pub fn backward(&mut self, dl_dout: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let mut d1 = self.final_norm.backward(dl_dout);

        for i in (0..self.blocks.len()).rev() {
            d1 = self.blocks[i].backward(&d1);
        }
        d1
    }

    pub fn apply_gradients(&mut self, opt: &mut AdamW, prefix: &str) {
        self.final_norm
            .apply_gradients(opt, &format!("{prefix}.final_norm"));
        for (i, block) in self.blocks.iter_mut().enumerate() {
            block.apply_gradients(opt, &format!("{prefix}.block{i}.final_norm"));
        }
    }
}
