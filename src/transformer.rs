use crate::{layer_normalization::LayerNormalization, transformer_block::TransformerBlock};

pub struct Transformer {
    blocks: Vec<TransformerBlock>,
    final_norm: LayerNormalization,
    d_model: usize,
}

impl Transformer {
    pub fn new(n_layers: usize, d_model: usize, n_heads: usize, d_ff: usize) -> Self {
        Self {
            blocks: (0..n_layers)
                .map(|_| TransformerBlock::new(d_model, n_heads, d_ff))
                .collect(),
            final_norm: LayerNormalization::new(d_model),
            d_model,
        }
    }

    pub fn forward(&mut self, x: &[Vec<f32>], mask: Option<&Vec<Vec<bool>>>) -> Vec<Vec<f32>> {
        let mut x = x.to_vec();
        for i in 0..self.blocks.len(){
        // for (i, block) in self.blocks.iter().enumerate() {
            x = self.blocks[i].forward(&x, mask);
            let norm: f32 = x
                .iter()
                .flat_map(|row| row.iter())
                .map(|v| v.powi(2))
                .sum::<f32>()
                .sqrt();
            println!("Block[{i}] output norm {:.4}", norm);
        }
        self.final_norm.forward(&x)
    }
}
