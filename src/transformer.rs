use std::io::Error;
use std::io::ErrorKind;
use std::io::Result;

use crate::normalization::Normalization;
use crate::normalization::NormalizationKind;
use crate::normalization::load_normalization;
use crate::{
    adam_w::AdamW,
    checkpoint::{Checkpointable, WeightMap},
    transformer_block::TransformerBlock,
};

pub struct Transformer {
    blocks: Vec<TransformerBlock>,
    final_norm: Box<dyn Normalization>,
}

impl Transformer {
    pub fn new(
        n_layers: usize,
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        dropout_p: f32,
        normalization_kind: NormalizationKind,
    ) -> Self {
        Self {
            blocks: (0..n_layers)
                .map(|_| {
                    TransformerBlock::new(d_model, n_heads, d_ff, dropout_p, normalization_kind)
                })
                .collect(),
            final_norm: load_normalization(normalization_kind, d_model),
        }
    }

    pub fn set_training(&mut self, training: bool) {
        for block in &mut self.blocks {
            block.set_training(training);
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

    pub fn zero_grad(&mut self) {
        for block in &mut self.blocks {
            block.zero_grad();
        }
        self.final_norm.zero_grad();
    }

    pub fn apply_gradients(&mut self, opt: &mut AdamW, prefix: &str) {
        self.final_norm
            .apply_gradients(opt, &format!("{prefix}.final_norm"));
        for (i, block) in self.blocks.iter_mut().enumerate() {
            block.apply_gradients(opt, &format!("{prefix}.block{i}"));
        }
    }
}

impl Checkpointable for Transformer {
    fn to_weight_map(&self) -> WeightMap {
        let mut map = WeightMap::new();
        map.insert_scalar("n_layers", self.blocks.len() as u64);
        for (i, block) in self.blocks.iter().enumerate() {
            map.merge(&format!("blocks.{i}"), block.to_weight_map());
        }
        map.merge("final_norm", self.final_norm.to_weight_map());
        map
    }

    fn from_weight_map(&mut self, map: &WeightMap) -> Result<()> {
        let n_layers = map.get_scalar("n_layers")? as usize;
        if n_layers != self.blocks.len() {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "transformer n_layer mismatch",
            ));
        }
        for (i, block) in self.blocks.iter_mut().enumerate() {
            block.from_weight_map(&map.scoped(&format!("blocks.{i}")))?;
        }
        self.final_norm.from_weight_map(&map.scoped("final_norm"))?;
        Ok(())
    }
}
