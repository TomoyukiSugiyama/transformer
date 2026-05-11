use std::io::{Error, ErrorKind, Result};

use crate::{
    adam_w::AdamW, checkpoint::Checkpointable, layer_normalization::LayerNormalization,
    root_mean_square_layer_normalization::RootMeanSquareLayerNormalization,
};

pub trait Normalization: Checkpointable {
    fn forward(&mut self, x: &[Vec<f32>]) -> Vec<Vec<f32>>;
    fn backward(&mut self, dl_dy: &[Vec<f32>]) -> Vec<Vec<f32>>;
    fn zero_grad(&mut self);
    fn apply_gradients(&mut self, opt: &mut AdamW, prefix: &str);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormalizationKind {
    Layer,
    Rms,
}

impl NormalizationKind {
    pub fn as_u64(&self) -> u64{
        match self {
            NormalizationKind::Layer => 1,
            NormalizationKind::Rms => 2,
        }
    }

    pub fn from_u64(v: u64) -> Result<Self> {
        match v {
            1 => Ok(NormalizationKind::Layer),
            2 => Ok(NormalizationKind::Rms),
            other => Err(Error::new(ErrorKind::InvalidData,format!("unknown normalization kind: {other}")))
        }
    }
}

pub fn load_normalization(kind: NormalizationKind, d_model: usize) -> Box<dyn Normalization> {
    match kind {
        NormalizationKind::Layer => Box::new(LayerNormalization::new(d_model)),
        NormalizationKind::Rms => Box::new(RootMeanSquareLayerNormalization::new(d_model)),
    }
}
