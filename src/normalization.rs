use std::io::{Error, ErrorKind, Result};

use crate::{
    adam_w::AdamW, checkpoint::Checkpointable, layer_normalization::LayerNormalization, matrix::Matrix,
    root_mean_square_layer_normalization::RootMeanSquareLayerNormalization,
};

pub trait Normalization: Checkpointable {
    /// 旧 API。 jagged で受け取り jagged で返す。 内部では Matrix 版に変換して呼ぶ。
    /// 新規コードからは `forward_matrix` を直接呼ぶこと (allocation を 1 ペア節約できる)。
    fn forward(&mut self, x: &[Vec<f32>]) -> Vec<Vec<f32>>;
    fn backward(&mut self, dl_dy: &[Vec<f32>]) -> Vec<Vec<f32>>;
    /// Matrix 直叩き API (Phase 7 高速化で導入)。 forward 中は内部 cache が Matrix で保持される。
    fn forward_matrix(&mut self, x: &Matrix) -> Matrix;
    fn backward_matrix(&mut self, dl_dy: &Matrix) -> Matrix;
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
