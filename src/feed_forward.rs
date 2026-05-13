use std::io::{Error, ErrorKind, Result};

use crate::{
    adam_w::AdamW, checkpoint::Checkpointable, feed_forward_network::FeedForwardNetwork,
    matrix::Matrix, swiglu_feed_forward_network::SwiGluFeedForwardNetwork,
};

pub trait FeedForward: Checkpointable {
    /// Matrix 直叩き forward。
    fn forward(&mut self, x: &Matrix) -> Matrix;
    fn backward(&mut self, dl_dy: &Matrix) -> Matrix;
    fn zero_grad(&mut self);
    fn apply_gradients(&mut self, opt: &mut AdamW, prefix: &str);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeedForwardKind {
    Gelu, // 既存
    SwiGlu,
}

impl FeedForwardKind {
    pub fn as_u64(&self) -> u64 {
        match self {
            FeedForwardKind::Gelu => 1,
            FeedForwardKind::SwiGlu => 2,
        }
    }
    pub fn from_u64(v: u64) -> Result<Self> {
        match v {
            1 => Ok(FeedForwardKind::Gelu),
            2 => Ok(FeedForwardKind::SwiGlu),
            other => Err(Error::new(
                ErrorKind::InvalidData,
                format!("unknown feed forward kind: {other}"),
            )),
        }
    }
}

pub fn load_feed_forward(
    kind: FeedForwardKind,
    d_model: usize,
    d_ff: usize,
) -> Box<dyn FeedForward> {
    match kind {
        FeedForwardKind::Gelu => Box::new(FeedForwardNetwork::new(d_model, d_ff)),
        FeedForwardKind::SwiGlu => Box::new(SwiGluFeedForwardNetwork::new(d_model, d_ff)),
    }
}
