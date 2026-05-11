use std::io::{Error, ErrorKind, Result};

/// 位置エンコーディングの種別。 checkpoint へ保存・復元するためのタグ。
///
/// - `Sinusoidal`: 元の Vaswani 2017 方式。 埋め込みに sin/cos の位置ベクトルを **加算**。
///   実装は `src/sinusoidal_pe.rs`。
/// - `Rope`: 回転位置埋め込み (Su et al. 2021)。 各 attention 層で Q, K に **回転** を掛ける。
///   実装は `src/rope.rs` と `src/multi_head_attention.rs` (内蔵)。 `pe` 層は使わない。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PositionalEncodingKind {
    Sinusoidal,
    Rope,
}

impl PositionalEncodingKind {
    pub fn as_u64(&self) -> u64 {
        match self {
            PositionalEncodingKind::Sinusoidal => 1,
            PositionalEncodingKind::Rope => 2,
        }
    }

    pub fn from_u64(v: u64) -> Result<Self> {
        match v {
            1 => Ok(PositionalEncodingKind::Sinusoidal),
            2 => Ok(PositionalEncodingKind::Rope),
            other => Err(Error::new(
                ErrorKind::InvalidData,
                format!("unknown positional encoding kind: {other}"),
            )),
        }
    }
}
