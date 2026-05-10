use std::io::{Error, ErrorKind, Result};

use crate::{
    bpe_tokenizer::BpeTokenizer,
    char_tokenizer::CharTokenizer,
    checkpoint::{Checkpointable, WeightMap},
};

/// 学習・推論で使用するトークナイザの抽象。
/// 実装は `BpeTokenizer` (subword) と `CharTokenizer` (char-level) の 2 種類。
pub trait Tokenizer: Send {
    fn vocab_size(&self) -> usize;
    fn pad_id(&self) -> usize;
    /// 生成停止に用いる ID。 char-level など EOS を持たない実装は
    /// `usize::MAX` を返し、 通常の生成中に一致しないようにする。
    fn eos_id(&self) -> usize;
    /// コーパス全体を 1 度だけエンコード（BPE は BOS/EOS を付ける、 char は付けない）。
    fn encode_long(&self, text: &str) -> Vec<usize>;
    /// 推論プロンプト用のエンコード。
    fn encode_prompt(&self, text: &str) -> Vec<usize>;
    fn decode(&self, ids: &[usize]) -> String;
    fn kind(&self) -> TokenizerKind;
    fn to_weight_map(&self) -> WeightMap;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenizerKind {
    Bpe,
    Char,
}

impl TokenizerKind {
    pub fn as_u64(&self) -> u64 {
        match self {
            TokenizerKind::Bpe => 1,
            TokenizerKind::Char => 2,
        }
    }

    pub fn from_u64(v: u64) -> Result<Self> {
        match v {
            1 => Ok(TokenizerKind::Bpe),
            2 => Ok(TokenizerKind::Char),
            other => Err(Error::new(
                ErrorKind::InvalidData,
                format!("unknown tokenizer kind: {other}"),
            )),
        }
    }
}

/// `kind` を見て対応する具象トークナイザを生成し、 残りのフィールドを復元する。
/// 旧 checkpoint (kind 未保存) は BPE として扱う (後方互換)。
pub fn load_tokenizer(map: &WeightMap) -> Result<Box<dyn Tokenizer>> {
    let kind = match map.get_scalar("kind") {
        Ok(v) => TokenizerKind::from_u64(v)?,
        Err(_) => TokenizerKind::Bpe,
    };
    match kind {
        TokenizerKind::Bpe => {
            let mut t = BpeTokenizer::empty();
            t.from_weight_map(map)?;
            Ok(Box::new(t))
        }
        TokenizerKind::Char => {
            let mut t = CharTokenizer::empty();
            t.from_weight_map(map)?;
            Ok(Box::new(t))
        }
    }
}

/// テキストから新規にトークナイザを学習する。
/// `vocab_size` は BPE のみで参照される (Char は corpus の文字種から自動決定)。
pub fn train_tokenizer(
    kind: TokenizerKind,
    text: &str,
    vocab_size: usize,
) -> Box<dyn Tokenizer> {
    match kind {
        TokenizerKind::Bpe => Box::new(BpeTokenizer::train(text, vocab_size)),
        TokenizerKind::Char => Box::new(CharTokenizer::train(text)),
    }
}
