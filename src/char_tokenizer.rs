use std::{
    collections::HashMap,
    io::{Error, ErrorKind, Result},
};

use crate::{
    checkpoint::{Checkpointable, WeightMap},
    tokenizer::{Tokenizer, TokenizerKind},
};

/// nanoGPT 流の char-level tokenizer。
///
/// vocab レイアウト:
///   - id 0: UNK (学習コーパスに存在しない char が prompt 等で来た場合の fallback)
///   - id 1..=N: コーパス出現順 (= ソート順) の各 char
///
/// BOS/EOS/PAD は持たない。 生成停止は `max_new_token` でのみ行う想定なので
/// `eos_id()` は `usize::MAX` を返し、 通常の生成中に偶然一致しないようにする。
/// `pad_id()` は UNK と同じ id 0 を返す (実際の学習ループでは pad は出現しないため
/// マスキング上の影響なし)。
pub struct CharTokenizer {
    id_to_char: Vec<char>,
    char_to_id: HashMap<char, usize>,
}

impl CharTokenizer {
    /// テキストに含まれる char をユニーク化・ソートして vocab を構築する。
    pub fn train(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort();
        chars.dedup();

        let mut id_to_char = Vec::with_capacity(chars.len() + 1);
        // id 0 は UNK (NUL を sentinel として持たせる)
        id_to_char.push('\0');
        id_to_char.extend(chars);

        let char_to_id: HashMap<char, usize> = id_to_char
            .iter()
            .enumerate()
            .skip(1) // id 0 (UNK) は char_to_id に登録しない
            .map(|(i, &c)| (c, i))
            .collect();

        Self {
            id_to_char,
            char_to_id,
        }
    }

    pub fn empty() -> Self {
        Self {
            id_to_char: Vec::new(),
            char_to_id: HashMap::new(),
        }
    }

    fn unk_id(&self) -> usize {
        0
    }

    fn encode_chars(&self, text: &str) -> Vec<usize> {
        let unk = self.unk_id();
        text.chars()
            .map(|c| self.char_to_id.get(&c).copied().unwrap_or(unk))
            .collect()
    }
}

impl Tokenizer for CharTokenizer {
    fn vocab_size(&self) -> usize {
        self.id_to_char.len()
    }

    fn pad_id(&self) -> usize {
        // UNK と同じ id。 学習コーパス内には UNK が出現しないため、
        // pad マスキングで本物のトークンが除外される心配はない。
        0
    }

    fn eos_id(&self) -> usize {
        // EOS を持たないので、 vocab に存在しえない値を返して
        // generate ループの停止条件に絶対一致させない。
        usize::MAX
    }

    fn encode_long(&self, text: &str) -> Vec<usize> {
        // BPE のような BOS/EOS は付けない (nanoGPT char と同じ)。
        self.encode_chars(text)
    }

    fn encode_prompt(&self, text: &str) -> Vec<usize> {
        self.encode_chars(text)
    }

    fn decode(&self, ids: &[usize]) -> String {
        let unk = self.unk_id();
        let mut out = String::with_capacity(ids.len());
        for &id in ids {
            if id == unk {
                continue;
            }
            if let Some(&c) = self.id_to_char.get(id) {
                out.push(c);
            }
        }
        out
    }

    fn kind(&self) -> TokenizerKind {
        TokenizerKind::Char
    }

    fn to_weight_map(&self) -> WeightMap {
        let mut map = WeightMap::new();
        map.insert_scalar("kind", TokenizerKind::Char.as_u64());
        // 各エントリを 1 char の文字列として保存。 id 0 (UNK) は空文字列。
        let strs: Vec<String> = self
            .id_to_char
            .iter()
            .map(|&c| if c == '\0' { String::new() } else { c.to_string() })
            .collect();
        map.insert_strings("id_to_char", strs);
        map
    }
}

impl Checkpointable for CharTokenizer {
    fn to_weight_map(&self) -> WeightMap {
        Tokenizer::to_weight_map(self)
    }

    fn from_weight_map(&mut self, map: &WeightMap) -> Result<()> {
        let strs = map.get_strings("id_to_char")?;
        if strs.is_empty() {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "char tokenizer: id_to_char is empty",
            ));
        }
        let mut id_to_char = Vec::with_capacity(strs.len());
        for (i, s) in strs.iter().enumerate() {
            if i == 0 {
                // UNK
                id_to_char.push('\0');
                continue;
            }
            let mut iter = s.chars();
            let c = iter.next().ok_or_else(|| {
                Error::new(
                    ErrorKind::InvalidData,
                    format!("char tokenizer: empty string at id {i}"),
                )
            })?;
            if iter.next().is_some() {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    format!("char tokenizer: multi-char string at id {i}: {s:?}"),
                ));
            }
            id_to_char.push(c);
        }
        let char_to_id: HashMap<char, usize> = id_to_char
            .iter()
            .enumerate()
            .skip(1)
            .map(|(i, &c)| (c, i))
            .collect();
        self.id_to_char = id_to_char;
        self.char_to_id = char_to_id;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn train_includes_unk_at_id_zero() {
        let t = CharTokenizer::train("abc");
        assert_eq!(t.vocab_size(), 4); // UNK + a, b, c
        assert_eq!(t.id_to_char[0], '\0');
        assert_eq!(t.encode_chars("a")[0], 1);
    }

    #[test]
    fn encode_decode_roundtrip() {
        let text = "Hello, World!\nThis is a test.";
        let t = CharTokenizer::train(text);
        let ids = t.encode_long(text);
        let decoded = t.decode(&ids);
        assert_eq!(decoded, text);
    }

    #[test]
    fn unknown_char_maps_to_unk_and_is_dropped_in_decode() {
        let t = CharTokenizer::train("abc");
        let ids = t.encode_prompt("axb");
        assert_eq!(ids, vec![1, 0, 2]); // a=1, UNK=0, b=2
        assert_eq!(t.decode(&ids), "ab");
    }

    #[test]
    fn save_and_load_roundtrip() {
        let original = CharTokenizer::train("the quick brown fox\n");
        let map = Tokenizer::to_weight_map(&original);

        let mut restored = CharTokenizer::empty();
        Checkpointable::from_weight_map(&mut restored, &map).unwrap();

        assert_eq!(restored.vocab_size(), original.vocab_size());
        let text = "fox\nthe";
        assert_eq!(restored.encode_long(text), original.encode_long(text));
    }
}
