use std::collections::HashMap;

use crate::checkpoint::{Checkpointable, WeightMap};
pub struct Tokenizer {
    vocab: HashMap<String, usize>,
    id_to_token: Vec<String>,
    unk_id: usize,
}

pub struct Encoding {
    pub input_ids: Vec<usize>,
    pub attention_mask: Vec<u8>,
}

impl Tokenizer {
    pub const PAD: &'static str = "<PAD>";
    pub const UNK: &'static str = "<UNK>";
    pub const BOS: &'static str = "<BOS>";
    pub const EOS: &'static str = "<EOS>";

    pub fn build(corpus: &[&str]) -> Self {
        let mut vocab: HashMap<String, usize> = HashMap::new();

        for special in [Self::PAD, Self::UNK, Self::BOS, Self::EOS] {
            vocab.insert(special.to_string(), vocab.len());
        }

        for text in corpus {
            let tokens = Self::tokenize_text(text);
            for token in tokens {
                if !vocab.contains_key(&token) {
                    vocab.insert(token, vocab.len());
                }
            }
        }
        let mut id_to_token = vec![String::new(); vocab.len()];
        for (token, &id) in &vocab {
            id_to_token[id] = token.clone();
        }

        let unk_id: usize = *vocab.get(Self::UNK).unwrap();
        Self {
            vocab,
            id_to_token,
            unk_id,
        }
    }

    fn tokenize_text(text: &str) -> Vec<String> {
        let text = text.to_lowercase();
        let mut tokens = Vec::new();
        let mut current = String::new();
        for ch in text.chars() {
            if ch.is_alphanumeric() || ch == '\'' {
                current.push(ch);
            } else {
                if !current.is_empty() {
                    tokens.push(current.clone());
                    current.clear();
                }

                if !ch.is_whitespace() {
                    tokens.push(ch.to_string());
                }
            }
        }
        if !current.is_empty() {
            tokens.push(current);
        }
        tokens
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn id_to_token_str(&self, id: usize) -> Option<&str> {
        self.id_to_token.get(id).map(String::as_str)
    }

    fn encode(&self, text: &str) -> Vec<usize> {
        let bos = *self.vocab.get(Self::BOS).unwrap();
        let eos = *self.vocab.get(Self::EOS).unwrap();
        let mut ids = vec![bos];
        let tokens = Self::tokenize_text(text);

        for token in tokens {
            ids.push(self.vocab.get(&token).copied().unwrap_or(self.unk_id));
        }
        ids.push(eos);
        ids
    }

    pub fn decord(&self, ids: &[usize]) -> String {
        ids.iter()
            .filter_map(|&id| self.id_to_token.get(id))
            .filter(|t| ![Self::PAD, Self::UNK, Self::BOS, Self::EOS].contains(&t.as_str()))
            .cloned()
            .collect::<Vec<_>>()
            .join(" ")
    }

    pub fn encode_simple(&self, text: &str) -> Vec<usize> {
        self.encode(text)
    }

    /// 生成用: BOS は付けるが EOS は付けない
    pub fn encode_prompt(&self, text: &str) -> Vec<usize> {
        let bos = *self.vocab.get(Self::BOS).unwrap();
        let mut ids = vec![bos];
        for token in Self::tokenize_text(text) {
            ids.push(self.vocab.get(&token).copied().unwrap_or(self.unk_id));
        }
        ids
    }

    pub fn encode_with_padding(&self, text: &str, max_len: usize) -> Encoding {
        let pad_id = *self.vocab.get(Self::PAD).unwrap_or(&0);
        let eos_id = *self.vocab.get(Self::EOS).unwrap_or(&0);
        let mut input_ids = self.encode(text);

        if input_ids.len() >= max_len {
            input_ids.truncate(max_len - 1);
            input_ids.push(eos_id);
        } else {
            input_ids.resize(max_len, pad_id);
        }

        let real_len = input_ids.len().min(max_len);
        let mut attention_mask = vec![1u8; real_len];
        attention_mask.resize(max_len, 0u8);

        Encoding {
            input_ids,
            attention_mask,
        }
    }

    pub fn encode_with_padding_from_ids(&self, ids: &[usize], max_len: usize) -> Encoding {
        let pad_id = *self.vocab.get(Self::PAD).unwrap_or(&0);
        let eos_id = *self.vocab.get(Self::EOS).unwrap_or(&0);

        let real_len = ids.len().min(max_len);
        let mut input_ids = ids[..real_len].to_vec();

        if input_ids.len() >= max_len {
            input_ids.truncate(max_len - 1);
            input_ids.push(eos_id);
        } else {
            input_ids.resize(max_len, pad_id);
        }

        let mut attention_mask = vec![1u8; real_len];
        attention_mask.resize(max_len, 0u8);

        Encoding {
            input_ids,
            attention_mask,
        }
    }

    pub fn bos_id(&self) -> usize {
        *self.vocab.get(Self::BOS).unwrap_or(&1)
    }

    pub fn eos_id(&self) -> usize {
        *self.vocab.get(Self::EOS).unwrap_or(&2)
    }
}

impl Checkpointable for Tokenizer {
    fn to_weight_map(&self) -> WeightMap {
        let mut map = WeightMap::new();
        map.insert_string("id_to_token", self.id_to_token.clone());
        map.insert_scalar("unk_id", self.unk_id as u64);
        map
    }
}
