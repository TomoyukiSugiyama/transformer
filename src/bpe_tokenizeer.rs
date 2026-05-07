use std::{
    collections::HashMap,
    io::{Error, ErrorKind, Result},
};

use crate::checkpoint::{Checkpointable, WeightMap};

const WORD_END: &str = "</w>";

pub struct BpeTokenizer {
    id_to_token: Vec<String>,
    token_to_id: HashMap<String, usize>,
    merges: Vec<(String, String)>,
    merge_rank: HashMap<(String, String), usize>,
    unk_id: usize,
}

type Vocab = HashMap<Vec<String>, usize>;

/// テキストを小文字化して単語頻度を集計
fn build_word_freq(corpus: &[&str]) -> HashMap<String, usize> {
    let mut freq: HashMap<String, usize> = HashMap::new();
    for text in corpus {
        for raw in text.split_whitespace() {
            let word: String = raw.to_lowercase().chars().collect();
            if !word.is_empty() {
                *freq.entry(word).or_insert(0) += 1;
            }
        }
    }
    freq
}

fn byte_token(b: u8) -> String {
    format!("b{:03}", b)
}
/// 単語を文字 + </w> 列に分解
/// "b099b097b116" → ["b099", "b097", "b116</w>"]
fn split_word_bytes(word: &str) -> Vec<String> {
    let bytes = word.as_bytes();
    let last = bytes.len().saturating_sub(1);
    bytes
        .iter()
        .enumerate()
        .map(|(i, &b)| {
            if i == last {
                format!("{}{}", byte_token(b), WORD_END)
            } else {
                byte_token(b)
            }
        })
        .collect()
}

/// 隣接ペアの頻度を集計
fn count_pairs(vocab: &Vocab) -> HashMap<(String, String), usize> {
    let mut pairs: HashMap<(String, String), usize> = HashMap::new();
    for (symbol, &freq) in vocab {
        for w in symbol.windows(2) {
            *pairs.entry((w[0].clone(), w[1].clone())).or_insert(0) += freq;
        }
    }
    pairs
}

/// 最頻出ペアを返す
fn best_pair(pairs: &HashMap<(String, String), usize>) -> Option<(String, String)> {
    pairs.iter().max_by_key(|(_, v)| *v).map(|(k, _)| k.clone())
}

/// vocab 内の (a, b) を "ab" にマージ
fn merge_vocab(vocab: Vocab, a: &str, b: &str) -> Vocab {
    let merged = format!("{}{}", a, b);
    vocab
        .into_iter()
        .map(|(symbols, freq)| {
            let mut new: Vec<String> = Vec::with_capacity(symbols.len());
            let mut i = 0;
            while i < symbols.len() {
                if i + 1 < symbols.len() && symbols[i] == a && symbols[i + 1] == b {
                    new.push(merged.clone());
                    i += 2;
                } else {
                    new.push(symbols[i].clone());
                    i += 1;
                }
            }
            (new, freq)
        })
        .collect()
}

impl BpeTokenizer {
    pub const PAD: &'static str = "<PAD>";
    pub const UNK: &'static str = "<UNK>";
    pub const BOS: &'static str = "<BOS>";
    pub const EOS: &'static str = "<EOS>";

    pub fn train(corpus: &[&str], vocab_size: usize) -> Self {
        let mut token_to_id: HashMap<String, usize> = HashMap::new();
        for special in [Self::PAD, Self::UNK, Self::BOS, Self::EOS] {
            token_to_id.insert(special.to_string(), token_to_id.len());
        }
        for b in 0..=255 {
            token_to_id.insert(byte_token(b), token_to_id.len());
            token_to_id.insert(format!("{}{}", byte_token(b), WORD_END), token_to_id.len());
        }
        let word_freq = build_word_freq(corpus);
        let mut vocab: Vocab = word_freq
            .iter()
            .map(|(word, &freq)| (split_word_bytes(word), freq))
            .collect();

        for symbols in vocab.keys() {
            for sym in symbols {
                if !token_to_id.contains_key(sym) {
                    token_to_id.insert(sym.clone(), token_to_id.len());
                }
            }
        }

        let mut merges: Vec<(String, String)> = Vec::new();
        let mut merge_rank: HashMap<(String, String), usize> = HashMap::new();

        while token_to_id.len() < vocab_size {
            let pairs = count_pairs(&vocab);
            if pairs.is_empty() {
                break;
            }
            let Some((a, b)) = best_pair(&pairs) else {
                break;
            };

            let merged = format!("{}{}", a, b);
            merge_rank.insert((a.clone(), b.clone()), merges.len());
            merges.push((a.clone(), b.clone()));

            if !token_to_id.contains_key(&merged) {
                token_to_id.insert(merged, token_to_id.len());
            }

            vocab = merge_vocab(vocab, &a, &b);
        }

        let mut id_to_token = vec![String::new(); token_to_id.len()];
        for (tok, &id) in &token_to_id {
            id_to_token[id] = tok.clone();
        }
        let unk_id: usize = *token_to_id.get(Self::UNK).unwrap();
        Self {
            id_to_token,
            token_to_id,
            merges,
            merge_rank,
            unk_id,
        }
    }

    /// 1単語に学習済みマージを適用して subword 列を返す
    fn encode_word(&self, word: &str) -> Vec<String> {
        let mut symbols = split_word_bytes(word);
        loop {
            // ランクが最小（最初にマージされた）ペアを探す
            let best = symbols
                .windows(2)
                .enumerate()
                .filter_map(|(i, w)| {
                    let pair = (w[0].clone(), w[1].clone());
                    self.merge_rank.get(&pair).map(|&rank| (i, rank))
                })
                .min_by_key(|&(_, rank)| rank);

            let Some((i, _)) = best else { break };
            let merged = format!("{}{}", symbols[i], symbols[i + 1]);
            symbols.remove(i + 1);
            symbols[i] = merged;
        }
        symbols
    }
    fn encode(&self, text: &str) -> Vec<usize> {
        let bos = self.bos_id();
        let eos = self.eos_id();
        let mut ids = vec![bos];
        for raw in text.split_whitespace() {
            let word = raw.trim_matches(|c: char| c.is_ascii_punctuation());
            if word.is_empty() {
                continue;
            }
            for sym in self.encode_word(word) {
                ids.push(*self.token_to_id.get(&sym).unwrap_or(&self.unk_id));
            }
        }
        ids.push(eos);
        ids
    }

    pub fn encode_simple(&self, text: &str) -> Vec<usize> {
        self.encode(text)
    }

    pub fn encode_prompt(&self, text: &str) -> Vec<usize> {
        let mut ids = self.encode(text);
        if ids.last() == Some(&self.eos_id()) {
            ids.pop();
        }
        ids
    }

    fn decode_token_to_bytes(tok: &str) -> Vec<u8> {
        let mut bytes = Vec::new();
        let mut s = tok;
        while s.len() >= 4 && s.starts_with('b') {
            if let Ok(n) = s[1..4].parse::<u8>() {
                bytes.push(n);
                s = &s[4..]
            } else {
                break;
            }
        }
        bytes
    }
    pub fn decord(&self, ids: &[usize]) -> String {
        let specials = [Self::PAD, Self::UNK, Self::BOS, Self::EOS];
        let mut words: Vec<String> = Vec::new();
        let mut current: Vec<u8> = Vec::new();

        for &id in ids {
            let Some(tok) = self.id_to_token.get(id) else {
                continue;
            };
            if specials.contains(&tok.as_str()) {
                continue;
            }
            let (tok_body, is_end) = if tok.ends_with(WORD_END) {
                (tok.trim_end_matches(WORD_END), true)
            } else {
                (tok.as_str(), false)
            };
            current.extend(Self::decode_token_to_bytes(tok_body));
            if is_end {
                words.push(String::from_utf8_lossy(&current).into_owned());
                current.clear();
            }
        }

        if !current.is_empty() {
            words.push(String::from_utf8_lossy(&current).into_owned());
        }
        words.join(" ")
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    pub fn pad_id(&self) -> usize {
        *self.token_to_id.get(Self::PAD).unwrap_or(&0)
    }

    pub fn bos_id(&self) -> usize {
        *self.token_to_id.get(Self::BOS).unwrap_or(&2)
    }

    pub fn eos_id(&self) -> usize {
        *self.token_to_id.get(Self::EOS).unwrap_or(&3)
    }

    pub fn empty() -> Self {
        Self {
            id_to_token: Vec::new(),
            token_to_id: HashMap::new(),
            merges: Vec::new(),
            merge_rank: HashMap::new(),
            unk_id: 0,
        }
    }
}

impl Checkpointable for BpeTokenizer {
    fn to_weight_map(&self) -> WeightMap {
        let mut map = WeightMap::new();
        map.insert_strings("id_to_token", self.id_to_token.clone());
        map.insert_scalar("unk_id", self.unk_id as u64);

        let flat_merges: Vec<String> = self
            .merges
            .iter()
            .flat_map(|(a, b)| [a.clone(), b.clone()])
            .collect();
        map.insert_strings("merges", flat_merges);

        map
    }

    fn from_weight_map(&mut self, map: &WeightMap) -> Result<()> {
        let id_to_token = map.get_strings("id_to_token")?.clone();
        let token_to_id: HashMap<String, usize> = id_to_token
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, t)| (t, i))
            .collect();

        let unk_id = map.get_scalar("unk_id")? as usize;

        let flat = map.get_strings("merges")?;
        if flat.len() % 2 != 0 {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "merges length must be even",
            ));
        }
        let merges: Vec<(String, String)> = flat
            .chunks(2)
            .map(|w| (w[0].clone(), w[1].clone()))
            .collect();
        let merge_rank: HashMap<(String, String), usize> = merges
            .iter()
            .enumerate()
            .map(|(i, pair)| (pair.clone(), i))
            .collect();

        self.id_to_token = id_to_token;
        self.token_to_id = token_to_id;
        self.merges = merges;
        self.merge_rank = merge_rank;
        self.unk_id = unk_id;

        Ok(())
    }
}
