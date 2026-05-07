use std::collections::HashMap;

const WORD_END: &str = "</w>";

pub struct BpeTokenizer {
    id_to_token: Vec<String>,
    token_to_id: HashMap<String, usize>,
    merges: Vec<(String, String)>,
    merge_rank: HashMap<(String, String), usize>,
    unk_id: usize,
}

pub struct Encoding {
    pub input_ids: Vec<usize>,
    pub attention_mask: Vec<u8>,
}

type Vocab = HashMap<Vec<String>, usize>;

/// テキストを小文字化して単語頻度を集計
fn build_word_freq(corpus: &[&str]) -> HashMap<String, usize> {
    let mut freq: HashMap<String, usize> = HashMap::new();
    for text in corpus {
        for raw in text.split_whitespace() {
            let word: String = raw
                .to_lowercase()
                .chars()
                .filter(|c| c.is_alphabetic())
                .collect();
            if !word.is_empty() {
                *freq.entry(word).or_insert(0) += 1;
            }
        }
    }
    freq
}

/// 単語を文字 + </w> 列に分解
/// "cat" → ["c", "a", "t</w>"]
fn split_word(word: &str) -> Vec<String> {
    let chars: Vec<char> = word.chars().collect();
    let last = chars.len().saturating_sub(1);
    chars
        .iter()
        .enumerate()
        .map(|(i, &c)| {
            if i == last {
                format!("{}{}", c, WORD_END)
            } else {
                c.to_string()
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

        let word_freq = build_word_freq(corpus);
        let mut vocab: Vocab = word_freq
            .iter()
            .map(|(word, &freq)| (split_word(word), freq))
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
        let mut symbols = split_word(word);
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
            let word: String = raw
                .to_lowercase()
                .chars()
                .filter(|c| c.is_alphabetic())
                .collect();
            if word.is_empty() {
                continue;
            }
            for sym in self.encode_word(&word) {
                ids.push(*self.token_to_id.get(&sym).unwrap_or(&self.unk_id));
            }
        }
        ids.push(eos);
        ids
    }

    pub fn encode_simple(&self, text: &str) -> Vec<usize> {
        self.encode(text)
    }

    pub fn decord(&self, ids: &[usize]) -> String {
        let specials = [Self::PAD, Self::UNK, Self::BOS, Self::EOS];
        let mut out = String::new();
        for &id in ids {
            let Some(tok) = self.id_to_token.get(id) else {
                continue;
            };
            if specials.contains(&tok.as_str()) {
                continue;
            }
            if tok.ends_with(WORD_END) {
                out.push_str(tok.trim_end_matches(WORD_END));
                out.push(' ');
            } else {
                out.push_str(tok);
            }
        }
        out.trim_end().to_string()
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }
    fn bos_id(&self) -> usize {
        *self.token_to_id.get(Self::BOS).unwrap_or(&2)
    }

    fn eos_id(&self) -> usize {
        *self.token_to_id.get(Self::EOS).unwrap_or(&3)
    }
}
