use std::collections::HashMap;
pub struct Tokenizer {
    vocab: HashMap<String, usize>,
    id_to_token: Vec<String>,
    unk_id: usize,
}

impl Tokenizer {
    pub const UNK: &'static str = "<UNK>";

    pub fn build(corpus: &[&str]) -> Self {
        let mut vocab: HashMap<String, usize> = HashMap::new();

        for special in [Self::UNK] {
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

    pub fn id_to_token_str(&self,id:usize) -> Option<&str>{
        self.id_to_token.get(id).map(String::as_str)
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut ids = vec![];
        let tokens = Self::tokenize_text(text);

        for token in tokens {
            ids.push(self.vocab.get(&token).copied().unwrap_or(self.unk_id));
        }
        ids
    }
}
