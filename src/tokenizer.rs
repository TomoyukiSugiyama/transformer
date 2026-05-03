use std::collections::HashMap;
pub struct Tokenizer {
    vocab: HashMap<String, usize>,
}

impl Tokenizer {
    pub fn build(corpus: &[&str]) -> Self {
        let mut vocab: HashMap<String, usize> = HashMap::new();
        for text in corpus {
            let tokens = Self::tokenize_text(text);
            for token in tokens {
                if !vocab.contains_key(&token) {
                    vocab.insert(token, vocab.len());
                }
            }
        }
        println!("{:?}",vocab);
        Self { vocab }
    }

    fn tokenize_text(text: &str) -> Vec<String> {
        let text = text.to_lowercase();
        let mut tokens = Vec::new();
        let mut current = String::new();
        for ch in text.chars() {
            if ch.is_alphanumeric() || ch == '\'' {
                current.push(ch);
            }else {
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
        println!("{:?}",tokens);
        tokens
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}
