mod tokenizer;
use crate::tokenizer::*;

fn main() {
    let corpus = vec!["Hello, world!", "Attention Is All You Need"];

    let tokenizer = Tokenizer::build(&corpus);
    println!("Vocab Size: {}", tokenizer.vocab_size());
}
