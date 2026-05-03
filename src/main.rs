mod tokenizer;
use crate::tokenizer::*;

fn main() {
    let corpus = vec!["Hello, world!", "Attention Is All You Need"];

    let tokenizer = Tokenizer::build(&corpus);
    println!("Corpus: {:?}",corpus);
    println!("Vocab Size: {}", tokenizer.vocab_size());

    let mut token_ids = tokenizer.encode("Hello, world!Attention Is All You Need");
    println!("Token Ids(Hello, world!Attention Is All You Need): {:?}", token_ids);

    token_ids = tokenizer.encode("unknown word");
    println!("Token Ids (Unknown Word): {:?}", token_ids);

}
