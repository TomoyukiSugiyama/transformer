//! Phase 7-1-C: 既存 CharBPE トークナイザに special token を追加するスクリプト。
//!
//! 入力: `tokenizers/charbpe_v8000_aozora_meiji_taisho_s500000.bin`
//! 出力: `tokenizers/charbpe_v8012_aozora_meiji_taisho_s500000_v2.bin`
//!
//! 追加する special token (12 個):
//!   - <TITLE>, </TITLE>
//!   - <DRAMA>, </DRAMA>
//!   - <AUTHOR=夏目漱石>, <AUTHOR=太宰治>, <AUTHOR=森鴎外>,
//!     <AUTHOR=宮沢賢治>, <AUTHOR=中島敦>, <AUTHOR=国木田独歩>
//!   (BOS / EOS は既登録)
//!
//! Phase 7-1-D で main.rs から `aozora_meiji_taisho_charbpe8k_max1024_wsd_v2()` config を呼ぶと、
//! 自動的にこの cache を読みに行く。
//!
//! 実行: cargo run --release --bin extend_tokenizer

use std::path::Path;

use transformer::char_bpe_tokenizer::CharBpeTokenizer;
use transformer::checkpoint::{Checkpointable, WeightMap};
use transformer::tokenizer::save_tokenizer_to_file;

const INPUT_PATH: &str = "tokenizers/charbpe_v8000_aozora_meiji_taisho_s500000.bin";
// 出力名は main.rs::tokenizer_cache_path() の自動命名規則 (charbpe_v{vocab}_{corpus_stem}_s{sample}.bin)
// と整合させる: corpus_v2 + vocab=8010 (= 8000 + 10 special tokens) + sample=500000
const OUTPUT_PATH: &str = "tokenizers/charbpe_v8010_aozora_meiji_taisho_v2_s500000.bin";

const NEW_SPECIAL_TOKENS: &[&str] = &[
    "<TITLE>",
    "</TITLE>",
    "<DRAMA>",
    "</DRAMA>",
    "<AUTHOR=夏目漱石>",
    "<AUTHOR=太宰治>",
    "<AUTHOR=森鴎外>",
    "<AUTHOR=宮沢賢治>",
    "<AUTHOR=中島敦>",
    "<AUTHOR=国木田独歩>",
];

fn main() -> std::io::Result<()> {
    if !Path::new(INPUT_PATH).exists() {
        eprintln!("ERROR: 入力 cache が見つかりません: {INPUT_PATH}");
        std::process::exit(1);
    }

    println!("# 入力 cache: {INPUT_PATH}");
    let map = WeightMap::load(INPUT_PATH)?;
    let mut tokenizer = CharBpeTokenizer::empty();
    Checkpointable::from_weight_map(&mut tokenizer, &map)?;

    let initial_vocab = tokenizer.vocab_size();
    let initial_specials = tokenizer.special_tokens().len();
    println!(
        "# 既存 vocab: {initial_vocab}, special_tokens: {initial_specials} \
         ({})",
        tokenizer
            .special_tokens()
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );

    println!();
    println!("# 追加する special token ({} 個):", NEW_SPECIAL_TOKENS.len());
    for tok in NEW_SPECIAL_TOKENS {
        let id = tokenizer.add_special_token(tok);
        println!("  id={id:>5}  {tok}");
    }

    let final_vocab = tokenizer.vocab_size();
    let final_specials = tokenizer.special_tokens().len();
    println!();
    println!("# 拡張後 vocab: {final_vocab} (+{})", final_vocab - initial_vocab);
    println!("# 拡張後 special_tokens: {final_specials}");

    save_tokenizer_to_file(&tokenizer, OUTPUT_PATH)?;
    println!();
    println!("→ 保存: {OUTPUT_PATH}");

    // 動作確認: 新コーパスの先頭 1KB を encode して、 special token が 1 ID として
    // 認識されているか確認する。
    let sample_corpus_path = "corpus/aozora_meiji_taisho_v2.txt";
    if Path::new(sample_corpus_path).exists() {
        let sample = std::fs::read_to_string(sample_corpus_path)?;
        let head: String = sample.chars().take(200).collect();
        let ids = tokenizer.encode_long(&head);
        println!();
        println!("--- 動作確認: 新コーパス先頭 200 char の encode ---");
        println!("入力 (先頭 100 char): {}", &head.chars().take(100).collect::<String>());
        println!("ID 列 (先頭 20): {:?}", &ids[..20.min(ids.len())]);
        // BOS の直後に <AUTHOR=国木田独歩> が来ているはず
        let bos_id = tokenizer.bos_id();
        let author_id = *tokenizer
            .special_tokens()
            .iter()
            .position(|s| s == "<AUTHOR=国木田独歩>")
            .map(|_| {
                // ID は order 通り。 PAD/UNK/BOS/EOS=0..3, 8000 char/merges, +TITLE/etc
                // ここでは encode 後の値を直接確認するのが楽
                &ids[1]
            })
            .unwrap_or(&0);
        println!("BOS id={bos_id}  ids[0]={}  ids[1]={author_id}", ids[0]);
        if ids[0] == bos_id && ids.get(1).copied().unwrap_or(0) > initial_vocab {
            println!("✓ <BOS> + <AUTHOR=...> が special token として認識されています");
        } else {
            eprintln!("✗ 期待と異なる: BOS + special token が連続していません");
        }
    }

    Ok(())
}
