//! Phase 8-1 [D]: 混合コーパス (Aozora + Wikipedia 日本語版) で CharBPE を vocab 32K で再訓練し、
//! Phase 7-1 と同じ 10 個の special token を末尾追加する。
//!
//! 出力: `tokenizers/charbpe_v32010_aozora_wikipedia.bin` (vocab 32,000 + 10 special token = 32,010)
//!
//! - **coverage_text**: `corpus/aozora_wikipedia_mixed.txt` 全体 (983M chars)
//!   → 全 unique char (~15.8K) を初期 vocab に登録
//! - **merge_text**: Aozora 先頭 500K chars + Wikipedia 先頭 1.5M chars = 2M chars stratified sample
//!   → BPE merge を学習 (Aozora の literary 語彙 + Wikipedia の一般語彙の両方をカバー)
//!
//! 実行: cargo run --release --bin train_tokenizer_phase8

use std::path::Path;
use std::time::Instant;

use transformer::char_bpe_tokenizer::CharBpeTokenizer;
use transformer::tokenizer::{Tokenizer, save_tokenizer_to_file};

const AOZORA_PATH: &str = "corpus/aozora_meiji_taisho_v2.txt";
const WIKI_PATH: &str = "corpus/wikipedia_ja.txt";
const MIXED_PATH: &str = "corpus/aozora_wikipedia_mixed.txt";

/// stratified sample: Aozora 部分の先頭から
const AOZORA_SAMPLE_CHARS: usize = 500_000;
/// stratified sample: Wikipedia 部分の先頭から
const WIKI_SAMPLE_CHARS: usize = 1_500_000;
/// 目標 BPE vocab size (special token 追加で +10 → 32,010)
const BPE_VOCAB: usize = 32_000;
const OUTPUT_PATH: &str = "tokenizers/charbpe_v32010_aozora_wikipedia.bin";

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
    for path in [AOZORA_PATH, WIKI_PATH, MIXED_PATH] {
        if !Path::new(path).exists() {
            eprintln!("ERROR: ファイルが見つかりません: {path}");
            std::process::exit(1);
        }
    }

    println!("# Phase 8-1 [D]: 混合コーパスで CharBPE 32K 再訓練");
    println!("# coverage_text  : {MIXED_PATH}");
    println!(
        "# merge_text     : Aozora 先頭 {AOZORA_SAMPLE_CHARS} + Wikipedia 先頭 {WIKI_SAMPLE_CHARS} chars (stratified)"
    );
    println!(
        "# target BPE vocab: {BPE_VOCAB}  (+ {} special token → 32,010)",
        NEW_SPECIAL_TOKENS.len()
    );
    println!();

    // [1] coverage text の読込
    let t_io = Instant::now();
    let mixed = std::fs::read_to_string(MIXED_PATH)?;
    let mixed_chars = mixed.chars().count();
    println!(
        "# loaded mixed corpus: {} chars in {:.1}s",
        mixed_chars,
        t_io.elapsed().as_secs_f32()
    );

    // [2] stratified merge_text を構築
    let t_io2 = Instant::now();
    let aozora = std::fs::read_to_string(AOZORA_PATH)?;
    let wiki = std::fs::read_to_string(WIKI_PATH)?;
    let aozora_sample: String = aozora.chars().take(AOZORA_SAMPLE_CHARS).collect();
    let wiki_sample: String = wiki.chars().take(WIKI_SAMPLE_CHARS).collect();
    let merge_text = format!("{aozora_sample}\n\n{wiki_sample}");
    println!(
        "# built merge_text : {} chars (Aozora {} + Wikipedia {} + separator) in {:.1}s",
        merge_text.chars().count(),
        aozora_sample.chars().count(),
        wiki_sample.chars().count(),
        t_io2.elapsed().as_secs_f32()
    );

    // [3] BPE 訓練
    println!();
    println!("# BPE 訓練開始...");
    let t_train = Instant::now();
    let mut tokenizer = CharBpeTokenizer::train_with_coverage(&merge_text, &mixed, BPE_VOCAB);
    let train_secs = t_train.elapsed().as_secs_f32();
    let initial_vocab = <CharBpeTokenizer as Tokenizer>::vocab_size(&tokenizer);
    println!(
        "# BPE 訓練完了: vocab={}, 経過 {:.1}s ({:.1} min)",
        initial_vocab,
        train_secs,
        train_secs / 60.0
    );

    // [4] special token 追加
    println!();
    println!("# 追加する special token ({} 個):", NEW_SPECIAL_TOKENS.len());
    for tok in NEW_SPECIAL_TOKENS {
        let id = tokenizer.add_special_token(tok);
        println!("  id={id:>5}  {tok}");
    }
    let final_vocab = <CharBpeTokenizer as Tokenizer>::vocab_size(&tokenizer);
    println!(
        "# 拡張後 vocab: {final_vocab} (+{})",
        final_vocab - initial_vocab
    );

    // [5] 保存
    save_tokenizer_to_file(&tokenizer, OUTPUT_PATH)?;
    println!();
    println!("→ 保存: {OUTPUT_PATH}");

    // [6] 圧縮率の概算 (Aozora と Wikipedia それぞれ)
    println!();
    println!("# 圧縮率の概算 (chars/token):");
    let aozora_head: String = aozora.chars().take(50_000).collect();
    let wiki_head: String = wiki.chars().take(50_000).collect();
    let aozora_tokens = tokenizer.encode_long(&aozora_head);
    let wiki_tokens = tokenizer.encode_long(&wiki_head);
    let aozora_ratio = aozora_head.chars().count() as f64 / aozora_tokens.len() as f64;
    let wiki_ratio = wiki_head.chars().count() as f64 / wiki_tokens.len() as f64;
    println!(
        "  Aozora    (50K char sample): {} tokens → {:.3} chars/token",
        aozora_tokens.len(),
        aozora_ratio
    );
    println!(
        "  Wikipedia (50K char sample): {} tokens → {:.3} chars/token",
        wiki_tokens.len(),
        wiki_ratio
    );

    // [7] 動作確認: BOS + AUTHOR + TITLE が special token で認識されるか
    let head: String = mixed.chars().take(200).collect();
    let ids = tokenizer.encode_long(&head);
    println!();
    println!("--- 動作確認: 混合コーパス先頭 200 char の encode ---");
    println!(
        "入力 (先頭 80 char): {}",
        &head.chars().take(80).collect::<String>()
    );
    println!("ID 列 (先頭 12): {:?}", &ids[..12.min(ids.len())]);
    let bos_id = tokenizer.bos_id();
    println!("BOS id={bos_id}  ids[0]={}", ids[0]);
    if ids[0] == bos_id && ids.get(1).copied().unwrap_or(0) >= initial_vocab {
        println!("✓ <BOS> + <AUTHOR=...> が special token として認識されています");
    } else {
        eprintln!("✗ 期待と異なる: BOS + special token が連続していません");
    }

    Ok(())
}
