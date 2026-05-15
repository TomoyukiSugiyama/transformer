//! Phase 7-1-C: クリーンコーパスで CharBPE を再訓練 + special token を追加するスクリプト。
//!
//! Phase 7-1-B 第 1-10 弾のクレンジング (章番号行 1,764 行削除 + 編集者注釈 33 個除去) 完了済の
//! `corpus/aozora_meiji_taisho_v2.txt` から CharBPE を 1 から訓練し、 末尾で 10 個の special token を追加する。
//!
//! 出力: `tokenizers/charbpe_v8010_aozora_meiji_taisho_v2_s500000.bin`
//!
//! 追加する special token (10 個):
//!   - <TITLE>, </TITLE>
//!   - <DRAMA>, </DRAMA>
//!   - <AUTHOR=夏目漱石>, <AUTHOR=太宰治>, <AUTHOR=森鴎外>,
//!     <AUTHOR=宮沢賢治>, <AUTHOR=中島敦>, <AUTHOR=国木田独歩>
//!   (BOS / EOS / PAD / UNK は既登録)
//!
//! 実行: cargo run --release --bin extend_tokenizer

use std::path::Path;
use std::time::Instant;

use transformer::char_bpe_tokenizer::CharBpeTokenizer;
use transformer::tokenizer::{Tokenizer, save_tokenizer_to_file};

const CORPUS_PATH: &str = "corpus/aozora_meiji_taisho_v2.txt";
/// BPE merge 学習に使う先頭サンプル数 (main.rs の Phase 7-a config と一致させる)。
const SAMPLE_CHARS: usize = 500_000;
/// BPE merges 部分の vocab (special token 追加で +10 → 8010)。
const BPE_VOCAB: usize = 8000;
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
    if !Path::new(CORPUS_PATH).exists() {
        eprintln!("ERROR: コーパスが見つかりません: {CORPUS_PATH}");
        std::process::exit(1);
    }

    println!("# クリーン v2 コーパスから CharBPE を訓練 + special token 付与");
    println!("# corpus: {CORPUS_PATH}");
    let corpus = std::fs::read_to_string(CORPUS_PATH)?;
    let total_chars = corpus.chars().count();
    println!("# 全 char 数: {total_chars}");
    println!(
        "# BPE 訓練: 先頭 {SAMPLE_CHARS} char サンプル → 目標 vocab={BPE_VOCAB} \
         (全コーパスで coverage 保証)"
    );

    let sample_text: String = corpus.chars().take(SAMPLE_CHARS).collect();
    let t0 = Instant::now();
    let mut tokenizer = CharBpeTokenizer::train_with_coverage(&sample_text, &corpus, BPE_VOCAB);
    println!(
        "# BPE 訓練完了: vocab={}, 経過 {:.1}s",
        <CharBpeTokenizer as Tokenizer>::vocab_size(&tokenizer),
        t0.elapsed().as_secs_f32()
    );

    let initial_vocab = <CharBpeTokenizer as Tokenizer>::vocab_size(&tokenizer);
    println!();
    println!(
        "# 追加する special token ({} 個):",
        NEW_SPECIAL_TOKENS.len()
    );
    for tok in NEW_SPECIAL_TOKENS {
        let id = tokenizer.add_special_token(tok);
        println!("  id={id:>5}  {tok}");
    }
    let final_vocab = <CharBpeTokenizer as Tokenizer>::vocab_size(&tokenizer);
    println!(
        "# 拡張後 vocab: {final_vocab} (+{})",
        final_vocab - initial_vocab
    );

    save_tokenizer_to_file(&tokenizer, OUTPUT_PATH)?;
    println!();
    println!("→ 保存: {OUTPUT_PATH}");

    // 動作確認: 先頭の <BOS><AUTHOR=...><TITLE>...</TITLE> が special token として認識されるか
    let head: String = corpus.chars().take(200).collect();
    let ids = tokenizer.encode_long(&head);
    println!();
    println!("--- 動作確認: 新コーパス先頭 200 char の encode ---");
    println!(
        "入力 (先頭 100 char): {}",
        &head.chars().take(100).collect::<String>()
    );
    println!("ID 列 (先頭 20): {:?}", &ids[..20.min(ids.len())]);
    let bos_id = tokenizer.bos_id();
    println!("BOS id={bos_id}  ids[0]={}", ids[0]);
    if ids[0] == bos_id && ids.get(1).copied().unwrap_or(0) >= initial_vocab {
        println!("✓ <BOS> + <AUTHOR=...> が special token として認識されています");
    } else {
        eprintln!("✗ 期待と異なる: BOS + special token が連続していません");
    }

    Ok(())
}
