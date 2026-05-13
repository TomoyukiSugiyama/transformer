//! Phase 8-1 [C]: Aozora + Wikipedia 混合コーパス作成。
//!
//! 入力:
//! - `corpus/aozora_meiji_taisho_v2.txt` (Phase 7-1 で生成、 ~8.28M chars)
//! - `corpus/wikipedia_ja.txt` (Phase 8-1 [B] で生成、 ~974.7M chars)
//!
//! 出力: `corpus/aozora_wikipedia_mixed.txt` (~983M chars)
//!
//! Aozora を `REPEAT_AOZORA` 倍に複製してから先頭に連結し、 残りを Wikipedia で埋める。
//! 区切りは `\n\n` (1 空行) で、 学習時のランダム窓サンプリングで自然に混在する。
//!
//! 注: 学習側は `chunk_len` 単位のスライディングウィンドウなので、
//! 入力テキスト中の Aozora 部分と Wikipedia 部分の境界は問題にならない。
//! Aozora の `<BOS><AUTHOR=...><TITLE>...</TITLE>` special token も維持される。

use std::fs;
use std::io::{BufWriter, Write};
use std::time::Instant;

const AOZORA_PATH: &str = "corpus/aozora_meiji_taisho_v2.txt";
const WIKI_PATH: &str = "corpus/wikipedia_ja.txt";
const OUTPUT_PATH: &str = "corpus/aozora_wikipedia_mixed.txt";
const SEPARATOR: &str = "\n\n";
const REPEAT_AOZORA: usize = 1;

fn main() -> std::io::Result<()> {
    let t = Instant::now();
    println!("# mix_corpus: {AOZORA_PATH} (x{REPEAT_AOZORA}) + {WIKI_PATH} -> {OUTPUT_PATH}");

    let aozora = fs::read_to_string(AOZORA_PATH)?;
    let wiki = fs::read_to_string(WIKI_PATH)?;
    let aozora_chars = aozora.chars().count();
    let wiki_chars = wiki.chars().count();
    let aozora_bytes = aozora.len();
    let wiki_bytes = wiki.len();
    println!(
        "# input aozora    : {} chars ({:.2} MB)",
        aozora_chars,
        aozora_bytes as f64 / 1e6
    );
    println!(
        "# input wikipedia : {} chars ({:.2} GB)",
        wiki_chars,
        wiki_bytes as f64 / 1e9
    );

    let out_file = fs::File::create(OUTPUT_PATH)?;
    let mut writer = BufWriter::with_capacity(8 * 1024 * 1024, out_file);

    let sep_chars = SEPARATOR.chars().count();
    let mut total_chars = 0usize;
    for i in 0..REPEAT_AOZORA {
        writer.write_all(aozora.as_bytes())?;
        writer.write_all(SEPARATOR.as_bytes())?;
        total_chars += aozora_chars + sep_chars;
        if REPEAT_AOZORA > 1 {
            println!("#   wrote aozora pass {}/{}", i + 1, REPEAT_AOZORA);
        }
    }
    writer.write_all(wiki.as_bytes())?;
    total_chars += wiki_chars;
    writer.flush()?;

    println!("# output {OUTPUT_PATH}:");
    println!("#   total chars   : {}", total_chars);
    let aozora_total = REPEAT_AOZORA * aozora_chars + REPEAT_AOZORA * sep_chars;
    println!(
        "#   aozora share  : {:.3}% ({} chars, repeat {}x)",
        aozora_total as f64 / total_chars as f64 * 100.0,
        REPEAT_AOZORA * aozora_chars,
        REPEAT_AOZORA,
    );
    println!(
        "#   wikipedia share: {:.3}% ({} chars)",
        wiki_chars as f64 / total_chars as f64 * 100.0,
        wiki_chars,
    );
    println!("# elapsed: {:.1}s", t.elapsed().as_secs_f32());
    Ok(())
}
