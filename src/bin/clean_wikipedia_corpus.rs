//! Phase 8-1 [B]: `corpus/wikipedia_ja_raw.txt` をクレンジングして
//! `corpus/wikipedia_ja.txt` を生成する。
//!
//! 入力フォーマット: 記事を `\n\n<DOC_SEP>\n\n` で連結したテキスト。
//!
//! クレンジングルール:
//! 1. 末尾の reference 系セクション (`脚注`, `注釈`, `出典`, `関連項目`, `外部リンク`,
//!    `参考文献`, `関連書籍`, `参照`, `補注`, `脚注・参考文献`, `参考`) 以降を切り捨て。
//!    自然文の prose に集中させるため。
//! 2. 各記事内の連続空行を 1 行に正規化。
//! 3. 行末の trailing whitespace を除去。
//! 4. 記事末尾の連続空行を全削除。
//! 5. 短い記事 (< MIN_CHARS) を破棄。
//! 6. 記事の区切りは `\n\n` (1 空行) を使う。 連続改行は不要。
//!
//! 実行: cargo run --release --bin clean_wikipedia_corpus

use std::fs::File;
use std::io::{BufWriter, Write};

const INPUT_PATH: &str = "corpus/wikipedia_ja_raw.txt";
const OUTPUT_PATH: &str = "corpus/wikipedia_ja.txt";
const DOC_SEP: &str = "\n\n<DOC_SEP>\n\n";
const MIN_CHARS: usize = 200;

const TRAILING_SECTIONS: &[&str] = &[
    "脚注",
    "注釈",
    "出典",
    "関連項目",
    "外部リンク",
    "参考文献",
    "関連書籍",
    "参照",
    "補注",
    "脚注・参考文献",
    "参考",
    "脚注・出典",
    "注",
    "出典・脚注",
];

fn main() -> std::io::Result<()> {
    println!("# clean_wikipedia_corpus: {INPUT_PATH} -> {OUTPUT_PATH}");
    let raw = std::fs::read_to_string(INPUT_PATH)?;
    let in_chars = raw.chars().count();
    let in_articles: Vec<&str> = raw.split(DOC_SEP).collect();
    let in_articles_n = in_articles.len();
    println!("# input: {} chars, {} articles", in_chars, in_articles_n);

    let mut out_articles: Vec<String> = Vec::with_capacity(in_articles_n);
    let mut dropped_short = 0usize;
    let mut truncated_at_section = 0usize;

    for art in in_articles {
        let cleaned = clean_article(art, &mut truncated_at_section);
        if cleaned.chars().count() >= MIN_CHARS {
            out_articles.push(cleaned);
        } else {
            dropped_short += 1;
        }
    }

    let out = File::create(OUTPUT_PATH)?;
    let mut writer = BufWriter::with_capacity(8 * 1024 * 1024, out);
    let mut out_chars = 0usize;
    for (i, art) in out_articles.iter().enumerate() {
        if i > 0 {
            writer.write_all(b"\n\n")?;
            out_chars += 2;
        }
        writer.write_all(art.as_bytes())?;
        out_chars += art.chars().count();
    }
    writer.flush()?;

    println!(
        "# output: {} chars, {} articles (input -> output: chars {:.1}%, articles {:.1}%)",
        out_chars,
        out_articles.len(),
        out_chars as f64 / in_chars as f64 * 100.0,
        out_articles.len() as f64 / in_articles_n as f64 * 100.0,
    );
    println!("# dropped short articles (< {MIN_CHARS} chars): {dropped_short}");
    println!("# articles truncated at trailing section: {truncated_at_section}");
    Ok(())
}

fn clean_article(art: &str, truncated_count: &mut usize) -> String {
    let lines: Vec<&str> = art.lines().collect();

    let mut cut_at: Option<usize> = None;
    for (i, line) in lines.iter().enumerate() {
        let s = line.trim();
        if TRAILING_SECTIONS.contains(&s) {
            cut_at = Some(i);
            break;
        }
    }
    let kept = match cut_at {
        Some(idx) => {
            *truncated_count += 1;
            &lines[..idx]
        }
        None => &lines[..],
    };

    let mut out = String::with_capacity(art.len());
    let mut prev_blank = false;
    let mut first = true;
    for line in kept {
        let trimmed = line.trim_end();
        let is_blank = trimmed.is_empty();
        if is_blank {
            if prev_blank || first {
                continue;
            }
            out.push('\n');
            prev_blank = true;
        } else {
            if !first {
                out.push('\n');
            }
            out.push_str(trimmed);
            prev_blank = false;
        }
        first = false;
    }
    while out.ends_with('\n') {
        out.pop();
    }
    out
}
