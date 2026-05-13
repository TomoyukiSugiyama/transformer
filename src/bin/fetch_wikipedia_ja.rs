//! Phase 8-1 [A]: HuggingFace `wikimedia/wikipedia` dataset から日本語版を取得し、
//! 各記事の `text` 列を `corpus/wikipedia_ja_raw.txt` に書き出す。
//!
//! - スナップショット: 20231101.ja (15 parquet ファイル、 合計 ~3.94 GB compressed)
//! - 1 ファイルずつ DL → arrow record_batch で text 列を抽出 → ファイルへ追記
//! - 累計 char 数が `TARGET_CHARS` を超えたら停止
//!
//! 出力フォーマット (記事区切り):
//! ```text
//! <記事 1 の text>
//!
//! <DOC_SEP>
//!
//! <記事 2 の text>
//! ...
//! ```
//!
//! `<DOC_SEP>` は後段クレンジングで使う目印トークン。 学習時には除去・置換する。

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Write};
use std::path::Path;
use std::time::Instant;

use arrow_array::{Array, RecordBatch, StringArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

const DATASET_BASE_URL: &str =
    "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.ja";
const NUM_FILES: usize = 15;
const TARGET_CHARS: usize = 1_000_000_000;
const OUTPUT_PATH: &str = "corpus/wikipedia_ja_raw.txt";
const DOC_SEP: &str = "\n\n<DOC_SEP>\n\n";
const TMP_DIR: &str = "corpus/_wiki_tmp";

fn main() -> std::io::Result<()> {
    println!("# fetch_wikipedia_ja: target={} chars from wikimedia/wikipedia 20231101.ja", TARGET_CHARS);
    std::fs::create_dir_all(TMP_DIR)?;
    if Path::new(OUTPUT_PATH).exists() {
        eprintln!("⚠ {OUTPUT_PATH} already exists. Aborting to avoid overwrite. Move it aside if you want to refresh.");
        std::process::exit(2);
    }
    let out_file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(OUTPUT_PATH)?;
    let mut out = BufWriter::with_capacity(8 * 1024 * 1024, out_file);

    let mut total_chars: usize = 0;
    let mut total_articles: usize = 0;
    let t_total = Instant::now();

    for idx in 0..NUM_FILES {
        let file_name = format!("train-{idx:05}-of-{NUM_FILES:05}.parquet");
        let url = format!("{DATASET_BASE_URL}/{file_name}");
        let local = format!("{TMP_DIR}/{file_name}");
        let t_file = Instant::now();
        if !Path::new(&local).exists() {
            println!("# [{idx}/{NUM_FILES}] downloading {file_name} ...");
            download(&url, &local)?;
            println!(
                "#   DL done: {} MB in {:.1}s",
                std::fs::metadata(&local)?.len() / 1_000_000,
                t_file.elapsed().as_secs_f32()
            );
        } else {
            println!("# [{idx}/{NUM_FILES}] cached parquet found at {local}");
        }
        let t_parse = Instant::now();
        let (file_chars, file_articles) = extract_text(&local, &mut out)?;
        total_chars += file_chars;
        total_articles += file_articles;
        println!(
            "#   parsed: +{} chars, +{} articles in {:.1}s   (cumulative {} chars / {} articles)",
            file_chars,
            file_articles,
            t_parse.elapsed().as_secs_f32(),
            total_chars,
            total_articles,
        );
        if total_chars >= TARGET_CHARS {
            println!("# target reached ({total_chars} >= {TARGET_CHARS}), stopping early");
            break;
        }
    }

    out.flush()?;
    println!(
        "# done. total chars={}, articles={}, elapsed={:.1}s, output={}",
        total_chars,
        total_articles,
        t_total.elapsed().as_secs_f32(),
        OUTPUT_PATH,
    );
    Ok(())
}

fn download(url: &str, local: &str) -> std::io::Result<()> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(600))
        .build()
        .map_err(|e| std::io::Error::other(format!("client build: {e}")))?;
    let mut resp = client
        .get(url)
        .send()
        .map_err(|e| std::io::Error::other(format!("get {url}: {e}")))?;
    if !resp.status().is_success() {
        return Err(std::io::Error::other(format!(
            "HTTP {} for {url}",
            resp.status()
        )));
    }
    let tmp = format!("{local}.partial");
    let mut out = BufWriter::with_capacity(8 * 1024 * 1024, File::create(&tmp)?);
    let mut buf = [0u8; 1024 * 64];
    let mut total: u64 = 0;
    let mut last_log = Instant::now();
    loop {
        let n = resp
            .read(&mut buf)
            .map_err(|e| std::io::Error::other(format!("read: {e}")))?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])?;
        total += n as u64;
        if last_log.elapsed().as_secs() >= 5 {
            println!("#     ... {} MB", total / 1_000_000);
            last_log = Instant::now();
        }
    }
    out.flush()?;
    std::fs::rename(tmp, local)?;
    Ok(())
}

fn extract_text<W: Write>(parquet_path: &str, out: &mut W) -> std::io::Result<(usize, usize)> {
    let file =
        File::open(parquet_path).map_err(|e| std::io::Error::other(format!("open parquet: {e}")))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| std::io::Error::other(format!("parquet builder: {e}")))?;
    let schema = builder.schema().clone();
    let text_idx = schema
        .index_of("text")
        .map_err(|e| std::io::Error::other(format!("no 'text' column: {e}")))?;
    let reader = builder
        .with_batch_size(1024)
        .build()
        .map_err(|e| std::io::Error::other(format!("parquet reader: {e}")))?;

    let mut chars = 0usize;
    let mut articles = 0usize;
    for batch in reader {
        let batch: RecordBatch =
            batch.map_err(|e| std::io::Error::other(format!("record_batch: {e}")))?;
        let col = batch
            .column(text_idx)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| std::io::Error::other("'text' column is not Utf8"))?;
        for i in 0..col.len() {
            if col.is_null(i) {
                continue;
            }
            let s = col.value(i);
            if s.is_empty() {
                continue;
            }
            if articles > 0 {
                out.write_all(DOC_SEP.as_bytes())?;
            }
            out.write_all(s.as_bytes())?;
            chars += s.chars().count();
            articles += 1;
        }
    }
    Ok((chars, articles))
}
