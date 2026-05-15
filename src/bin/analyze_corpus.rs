//! Phase 7-1-A: 青空文庫コーパスの構造分析。
//!
//! 入力 `corpus/aozora_meiji_taisho.txt` の以下の指標を計測:
//! - 作家ヘッダ (`===== 作家『タイトル』 =====`) の数 / 作家別作品数
//! - 振り仮名 `《...》` の出現数と消費 char 数
//! - 注釈マーカ `［＃...］` の出現数と消費 char 数
//! - 戯曲台詞行 (キャラクタ名 + 句読点 + 発言)
//! - 行頭スペース類 (ト書き候補)
//! - 1 行 1 char ぴったりの行 (戯曲フォーマットでよくある)
//!
//! 結果を `corpus/analysis_report.txt` に書き出す。
//!
//! 実行:
//!   cargo run --release --bin analyze_corpus

use std::collections::BTreeMap;
use std::fs;

const CORPUS_PATH: &str = "corpus/aozora_meiji_taisho.txt";
const REPORT_PATH: &str = "corpus/analysis_report.txt";

fn main() -> std::io::Result<()> {
    let corpus = fs::read_to_string(CORPUS_PATH)?;
    let total_chars = corpus.chars().count();
    let total_lines = corpus.lines().count();

    let mut report = String::new();
    macro_rules! out {
        ($($arg:tt)*) => {{
            let line = format!($($arg)*);
            println!("{line}");
            report.push_str(&line);
            report.push('\n');
        }};
    }

    out!("=== 青空文庫コーパス分析レポート ===");
    out!("入力: {CORPUS_PATH}");
    out!("総 char 数: {total_chars}");
    out!("総 line 数: {total_lines}");
    out!("");

    // ----- 1. 作家ヘッダ -----
    out!("--- 1. 作家ヘッダ ===== ... ===== ---");
    let mut author_works: BTreeMap<String, usize> = BTreeMap::new();
    let mut author_chars: BTreeMap<String, usize> = BTreeMap::new();
    let mut header_total_chars = 0usize;

    let mut current_author: Option<String> = None;
    let mut current_work_chars = 0usize;
    for line in corpus.lines() {
        if line.starts_with("=====") {
            if let Some(author) = &current_author {
                *author_chars.entry(author.clone()).or_insert(0) += current_work_chars;
            }
            header_total_chars += line.chars().count() + 1; // +1 = '\n'
            // 「===== 作家名『タイトル』 =====」 から作家名を抽出
            if let Some(rest) = line.strip_prefix("===== ") {
                if let Some(idx) = rest.find('『') {
                    let author = rest[..idx].trim().to_string();
                    *author_works.entry(author.clone()).or_insert(0) += 1;
                    current_author = Some(author);
                    current_work_chars = 0;
                }
            }
        } else {
            current_work_chars += line.chars().count() + 1;
        }
    }
    if let Some(author) = &current_author {
        *author_chars.entry(author.clone()).or_insert(0) += current_work_chars;
    }

    out!(
        "ヘッダ行 自体の char 数: {header_total_chars} ({:.3}%)",
        header_total_chars as f64 * 100.0 / total_chars as f64
    );
    out!("");
    out!("作家別 (作品数 / char 数):");
    let mut authors: Vec<(&String, &usize)> = author_works.iter().collect();
    authors.sort_by(|a, b| b.1.cmp(a.1));
    let mut sum_works = 0;
    let mut sum_chars = 0;
    for (author, works) in &authors {
        let chars = author_chars.get(*author).copied().unwrap_or(0);
        let pct = chars as f64 * 100.0 / total_chars as f64;
        out!(
            "  {} : {:>3} 作 / {:>10} char ({:>5.2}%)",
            author,
            works,
            chars,
            pct
        );
        sum_works += **works;
        sum_chars += chars;
    }
    out!(
        "  -- 合計 {} 作家 / {} 作 / {} char ({:.2}%)",
        authors.len(),
        sum_works,
        sum_chars,
        sum_chars as f64 * 100.0 / total_chars as f64
    );
    out!("");

    // ----- 2. 振り仮名 《...》 -----
    out!("--- 2. 振り仮名 《...》 ---");
    let (ruby_count, ruby_chars) = count_pattern(&corpus, '《', '》');
    out!("出現数: {ruby_count}");
    out!(
        "消費 char 数 (区切り文字含む): {ruby_chars} ({:.3}%)",
        ruby_chars as f64 * 100.0 / total_chars as f64
    );
    out!("");

    // ----- 3. 注釈マーカ ［＃...］ -----
    out!("--- 3. 注釈マーカ ［＃...］ ---");
    let mut annot_count = 0usize;
    let mut annot_chars = 0usize;
    let bytes = corpus.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        // ［＃ で始まり ］ で閉じる範囲を探す (UTF-8 安全に)
        if corpus[i..].starts_with("［＃") {
            if let Some(rel_end) = corpus[i..].find('］') {
                annot_count += 1;
                annot_chars += corpus[i..i + rel_end + '］'.len_utf8()].chars().count();
                i += rel_end + '］'.len_utf8();
                continue;
            }
        }
        // 1 char 進める (UTF-8 char boundary)
        let ch_len = corpus[i..]
            .chars()
            .next()
            .map(|c| c.len_utf8())
            .unwrap_or(1);
        i += ch_len;
    }
    out!("出現数: {annot_count}");
    out!(
        "消費 char 数: {annot_chars} ({:.3}%)",
        annot_chars as f64 * 100.0 / total_chars as f64
    );
    out!("");

    // ----- 4. 戯曲台詞行の検出 -----
    // ヒューリスティック: 行頭が漢字/カタカナ 1-6 文字 + 句読点 (「、」「。」) もしくは
    // 直後にスペース → 戯曲フォーマットでよく見る「ハムレット　お、何ぞ」など
    // 簡略化: 行が「漢字/カナ + 　 (全角スペース) + 続く」パターン
    out!("--- 4. 戯曲台詞行 (ヒューリスティック) ---");
    let mut drama_line_count = 0usize;
    let mut drama_line_chars = 0usize;
    for line in corpus.lines() {
        if is_drama_speaker_line(line) {
            drama_line_count += 1;
            drama_line_chars += line.chars().count();
        }
    }
    out!("検出行数: {drama_line_count}");
    out!(
        "char 数: {drama_line_chars} ({:.3}%)",
        drama_line_chars as f64 * 100.0 / total_chars as f64
    );
    out!("");

    // ----- 5. ト書き行 (行頭が () もしくは （） で始まる) -----
    out!("--- 5. ト書き行 (行頭 ( or （) ---");
    let mut stage_count = 0usize;
    let mut stage_chars = 0usize;
    for line in corpus.lines() {
        let trimmed = line.trim_start_matches([' ', '　', '\t']);
        if trimmed.starts_with('(') || trimmed.starts_with('（') {
            stage_count += 1;
            stage_chars += line.chars().count();
        }
    }
    out!("検出行数: {stage_count}");
    out!(
        "char 数: {stage_chars} ({:.3}%)",
        stage_chars as f64 * 100.0 / total_chars as f64
    );
    out!("");

    // ----- 6. 行長分布 -----
    out!("--- 6. 行長分布 ---");
    let mut bins = [0usize; 8]; // 0:0, 1:1-9, 2:10-49, 3:50-199, 4:200-499, 5:500-999, 6:1000-1999, 7:2000+
    for line in corpus.lines() {
        let n = line.chars().count();
        let b = match n {
            0 => 0,
            1..=9 => 1,
            10..=49 => 2,
            50..=199 => 3,
            200..=499 => 4,
            500..=999 => 5,
            1000..=1999 => 6,
            _ => 7,
        };
        bins[b] += 1;
    }
    let labels = [
        "0 char (空行)",
        "1-9 char",
        "10-49 char",
        "50-199 char",
        "200-499 char",
        "500-999 char",
        "1000-1999 char",
        "2000+ char",
    ];
    for (i, label) in labels.iter().enumerate() {
        let pct = bins[i] as f64 * 100.0 / total_lines as f64;
        out!("  {:>16}: {:>7} 行 ({:>5.2}%)", label, bins[i], pct);
    }
    out!("");

    // ----- 7. 戯曲を多く含むと思われる作品の抽出 -----
    // 各作品 (===== ...) ごとに「戯曲台詞行率」 を計算し、上位 20 作品を表示
    out!("--- 7. 戯曲台詞行率の高い作品 TOP 20 ---");
    let mut works: Vec<WorkStat> = Vec::new();
    let mut current = WorkStat::default();
    for line in corpus.lines() {
        if line.starts_with("=====") {
            if !current.title.is_empty() {
                works.push(std::mem::take(&mut current));
            }
            current.title = line.to_string();
            continue;
        }
        current.line_count += 1;
        if is_drama_speaker_line(line) {
            current.drama_count += 1;
        }
    }
    if !current.title.is_empty() {
        works.push(current);
    }
    let mut sorted: Vec<&WorkStat> = works.iter().filter(|w| w.line_count >= 20).collect();
    sorted.sort_by(|a, b| {
        let ar = a.drama_ratio();
        let br = b.drama_ratio();
        br.partial_cmp(&ar).unwrap_or(std::cmp::Ordering::Equal)
    });
    for w in sorted.iter().take(20) {
        out!(
            "  {:>5.2}%  ({:>4}/{:>4} 行)  {}",
            w.drama_ratio() * 100.0,
            w.drama_count,
            w.line_count,
            w.title
        );
    }
    out!("");

    // ----- まとめ -----
    out!("=== サマリ ===");
    out!("クレンジング候補の合計 char 数:");
    out!(
        "  作家ヘッダ          : {:>10} ({:>5.3}%)",
        header_total_chars,
        header_total_chars as f64 * 100.0 / total_chars as f64
    );
    out!(
        "  振り仮名 《...》    : {:>10} ({:>5.3}%)",
        ruby_chars,
        ruby_chars as f64 * 100.0 / total_chars as f64
    );
    out!(
        "  注釈 ［＃...］      : {:>10} ({:>5.3}%)",
        annot_chars,
        annot_chars as f64 * 100.0 / total_chars as f64
    );
    out!(
        "  戯曲台詞行          : {:>10} ({:>5.3}%)",
        drama_line_chars,
        drama_line_chars as f64 * 100.0 / total_chars as f64
    );
    out!(
        "  ト書き行            : {:>10} ({:>5.3}%)",
        stage_chars,
        stage_chars as f64 * 100.0 / total_chars as f64
    );
    let cleansable = ruby_chars + annot_chars + drama_line_chars + stage_chars;
    out!(
        "  -- 合計 (重複あり)  : {:>10} ({:>5.3}%)",
        cleansable,
        cleansable as f64 * 100.0 / total_chars as f64
    );

    fs::write(REPORT_PATH, &report)?;
    println!("\n→ レポートを {REPORT_PATH} に保存しました");
    Ok(())
}

#[derive(Default)]
struct WorkStat {
    title: String,
    line_count: usize,
    drama_count: usize,
}

impl WorkStat {
    fn drama_ratio(&self) -> f64 {
        if self.line_count == 0 {
            0.0
        } else {
            self.drama_count as f64 / self.line_count as f64
        }
    }
}

/// `《` で始まり `》` で終わる ペアの出現数 と それが消費する char 数 (区切り含む) を返す。
fn count_pattern(text: &str, open: char, close: char) -> (usize, usize) {
    let mut count = 0usize;
    let mut chars = 0usize;
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if let Some(c) = text[i..].chars().next() {
            if c == open {
                if let Some(rel_end) = text[i..].find(close) {
                    count += 1;
                    chars += text[i..i + rel_end + close.len_utf8()].chars().count();
                    i += rel_end + close.len_utf8();
                    continue;
                }
            }
            i += c.len_utf8();
        } else {
            break;
        }
    }
    (count, chars)
}

/// 戯曲の台詞行ヒューリスティック:
/// - 行頭がカタカナ・漢字 1-6 文字
/// - 続いて全角/半角スペース or 句読点 (、。)
/// - その後に何か続く (発言テキスト)
///
/// 例: 「ハムレット　お、何ぞ」、「王妃 私は……」
fn is_drama_speaker_line(line: &str) -> bool {
    let s = line.trim_start_matches([' ', '　', '\t']);
    let mut chars = s.chars();
    let mut speaker_len = 0;
    let mut iter_clone = s.chars();
    for _ in 0..6 {
        match iter_clone.next() {
            Some(c) if is_kanji(c) || is_katakana(c) => speaker_len += 1,
            Some(_) | None => break,
        }
    }
    if speaker_len < 1 {
        return false;
    }
    // ヒューリスティックの強化: speaker_len 文字後に区切り or 全角空白が来ること
    let after: String = chars.by_ref().skip(speaker_len).take(2).collect();
    if let Some(c) = after.chars().next() {
        if c == '　' || c == ' ' || c == '、' || c == '。' || c == '：' || c == ':' {
            // さらに後ろに少なくとも 5 char はテキストが続くこと (空 / コメントを除外)
            let rest_len = s.chars().skip(speaker_len + 1).count();
            if rest_len >= 5 {
                return true;
            }
        }
    }
    false
}

fn is_kanji(c: char) -> bool {
    matches!(c as u32,
        0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0x20000..=0x2A6DF
    )
}

fn is_katakana(c: char) -> bool {
    matches!(c as u32, 0x30A0..=0x30FF | 0x31F0..=0x31FF)
}
