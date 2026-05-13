//! Phase 7-1-B: 青空文庫コーパスクレンジング + special token 化スクリプト。
//!
//! 入力 `corpus/aozora_meiji_taisho.txt` (旧形式: `===== 作家『タイトル』 =====` ヘッダ + 本文)
//! を読み込み、 下記 v2 形式に変換して `corpus/aozora_meiji_taisho_v2.txt` に書き出す。
//!
//! 出力フォーマット (1 作品ごと):
//!   <BOS><AUTHOR=作家名><TITLE>タイトル</TITLE>
//!   本文1
//!   本文2
//!   ...
//!   <EOS>
//!
//! 戯曲フォーマット作品 (戯曲台詞行率 >= DRAMA_THRESHOLD) は本文を `<DRAMA>` で囲む:
//!   <BOS><AUTHOR=作家名><TITLE>タイトル</TITLE><DRAMA>
//!   ハムレット　お、何ぞ
//!   王妃　わが子よ……
//!   ...
//!   </DRAMA><EOS>
//!
//! 期待効果 (Phase 7-a 学習で):
//!   - 散文プロンプトに戯曲台詞 (「ハムレット」「王妃」) が混入しない
//!   - 推論時に作家を `<AUTHOR=漱石>` で指定できる
//!   - 旧 `===== ... =====` ヘッダ自体を生成しなくなる
//!
//! 実行: cargo run --release --bin clean_aozora_corpus

use std::fs;

const INPUT_PATH: &str = "corpus/aozora_meiji_taisho.txt";
const OUTPUT_PATH: &str = "corpus/aozora_meiji_taisho_v2.txt";
const REPORT_PATH: &str = "corpus/clean_aozora_corpus_report.txt";

/// 戯曲フォーマット判定の閾値: 戯曲台詞行率がこれを超える作品は `<DRAMA>` で囲む。
const DRAMA_THRESHOLD: f64 = 0.45;
/// 上記しきい値での判定で必要な最小行数 (短編で偶発的に率が高くなるのを除外)。
const DRAMA_MIN_LINES: usize = 20;

/// 章番号として扱う漢数字 (`〇` 「ゼロ」 を含む)。 「百」「千」 はタイトル本文への混在リスクがあるので含めない。
fn is_chapter_kanji_digit(c: char) -> bool {
    matches!(c, '一' | '二' | '三' | '四' | '五' | '六' | '七' | '八' | '九' | '十' | '〇')
}

const BOS: &str = "<BOS>";
const EOS: &str = "<EOS>";
const TITLE_OPEN: &str = "<TITLE>";
const TITLE_CLOSE: &str = "</TITLE>";
const DRAMA_OPEN: &str = "<DRAMA>";
const DRAMA_CLOSE: &str = "</DRAMA>";

fn main() -> std::io::Result<()> {
    let input = fs::read_to_string(INPUT_PATH)?;
    let total_in_chars = input.chars().count();

    let works = split_into_works(&input);
    println!("# 入力: {INPUT_PATH}");
    println!("# 総 char 数: {total_in_chars}");
    println!("# 作品数: {}", works.len());
    println!();

    let mut output = String::with_capacity(input.len() + works.len() * 64);
    let mut report = String::new();
    let mut drama_count = 0usize;
    let mut drama_chars = 0usize;
    let mut prose_chars = 0usize;
    let mut total_removed_chapters = 0usize;
    let mut author_stats: std::collections::BTreeMap<String, (usize, usize)> = Default::default();

    for w in &works {
        let drama_ratio = w.drama_ratio();
        let is_drama = drama_ratio >= DRAMA_THRESHOLD && w.line_count() >= DRAMA_MIN_LINES;
        let (body, removed_chapters) = w.normalized_body();
        total_removed_chapters += removed_chapters;

        // 作品ヘッダ + 本文 + EOS を出力
        output.push_str(BOS);
        output.push_str(&format!("<AUTHOR={}>", w.author));
        output.push_str(TITLE_OPEN);
        output.push_str(&w.title);
        output.push_str(TITLE_CLOSE);
        output.push('\n');

        if is_drama {
            output.push_str(DRAMA_OPEN);
            output.push('\n');
            output.push_str(&body);
            if !body.ends_with('\n') {
                output.push('\n');
            }
            output.push_str(DRAMA_CLOSE);
            output.push('\n');
        } else {
            output.push_str(&body);
            if !body.ends_with('\n') {
                output.push('\n');
            }
        }
        output.push_str(EOS);
        output.push('\n');

        let body_chars = body.chars().count();
        if is_drama {
            drama_count += 1;
            drama_chars += body_chars;
        } else {
            prose_chars += body_chars;
        }
        let entry = author_stats.entry(w.author.clone()).or_insert((0, 0));
        entry.0 += 1;
        entry.1 += body_chars;
    }

    let total_out_chars = output.chars().count();
    fs::write(OUTPUT_PATH, &output)?;

    macro_rules! r {
        ($($arg:tt)*) => {{
            let line = format!($($arg)*);
            println!("{line}");
            report.push_str(&line);
            report.push('\n');
        }};
    }

    r!("=== Phase 7-1-B クレンジング結果 ===");
    r!("入力: {INPUT_PATH} ({total_in_chars} chars)");
    r!("出力: {OUTPUT_PATH} ({total_out_chars} chars, {:+.2}%)",
       (total_out_chars as i64 - total_in_chars as i64) as f64 * 100.0 / total_in_chars as f64);
    r!("");
    r!("--- クレンジング操作 ---");
    r!("  作家ヘッダ (===== ... =====) → <BOS><AUTHOR=...><TITLE>...</TITLE>: 全 {} 行", works.len());
    r!("  章番号行削除 (`　　　五`, `一`, `第二巻` 等): {total_removed_chapters} 行");
    r!("");
    r!("--- 作品分類 ---");
    r!("  戯曲扱い (DRAMA で囲む): {drama_count} 作 / {drama_chars} char ({:.2}%)",
       drama_chars as f64 * 100.0 / total_in_chars as f64);
    r!("  散文扱い:                 {} 作 / {} char ({:.2}%)",
       works.len() - drama_count, prose_chars,
       prose_chars as f64 * 100.0 / total_in_chars as f64);
    r!("");
    r!("--- 戯曲扱いになった作品 ---");
    let mut drama_works: Vec<&Work> = works.iter()
        .filter(|w| w.drama_ratio() >= DRAMA_THRESHOLD && w.line_count() >= DRAMA_MIN_LINES)
        .collect();
    drama_works.sort_by(|a, b| b.drama_ratio().partial_cmp(&a.drama_ratio()).unwrap());
    for w in &drama_works {
        r!("  {:>5.1}% ({:>3}/{:>3} 行)  {} 『{}』",
           w.drama_ratio() * 100.0, w.drama_count(), w.line_count(), w.author, w.title);
    }
    r!("");
    r!("--- 作家別 char 分布 (新コーパス) ---");
    let total_body_chars = prose_chars + drama_chars;
    let mut stats: Vec<(&String, &(usize, usize))> = author_stats.iter().collect();
    stats.sort_by(|a, b| b.1.1.cmp(&a.1.1));
    for (author, (works_n, chars)) in &stats {
        let pct = *chars as f64 * 100.0 / total_body_chars as f64;
        r!("  {:<10} {:>3} 作  {:>10} char ({:>5.2}%)", author, works_n, chars, pct);
    }
    r!("");
    r!("--- 追加すべき special token (12 個) ---");
    r!("  <BOS>, <EOS>");
    r!("  <TITLE>, </TITLE>");
    r!("  <DRAMA>, </DRAMA>");
    for (author, _) in &stats {
        r!("  <AUTHOR={}>", author);
    }

    fs::write(REPORT_PATH, &report)?;
    println!("\n→ レポート: {REPORT_PATH}");
    println!("→ 新コーパス: {OUTPUT_PATH}");
    Ok(())
}

#[derive(Default)]
struct Work {
    author: String,
    title: String,
    body_lines: Vec<String>,
}

impl Work {
    fn line_count(&self) -> usize {
        self.body_lines.iter().filter(|l| !l.is_empty()).count()
    }
    fn drama_count(&self) -> usize {
        self.body_lines
            .iter()
            .filter(|l| is_drama_speaker_line(l))
            .count()
    }
    fn drama_ratio(&self) -> f64 {
        let n = self.line_count();
        if n == 0 {
            0.0
        } else {
            self.drama_count() as f64 / n as f64
        }
    }
    /// 旧 `=====` ヘッダ行を完全除去 + 章番号行を除外した本文を返す。 末尾の連続空行は trim。
    /// Phase 7-1-B 第 2 弾: 章番号 (例: 「　　　　　五」 「一」 「第二巻」) を削除して
    /// 「謎の漢数字」 が学習データに混ざるのを防ぐ。
    fn normalized_body(&self) -> (String, usize) {
        let mut s = String::with_capacity(self.body_lines.iter().map(|l| l.len() + 1).sum());
        let mut removed_chapters = 0usize;
        for line in &self.body_lines {
            if is_chapter_number_line(line) {
                removed_chapters += 1;
                continue;
            }
            s.push_str(line);
            s.push('\n');
        }
        while s.ends_with("\n\n") {
            s.pop();
        }
        (s, removed_chapters)
    }
}

/// 入力テキストを `===== 作家『タイトル』 =====` ヘッダで作品単位に分割。
fn split_into_works(text: &str) -> Vec<Work> {
    let mut works = Vec::new();
    let mut current: Option<Work> = None;
    for line in text.lines() {
        if line.starts_with("=====") {
            if let Some(w) = current.take() {
                works.push(w);
            }
            // 「===== 作家『タイトル』 =====」 から作家・タイトル抽出
            let trimmed = line
                .trim_start_matches("=====")
                .trim_end_matches("=====")
                .trim();
            let (author, title) = parse_header(trimmed);
            current = Some(Work {
                author,
                title,
                body_lines: Vec::new(),
            });
            continue;
        }
        if let Some(w) = current.as_mut() {
            w.body_lines.push(line.to_string());
        }
        // ヘッダ前のプリアンブルがあっても無視
    }
    if let Some(w) = current {
        works.push(w);
    }
    works
}

/// 「作家名『タイトル』」 から (作家, タイトル) を抽出する。
/// マッチしない場合は (全文, "") を返す。
fn parse_header(s: &str) -> (String, String) {
    if let Some(open_idx) = s.find('『') {
        let author = s[..open_idx].trim().to_string();
        let after = &s[open_idx + '『'.len_utf8()..];
        let title = if let Some(close_idx) = after.find('』') {
            after[..close_idx].trim().to_string()
        } else {
            after.trim().to_string()
        };
        (author, title)
    } else {
        (s.to_string(), String::new())
    }
}

/// 戯曲台詞行ヒューリスティック (analyze_corpus.rs と同一のロジック)。
fn is_drama_speaker_line(line: &str) -> bool {
    let s = line.trim_start_matches([' ', '　', '\t']);
    let mut speaker_len = 0;
    let mut iter_clone = s.chars();
    for _ in 0..6 {
        match iter_clone.next() {
            Some(c) if is_kanji(c) || is_katakana(c) => speaker_len += 1,
            _ => break,
        }
    }
    if speaker_len == 0 {
        return false;
    }
    let after: String = s.chars().skip(speaker_len).take(2).collect();
    if let Some(c) = after.chars().next() {
        if matches!(c, '　' | ' ' | '、' | '。' | '：' | ':') {
            let rest_len = s.chars().skip(speaker_len + 1).count();
            if rest_len >= 5 {
                return true;
            }
        }
    }
    false
}

fn is_kanji(c: char) -> bool {
    matches!(c as u32, 0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0x20000..=0x2A6DF)
}

fn is_katakana(c: char) -> bool {
    matches!(c as u32, 0x30A0..=0x30FF | 0x31F0..=0x31FF)
}

/// 章番号行の判定 (Phase 7-1-B 第 2 弾):
/// - パターン 1: 行頭が全角/半角スペースで始まり、 残りが漢数字 1-3 文字のみ
///   例: 「　　　　　　五」、 「　　　　　　　第二回」
/// - パターン 2: 行全体が漢数字 1-3 文字 (空白なし) ←短編の章番号でよくある
///   例: 「一」、 「二」、 「三」
/// - パターン 3: 「第X章/節/巻/編/部/話/卷/回」 形式 (空白あり/なし)
///
/// `（一）` 「(1)」 等の括弧付き番号は本文に出現する可能性があるため対象外。
fn is_chapter_number_line(line: &str) -> bool {
    let trimmed = line.trim_matches([' ', '　', '\t']);
    if trimmed.is_empty() {
        return false;
    }
    let chars: Vec<char> = trimmed.chars().collect();

    // パターン 1+2: 漢数字 1-3 文字のみ
    if chars.len() <= 3 && chars.iter().all(|&c| is_chapter_kanji_digit(c)) {
        return true;
    }

    // パターン 3: 「第X章/巻/回/節/編/部/話/卷」
    if chars.len() <= 6 && chars[0] == '第' {
        let suffix = chars.last().copied().unwrap_or(' ');
        if matches!(suffix, '章' | '節' | '部' | '編' | '話' | '巻' | '卷' | '回') {
            // 中間が漢数字または ASCII/全角数字のみであることを確認
            let middle = &chars[1..chars.len() - 1];
            if !middle.is_empty()
                && middle.iter().all(|&c| {
                    is_chapter_kanji_digit(c)
                        || c == '百'
                        || c == '千'
                        || c.is_ascii_digit()
                        || ('０'..='９').contains(&c)
                })
            {
                return true;
            }
        }
    }
    false
}
