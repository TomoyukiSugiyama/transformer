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
    let mut total_removed_annotations = 0usize;
    let mut author_stats: std::collections::BTreeMap<String, (usize, usize)> = Default::default();

    for w in &works {
        let drama_ratio = w.drama_ratio();
        let is_drama = drama_ratio >= DRAMA_THRESHOLD && w.line_count() >= DRAMA_MIN_LINES;
        let (body, removed_chapters, removed_annotations) = w.normalized_body();
        total_removed_chapters += removed_chapters;
        total_removed_annotations += removed_annotations;

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
    r!("  編集者注釈除去 (`〔以下空白〕`, `〔一字不明〕` 等): {total_removed_annotations} 個");
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
    /// 戻り値 = (本文, 削除した章番号行数, 除去した編集者注釈数)。
    /// Phase 7-1-B 第 2 弾: 章番号 (例: 「　　　　　五」 「一」 「第二巻」) を削除。
    /// Phase 7-1-B 第 7 弾: インライン編集者注釈 `〔以下空白〕` 等を除去。
    fn normalized_body(&self) -> (String, usize, usize) {
        let mut s = String::with_capacity(self.body_lines.iter().map(|l| l.len() + 1).sum());
        let mut removed_chapters = 0usize;
        let mut removed_annotations = 0usize;
        for line in &self.body_lines {
            if is_chapter_number_line(line) {
                removed_chapters += 1;
                continue;
            }
            let (cleaned, n) = strip_editorial_annotations(line);
            removed_annotations += n;
            s.push_str(&cleaned);
            s.push('\n');
        }
        while s.ends_with("\n\n") {
            s.pop();
        }
        (s, removed_chapters, removed_annotations)
    }
}

/// インラインの編集者注釈 `〔以下空白〕` 系を除去する。
///
/// 青空文庫テキストには `〔` `〕` で囲まれた 2 種類の挿入があり、 内容で判別する:
///   - Type A (保持): Latin 文字転記 (`〔natu:rlich〕` = natürlich、 `〔retrouve'e〕` = retrouvée 等)。
///                    本文の一部としてアクセント付きラテン文字の代用表現。
///   - Type B (削除): 編集者による欠落・空白・原稿状態の注釈
///                    例: `〔以下空白〕` 「〔一字不明〕」 `〔以下原稿数枚なし〕` `〔冒頭原稿数枚焼失〕`
///
/// Type B は以下の確定キーワードを内部に含む `〔...〕` を対象とする:
///   `以下` `不明` `空白` `脱落` `原稿` `字分` `焼失` `冒頭`
///
/// 該当部分は前後の空白 1 つも含めて除去 (連続空白を避けるため)。 戻り値 = (除去後文字列, 除去個数)。
fn strip_editorial_annotations(line: &str) -> (String, usize) {
    const TYPE_B_KEYWORDS: &[&str] = &[
        "以下", "不明", "空白", "脱落", "原稿", "字分", "焼失", "冒頭",
    ];
    let mut out = String::with_capacity(line.len());
    let mut removed = 0usize;
    let chars: Vec<char> = line.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '〔' {
            // `〕` を探す (最大 60 char 以内、 改行を含まない)
            let mut j = i + 1;
            let mut found_close = None;
            while j < chars.len() && j - i < 60 && chars[j] != '\n' {
                if chars[j] == '〕' {
                    found_close = Some(j);
                    break;
                }
                j += 1;
            }
            if let Some(close_idx) = found_close {
                let inner: String = chars[i + 1..close_idx].iter().collect();
                let is_type_b = TYPE_B_KEYWORDS.iter().any(|kw| inner.contains(kw));
                if is_type_b {
                    // 直前の半角/全角空白 1 つを削る (連続スペースを避ける)
                    if out.ends_with(' ') || out.ends_with('\u{3000}') {
                        out.pop();
                    }
                    i = close_idx + 1;
                    // 直後の半角/全角空白 1 つも飛ばす
                    if i < chars.len() && matches!(chars[i], ' ' | '\u{3000}') {
                        i += 1;
                    }
                    removed += 1;
                    continue;
                }
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    (out, removed)
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

/// 上中下巻 / 章節区切りに使われる単独漢字 (1 文字行のみ章番号扱い)。
/// 本文に高頻度で出現するが、 1 行 1 文字のレイアウトは慣習的に章/巻見出しに限定される。
fn is_volume_marker_kanji(c: char) -> bool {
    matches!(c, '上' | '中' | '下' | '前' | '後')
}

/// 単独 1 文字で章/節区切りに使われる装飾文字 (`※`, `☆`, `★`, `＊`, `*`, `×`, `○`, `●`, `◇`, `◆`)。
/// `―` (em-dash) は本文中の声引きで頻出するため含めない。
fn is_single_decorative_separator(c: char) -> bool {
    matches!(
        c,
        '※' | '☆' | '★' | '＊' | '*' | '×' | '○' | '●' | '◇' | '◆'
    )
}

/// 章番号 (漢数字) として扱える文字。 `百`/`千` も含むが、 単体使用は誤検出リスクがあるため
/// 「2 文字以上で `一-九/十/〇` を最低 1 文字含む」 行のみ章番号判定で使う。
fn is_extended_chapter_kanji_digit(c: char) -> bool {
    is_chapter_kanji_digit(c) || c == '百' || c == '千'
}

/// 全角数字 (`０` ～ `９`) もしくは ASCII 数字 (`0` ～ `9`)。
fn is_arabic_digit(c: char) -> bool {
    c.is_ascii_digit() || ('０'..='９').contains(&c)
}

/// 章番号行の判定 (Phase 7-1-B 第 2-8 弾):
/// - パターン 1: 行頭が全角/半角スペースで始まり、 残りが漢数字 1-3 文字のみ
/// - パターン 2: 行全体が漢数字 1-3 文字
/// - パターン 2b (第 5 弾): 漢数字 4-6 文字 (`百`/`千` 含む、 ただし 一-九/十/〇 を最低 1 文字含む)
///   例: 「百八十六」、 「百二十一」、 「千二百三十四」
/// - パターン 2c (第 6 弾): 全角/ASCII 数字のみ 1-4 文字
///   例: 「１」、 「２」、 「　　　３」、 「12」
/// - パターン 3: 「第X章/節/巻/編/部/話/卷/回」 形式
/// - パターン 4: 単独 `上` `中` `下` `前` `後`
/// - パターン 5: 単独装飾文字 (`※` `☆` `★` `＊` `*` `×` `○` `●` `◇` `◆`)
///   例: 「※」、 「　　　　　　　☆」、 「　　　　　　＊」
///   ※ `―` は em-dash の本文用法があるため除外。
/// - パターン 6: `※X※` 形式 (X = 漢数字 or 上/中/下/前/後)
/// - パターン 7 (第 4 弾): 装飾区切り行
///   装飾文字 A (`＊` `*` `×` `※`): 2 文字以上で区切り扱い (本文文脈はほぼ皆無)
///   装飾文字 B (`―` `─` `━`):     3 文字以上で区切り扱い
/// - パターン 8 (第 5 弾): 章番号 + タイトル (例: 「　　　二　ポローニヤス邸の一室」、 「　　　　三　袖」)
///   leading 2+ 全角空白 + 漢数字 1-3 文字 + 1+ 全角空白 + タイトル (1-15 文字、 句読点なし)
/// - パターン 8b (第 8 弾): 章番号 + 「、」 + タイトル (例: 「　　　四、ペンネンネンネンネン・ネネムの安心」)
///   leading 2+ 全角空白 + 漢数字 (一-九/十) 1-3 文字 + 「、」 + タイトル (1-15 文字、 内部に全角空白なし)
/// - パターン 9 (第 8 弾): 「その + 漢数字」 (例: 「その一」、 「　　　　その七十七」)
/// - パターン 10 (第 8 弾、 第 9 弾で拡張): 「第 + 漢数字」 (suffix なし)
///   例: 「　　　　第六」、 「　第二」、 「第一」
///   ※ chars.len() <= 4 で「第一次世界大戦」 等の本文を誤検出回避。
/// - パターン 13 (第 9 弾): 単独全角アルファベット Ａ-Ｚ (大文字)
///   例: 「Ａ」、 「　　　　　　Ａ」、 「Ｂ」 … 太宰治 「ＨＵＭＡＮ ＬＯＳＴ」 等の章記号。
/// - パターン 14 (第 10 弾): 単独括弧付き漢数字 「（漢数字）」 行
///   例: 「（一）」、 「　　　（二）」、 「（百）」
///   ※ 本文インライン (例: 「彼の（一）の発言」) は影響しない (whole-line マッチ)。
/// - パターン 11 (第 8 弾): 「タイトル　（漢数字）」 形式 (例: 「変心　（一）」、 「行進　（二）」)
/// - パターン 12 (第 8 弾): 「第 + 漢数字 + 短いタイトル」 (例: 「第二の手記」、 「第一夜」)
///   末尾に 句読点 (`。、？！`) を含む行 (例: 「第一場と同じ日。」) は本文扱いで除外。
fn is_chapter_number_line(line: &str) -> bool {
    let trimmed = line.trim_matches([' ', '　', '\t']);
    if trimmed.is_empty() {
        return false;
    }
    let chars: Vec<char> = trimmed.chars().collect();

    // パターン 1+2: 漢数字 (一-九/十/〇) 1-3 文字のみ
    if chars.len() <= 3 && chars.iter().all(|&c| is_chapter_kanji_digit(c)) {
        return true;
    }

    // パターン 2b: 漢数字 (百千含む) 4-6 文字 + 一-九/十/〇 を最低 1 文字含む
    if chars.len() >= 2
        && chars.len() <= 6
        && chars.iter().all(|&c| is_extended_chapter_kanji_digit(c))
        && chars.iter().any(|&c| is_chapter_kanji_digit(c))
    {
        return true;
    }

    // パターン 2c: 全角/ASCII 数字のみ 1-4 文字
    if chars.len() <= 4 && chars.iter().all(|&c| is_arabic_digit(c)) {
        return true;
    }

    // パターン 4: 単独 `上` `中` `下` `前` `後`
    if chars.len() == 1 && is_volume_marker_kanji(chars[0]) {
        return true;
    }

    // パターン 5: 単独装飾文字 (`※` `☆` `★` `＊` `*` `×` `○` `●` `◇` `◆`)
    if chars.len() == 1 && is_single_decorative_separator(chars[0]) {
        return true;
    }

    // パターン 6: `※X※` (X = 漢数字 1-3 文字 または 上/中/下/前/後)
    if chars.len() >= 3
        && chars.len() <= 5
        && chars[0] == '※'
        && *chars.last().unwrap() == '※'
    {
        let middle = &chars[1..chars.len() - 1];
        if middle
            .iter()
            .all(|&c| is_chapter_kanji_digit(c) || is_volume_marker_kanji(c))
        {
            return true;
        }
    }

    // パターン 7: 装飾区切り行 (複数装飾文字)
    let stripped: String = chars
        .iter()
        .filter(|&&c| !matches!(c, ' ' | '　' | '\t'))
        .collect();
    let stripped_len = stripped.chars().count();
    let all_decorative = stripped.chars().all(|c| {
        matches!(
            c,
            '＊' | '*' | '×' | '※' | '―' | '─' | '━'
        )
    });
    let has_strong_decorative = stripped
        .chars()
        .any(|c| matches!(c, '＊' | '*' | '×' | '※'));
    if !stripped.is_empty() && all_decorative {
        let threshold = if has_strong_decorative { 2 } else { 3 };
        if stripped_len >= threshold {
            return true;
        }
    }

    // パターン 3: 「第X章/巻/回/節/編/部/話/卷」
    if chars.len() <= 6 && chars[0] == '第' {
        let suffix = chars.last().copied().unwrap_or(' ');
        if matches!(suffix, '章' | '節' | '部' | '編' | '話' | '巻' | '卷' | '回') {
            let middle = &chars[1..chars.len() - 1];
            if !middle.is_empty()
                && middle.iter().all(|&c| {
                    is_extended_chapter_kanji_digit(c)
                        || c.is_ascii_digit()
                        || ('０'..='９').contains(&c)
                })
            {
                return true;
            }
        }
    }

    // パターン 8 + 8b: 章番号 + (空白 or 「、」) + タイトル
    let leading_ws_count = line
        .chars()
        .take_while(|c| matches!(c, ' ' | '　' | '\t'))
        .count();
    if leading_ws_count >= 2 {
        let after_ws: Vec<char> = line.chars().skip(leading_ws_count).collect();
        let mut digit_end = 0;
        while digit_end < after_ws.len()
            && digit_end < 3
            && is_chapter_kanji_digit(after_ws[digit_end])
        {
            digit_end += 1;
        }
        if digit_end >= 1 && digit_end < after_ws.len() {
            let separator = after_ws[digit_end];
            // パターン 8: 漢数字 + 空白 + タイトル (タイトル 1-25 文字、 句読点なし)
            if matches!(separator, ' ' | '　' | '\t') {
                let title_part: String = after_ws[digit_end + 1..].iter().collect();
                let title_trimmed = title_part.trim_matches([' ', '　', '\t']);
                let title_len = title_trimmed.chars().count();
                if title_len >= 1
                    && title_len <= 25
                    && !title_trimmed.contains([
                        '。', '、', '？', '！', '」', '』', '）', ')', '?', '!',
                    ])
                {
                    return true;
                }
            }
            // パターン 8b: 漢数字 + 「、」 + タイトル (タイトル内に全角空白なし、 1-25 文字)
            if separator == '、' {
                let title_part: String = after_ws[digit_end + 1..].iter().collect();
                let title_trimmed = title_part.trim_matches([' ', '　', '\t']);
                let title_len = title_trimmed.chars().count();
                if title_len >= 1
                    && title_len <= 25
                    && !title_trimmed.contains([
                        '　', '。', '、', '？', '！', '」', '』', '）', ')', '?', '!',
                    ])
                {
                    return true;
                }
            }
        }
    }

    // パターン 9: `その` + 漢数字 (1-4 文字)
    if chars.len() >= 3 && chars.len() <= 6 && chars[0] == 'そ' && chars[1] == 'の' {
        let rest = &chars[2..];
        if !rest.is_empty() && rest.iter().all(|&c| is_extended_chapter_kanji_digit(c)) {
            return true;
        }
    }

    // パターン 10: 「第 + 漢数字」 (suffix なし、 chars.len() <= 4 で本文複合語を回避)
    if chars.len() >= 2
        && chars.len() <= 4
        && chars[0] == '第'
        && chars[1..]
            .iter()
            .all(|&c| is_extended_chapter_kanji_digit(c))
    {
        return true;
    }

    // パターン 13: 単独全角アルファベット (Ａ-Ｚ)
    if chars.len() == 1 && ('Ａ'..='Ｚ').contains(&chars[0]) {
        return true;
    }

    // パターン 14: 「（漢数字）」 単独行 (1-3 文字の漢数字、 百千含む)
    if chars.len() >= 3
        && chars.len() <= 5
        && matches!(chars[0], '（' | '(')
        && matches!(*chars.last().unwrap(), '）' | ')')
    {
        let middle = &chars[1..chars.len() - 1];
        if !middle.is_empty()
            && middle.iter().all(|&c| is_extended_chapter_kanji_digit(c))
        {
            return true;
        }
    }

    // パターン 11: 「タイトル　（漢数字）」 (例: 「変心　（一）」)
    //   タイトル 1-12 文字 + 全角空白 + 全角/半角括弧 + 漢数字 1-3 + 閉じ括弧
    if chars.len() >= 5 && chars.len() <= 20 {
        // 末尾が 「）」 or 「)」
        let last = chars[chars.len() - 1];
        if last == '）' || last == ')' {
            // 末尾から逆走して 「（」 or 「(」 を探す
            if let Some(open_rev) = chars
                .iter()
                .rev()
                .skip(1)
                .position(|&c| c == '（' || c == '(')
            {
                let open_idx = chars.len() - 2 - open_rev;
                let inside = &chars[open_idx + 1..chars.len() - 1];
                let inside_is_digit = !inside.is_empty()
                    && inside.len() <= 3
                    && inside.iter().all(|&c| is_extended_chapter_kanji_digit(c));
                if inside_is_digit && open_idx >= 2 {
                    // タイトル部 (open_idx 直前まで) が `<title>　` で終わる
                    if chars[open_idx - 1] == '　' {
                        let title_part = &chars[..open_idx - 1];
                        let title_len = title_part.len();
                        if (1..=12).contains(&title_len)
                            && !title_part.iter().any(|&c| {
                                matches!(
                                    c,
                                    '。' | '、'
                                        | '？'
                                        | '！'
                                        | '」'
                                        | '』'
                                        | '　'
                                        | ' '
                                        | '?'
                                        | '!'
                                )
                            })
                        {
                            return true;
                        }
                    }
                }
            }
        }
    }

    // パターン 12: 「第 + 漢数字 + 短いタイトル」 (例: 「第二の手記」 (suffix 3 文字)、 「第一夜」 (1 文字))
    //   suffix は 1-4 文字に厳格制限 (`第一次世界大戦` 等の本文複合語を誤検出しないため)。
    //   末尾に句読点を含む行 (本文ト書き) は除外。
    if chars.len() >= 3 && chars.len() <= 8 && chars[0] == '第' {
        let mut digit_end = 1;
        while digit_end < chars.len()
            && digit_end < 4
            && is_extended_chapter_kanji_digit(chars[digit_end])
        {
            digit_end += 1;
        }
        if digit_end >= 2 && digit_end < chars.len() {
            let suffix = &chars[digit_end..];
            let has_punctuation = suffix.iter().any(|&c| {
                matches!(
                    c,
                    '。' | '、' | '？' | '！' | '」' | '』' | '）' | ')' | '?' | '!'
                )
            });
            // 既に 章/節/部/編/話/巻/卷/回 で終わる場合は パターン 3 で処理済みなのでスキップ
            let already_p3 = matches!(
                *suffix.last().unwrap(),
                '章' | '節' | '部' | '編' | '話' | '巻' | '卷' | '回'
            );
            if !has_punctuation && !already_p3 && (1..=4).contains(&suffix.len()) {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chapter_kanji_digit_only() {
        for s in ["一", "二", "三", "十", "〇", "　　　　　五", "　二十", "　　　九"] {
            assert!(is_chapter_number_line(s), "should match: {s:?}");
        }
    }

    #[test]
    fn chapter_dai_x_suffix() {
        for s in ["第一章", "第二回", "第三節", "第十巻", "第百話", "　　第二編"] {
            assert!(is_chapter_number_line(s), "should match: {s:?}");
        }
    }

    #[test]
    fn chapter_volume_markers_alone() {
        for s in ["上", "中", "下", "前", "後", "　　　　　上", "　　　下"] {
            assert!(is_chapter_number_line(s), "should match: {s:?}");
        }
    }

    #[test]
    fn chapter_komejirushi_alone() {
        for s in ["※", "　　　　　　　※", " ※"] {
            assert!(is_chapter_number_line(s), "should match: {s:?}");
        }
    }

    #[test]
    fn chapter_single_decorative_alone() {
        for s in [
            "☆",
            "★",
            "＊",
            "*",
            "×",
            "○",
            "●",
            "◇",
            "◆",
            "　　　　　　　＊",
            "　　　　　　　　　☆",
            "　　　　　　　　　×",
        ] {
            assert!(is_chapter_number_line(s), "should match: {s:?}");
        }
    }

    #[test]
    fn chapter_arabic_digits() {
        for s in [
            "１", "２", "３", "４", "５", "10", "12", "１２", "1234", "　　　３", "　　　２０",
        ] {
            assert!(is_chapter_number_line(s), "should match: {s:?}");
        }
    }

    #[test]
    fn chapter_long_kanji_digits() {
        for s in [
            "百八十六",
            "百二十一",
            "百三十一",
            "千二百三十四",
            "二十一",
            "三十",
            "　　百八十六",
        ] {
            assert!(is_chapter_number_line(s), "should match: {s:?}");
        }
    }

    #[test]
    fn chapter_num_with_comma_title() {
        for s in [
            "　　　四、ペンネンネンネンネン・ネネムの安心",
            "　　　一、赤い手長の蜘蛛",
            "　　　二、銀色のなめくじ",
            "　　　三、顔を洗わない狸",
        ] {
            assert!(is_chapter_number_line(s), "should match: {s:?}");
        }
    }

    #[test]
    fn chapter_sono_kanji() {
        for s in [
            "その一",
            "その二",
            "その七十七",
            "　　　　　　　その三",
            "その十",
            "その百",
        ] {
            assert!(is_chapter_number_line(s), "should match: {s:?}");
        }
    }

    #[test]
    fn chapter_dai_x_no_suffix() {
        for s in [
            "　　　　第二",
            "　　　　第六",
            "　第三",
            "　　第十",
            "第一",   // leading whitespace なし (第 9 弾)
            "第二",
            "第百",
        ] {
            assert!(is_chapter_number_line(s), "should match: {s:?}");
        }
    }

    #[test]
    fn chapter_paren_kanji_number_alone() {
        for s in [
            "（一）",
            "（二）",
            "（百）",
            "　　　（二）",
            "　　　　（三）",
            "(一)",     // ASCII 括弧
            "（十）",
        ] {
            assert!(is_chapter_number_line(s), "should match: {s:?}");
        }
    }

    #[test]
    fn chapter_single_fullwidth_latin() {
        for s in [
            "Ａ", "Ｂ", "Ｃ", "Ｚ",
            "　　　　　　　　Ａ",
            " Ｂ",
        ] {
            assert!(is_chapter_number_line(s), "should match: {s:?}");
        }
    }

    #[test]
    fn chapter_title_paren_number() {
        for s in [
            "変心　（一）",
            "行進　（二）",
            "怪力　（三）",
            "短い題　（七）",
        ] {
            assert!(is_chapter_number_line(s), "should match: {s:?}");
        }
    }

    #[test]
    fn chapter_dai_x_with_short_suffix() {
        for s in [
            "第一の手記",
            "第二の手記",
            "第三の手記",
            "第一夜",
            "第二夜",
            "第十夜",
        ] {
            assert!(is_chapter_number_line(s), "should match: {s:?}");
        }
    }

    #[test]
    fn chapter_num_with_title() {
        for s in [
            "　　　二　ポローニヤス邸の一室",
            "　　　三　高台",
            "　　　　　一　夢",
            "　　　　　二　鏡",
            "　　　　　三　袖",
            "　　　四　王妃の居間",
            "　　二　海賊",
        ] {
            assert!(is_chapter_number_line(s), "should match: {s:?}");
        }
    }

    #[test]
    fn chapter_komejirushi_wrap() {
        for s in [
            "※一※", "※二※", "※三※", "※下※", "※上※", "※中※",
            "　　　　　　※五※",
        ] {
            assert!(is_chapter_number_line(s), "should match: {s:?}");
        }
    }

    #[test]
    fn chapter_decorative_separator() {
        for s in [
            "　　　　　＊　　　　　＊　　　　　＊",
            "　　　　　＊　　　　　＊",                   // 装飾文字 A は 2 個でも区切り
            "　　　　　×　　　　×　　　　×",
            "　　　　　――――――――――――",
            "＊＊＊",
            "＊＊",
            "***",
            "**",
            "××",
            "―――",
            "──────",
            "＊　×　＊", // mixed A
            "※　※　※", // ※ space ※ space ※
            "※※",       // ※ 2 個
        ] {
            assert!(is_chapter_number_line(s), "should match: {s:?}");
        }
    }

    #[test]
    fn strip_editorial_annotations_removes_type_b() {
        let cases = [
            ("四月九日〔以下空白〕", "四月九日"),
            ("一九二六、五、一九、〔以下空白〕", "一九二六、五、一九、"),
            ("おや〔一字不明〕、川へはいっちゃいけないったら。", "おや、川へはいっちゃいけないったら。"),
            ("　ただ林の濶い木の葉がぱちぱち鳴っている〔以下原稿数枚？なし〕", "　ただ林の濶い木の葉がぱちぱち鳴っている"),
            ("「そうです、先生。」〔以下原稿数枚なし〕", "「そうです、先生。」"),
            ("〔冒頭原稿数枚焼失〕本文がはじまる。", "本文がはじまる。"),
            ("一千九百二十六年三月廿〔一字分空白〕日、", "一千九百二十六年三月廿日、"),
        ];
        for (input, expected) in cases {
            let (out, n) = strip_editorial_annotations(input);
            assert!(n >= 1, "should remove at least 1 from {input:?}");
            assert_eq!(out, expected, "input={input:?}");
        }
    }

    #[test]
    fn strip_editorial_annotations_keeps_type_a() {
        let cases = [
            "もう 〔natu:rlich〕 なのですね",
            "心の中で Elle est 〔retrouve'e〕! ―― Quoi?",
            "〔Theatron, Orche^stra, Ske^ne^, Proske^nion〕",
            "〔ve'rite' vraie.〕 なんでも事実でなければ",
            "「〔Keine Bru:cke fu:hrt von Mensch zu Mensch.〕（人から人へ掛け渡す橋はない）」",
            "〔二十分停車〕と時計の下に書いてありました。", // 看板内容 (本文)
            "〔ほう。戻れ。ほう。〕",                          // 台詞 (本文)
        ];
        for input in cases {
            let (out, n) = strip_editorial_annotations(input);
            assert_eq!(n, 0, "should NOT remove from {input:?}");
            assert_eq!(out, input);
        }
    }

    #[test]
    fn body_text_must_not_match() {
        for s in [
            "上を向いて歩こう",
            "下りて来た",
            "中の事情",
            "これは本文の一部である。",
            "(1)",   // 全角数字ではないので パターン 14 にもマッチしない
            "「上」と書かれていた",
            "※印は脚注を意味する",
            "上下",
            "上中下",
            "※とは",
            "――", // em-dash 2 文字 (本文中の dash 表現は許容)
            "―",  // em-dash 1 文字
            "8×6=48",
            "「これは＊である」",
            "夏目漱石は明治に活躍した作家である。",
            "百",   // 単独「百」は本文 (100 を意味する用法)
            "千",   // 単独「千」も本文用法あり
            "百年",  // 「百」「千」は単独以外でも一-九/十/〇 を含まないとマッチしない
            "千万",
            "　　　二人の旅人が現れた。", // 句読点を含むので章番号扱いしない
            "　二郎",                    // leading whitespace 1 つでは本文段落扱い
            "第一次世界大戦",             // chars.len()=7 > 4 → 本文扱い
            "ＡとＢ",                    // 2 文字以上の Ａ-Ｚ → 本文扱い
            "Ａ社",                      // Ａ + 漢字 → 本文扱い
            "第一場と同じ日。",           // 末尾句読点 → 本文ト書き
            "　　　一、金　二両　山椒皮　一俵", // タイトル内に全角空白 → 帳簿 (本文)
            "〇、〇〇〇七六粍",          // 一-九/十 の数字を含まない → 本文 (数値表記)
            "そのため、彼は",             // 「その + 漢数字以外」 は本文
            "そのこと",                  // 「その + 漢数字以外」 は本文
        ] {
            assert!(!is_chapter_number_line(s), "should NOT match: {s:?}");
        }
    }
}
