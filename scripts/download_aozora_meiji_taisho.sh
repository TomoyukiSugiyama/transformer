#!/usr/bin/env bash
# 青空文庫から明治-大正期の主要作家 6 名の 新字新仮名 作品を一括取得し、
# ルビ・編集注記等を除去して 1 ファイルに連結する。 漱石コーパス (1.08M char) の
# 3-4 倍規模 (推定 3.5-4.5M char) を目指す Phase 5-3 用コーパス。
#
# 出力: corpus/aozora_meiji_taisho.txt
#       各作品の前に "===== <作家名>『<タイトル>』 =====" のヘッダ行を入れる
#
# 含まれる作家 (全て 新字新仮名 のみ抽出):
#   - 太宰治 (id=000035)   ~203 作品
#   - 国木田独歩 (id=000038)~35 作品
#   - 宮沢賢治 (id=000081)  ~87 作品 (URL なし 1 件除外)
#   - 中島敦 (id=000119)    ~25 作品
#   - 森鴎外 (id=000129)    ~71 作品 (新字旧仮名や旧字旧仮名は対象外)
#   - 夏目漱石 (id=000148)  ~84 作品 (既存 aozora_soseki_works.txt の 7 作品の superset)
#
# 仕組み:
#   1. Aozora 公式 list_person_all_extended_utf8.csv をダウンロード
#   2. 文字遣い種別=新字新仮名 + 指定作家 ID で絞込
#   3. テキストファイル URL から zip を取得・展開
#   4. python3 で本文抽出・ルビ除去・編集注記除去
#
# 必要なコマンド: curl, unzip, python3

set -uo pipefail  # -e は付けない (個別 work の失敗で全体停止しないように)

AUTHOR_IDS="000035 000038 000081 000119 000129 000148"
CSV_URL="https://www.aozora.gr.jp/index_pages/list_person_all_extended_utf8.zip"
CSV_FILE="list_person_all_extended_utf8.csv"
OUT="corpus/aozora_meiji_taisho.txt"

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

mkdir -p corpus

echo "==> downloading Aozora全作品メタデータ CSV"
curl -L --silent --show-error "$CSV_URL" -o "$TMPDIR/list.zip"
unzip -q -o -d "$TMPDIR" "$TMPDIR/list.zip"

# 対象作家・新字新仮名で絞込、 URL 空行は除外
TARGET_LIST="$TMPDIR/target.tsv"  # author_id<TAB>work_id<TAB>title<TAB>surname<TAB>given<TAB>url
python3 - "$TMPDIR/$CSV_FILE" "$TARGET_LIST" "$AUTHOR_IDS" <<'PY'
import csv
import sys

csv_path, out_path, authors_str = sys.argv[1:4]
authors = set(authors_str.split())

with open(csv_path, encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as out:
    reader = csv.reader(f)
    next(reader, None)  # header
    for row in reader:
        if len(row) < 46:
            continue
        work_id = row[0]
        title = row[1]
        kana = row[9]
        pid = row[14]
        surname = row[15]
        given = row[16]
        url = row[45]
        if kana != "新字新仮名":
            continue
        if pid not in authors:
            continue
        if not url:
            continue
        out.write("\t".join([pid, work_id, title, surname, given, url]) + "\n")
PY

TOTAL=$(wc -l < "$TARGET_LIST" | tr -d ' ')
echo "==> $TOTAL 件の作品を取得対象"

: > "$OUT"  # truncate

OK_COUNT=0
FAIL_COUNT=0
TOTAL_CHARS=0

# 作家 ID ごとにグループして処理 (見た目のヘッダーが整う)
while IFS=$'\t' read -r pid work_id title surname given url; do
    slug="${pid}_${work_id}"
    author="${surname}${given}"

    zip_path="$TMPDIR/$slug.zip"
    extract_dir="$TMPDIR/$slug"

    if ! curl -L --silent --show-error --fail "$url" -o "$zip_path" 2>/dev/null; then
        echo "  [!] download failed: ${author}『${title}』 ($url)" >&2
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi

    mkdir -p "$extract_dir"
    if ! unzip -q -o -d "$extract_dir" "$zip_path" 2>/dev/null; then
        echo "  [!] unzip failed: ${author}『${title}』" >&2
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi

    raw="$(find "$extract_dir" -maxdepth 1 -name '*.txt' | head -1)"
    if [[ -z "$raw" ]]; then
        echo "  [!] no .txt extracted: ${author}『${title}』" >&2
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi

    # python3 で本文抽出 → OUT に追記。 戻り値は char 数 (STDOUT に整数 1 行)
    chars=$(python3 - "$raw" "$author" "$title" "$OUT" <<'PY'
import re
import sys

raw_path, author, title, out_path = sys.argv[1:5]

try:
    with open(raw_path, encoding="shift_jis", errors="replace") as f:
        text = f.read()
except Exception as e:
    print(0)
    sys.exit(0)

text = text.replace("\r\n", "\n").replace("\r", "\n")

# 青空文庫プレーンテキスト構造: meta -- header notes -- body -- 底本
parts = re.split(r"^-{20,}\n", text, flags=re.M)
if len(parts) >= 3:
    body = parts[2]
elif len(parts) == 2:
    body = parts[1]
else:
    body = text

m = re.search(r"^底本[：:]", body, flags=re.M)
if m:
    body = body[: m.start()]

# ルビ・編集注記の除去
body = re.sub(r"《[^》]*》", "", body)         # ルビ
body = body.replace("｜", "")                    # ルビ範囲開始マーカー
body = re.sub(r"［＃[^］]*］", "", body)        # 編集注記

# 連続改行を整理
body = re.sub(r"\n{3,}", "\n\n", body).strip()

# 空コンテンツ・極端に短いものはスキップ (~100 char 未満は本文抽出失敗の可能性)
if len(body) < 100:
    print(0)
    sys.exit(0)

with open(out_path, "a", encoding="utf-8") as f:
    f.write(f"===== {author}『{title}』 =====\n\n")
    f.write(body)
    f.write("\n\n")

print(len(body))
PY
)

    if [[ "$chars" -gt 0 ]]; then
        OK_COUNT=$((OK_COUNT + 1))
        TOTAL_CHARS=$((TOTAL_CHARS + chars))
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo "  [!] parsing yielded empty body: ${author}『${title}』" >&2
    fi

    rm -rf "$extract_dir" "$zip_path"

    # 進捗 (100 件ごと)
    if (( (OK_COUNT + FAIL_COUNT) % 100 == 0 )); then
        echo "  ... processed $((OK_COUNT + FAIL_COUNT))/$TOTAL works, current chars=$TOTAL_CHARS"
    fi
done < "$TARGET_LIST"

echo "==> done"
echo "  works: ok=$OK_COUNT fail=$FAIL_COUNT total=$TOTAL"
TOTAL_BYTES=$(wc -c < "$OUT")
echo "  wrote $OUT: $TOTAL_CHARS chars ($TOTAL_BYTES bytes)"
ls -la "$OUT"
